import os
import jieba
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
from collections import Counter
import pickle
import time
torch.manual_seed(42)
np.random.seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# 1. 数据预处理与分词
def load_and_process_data(pos_folder, neg_folder):
    """加载数据并进行分词处理"""
    texts = []
    labels = []

    # 加载正面评价数据
    for file in os.listdir(pos_folder):
        with open(os.path.join(pos_folder, file), 'r', encoding='GBK', errors='ignore') as f:
            content = f.read().strip()
            words = jieba.cut(content)
            texts.append(' '.join(words))
            labels.append(1)  # 正面评价标签为1

    # 加载负面评价数据
    for file in os.listdir(neg_folder):
        with open(os.path.join(neg_folder, file), 'r', encoding='GBK', errors='ignore') as f:
            content = f.read().strip()
            words = jieba.cut(content)
            texts.append(' '.join(words))
            labels.append(0)  # 负面评价标签为0

    return texts, labels


# 2. 构建词汇表
def build_vocab(texts, max_vocab_size=20000):
    word_counts = Counter()
    for text in texts:
        words = text.split()
        word_counts.update(words)
    sorted_vocab = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
    vocab = {word: idx + 2 for idx, (word, count) in enumerate(sorted_vocab[:max_vocab_size - 2])}
    vocab['<PAD>'] = 0  # 填充标记
    vocab['<UNK>'] = 1  # 未知词标记

    return vocab


def text_to_sequence(text, vocab):
    words = text.split()
    sequence = [vocab.get(word, vocab['<UNK>']) for word in words]
    return sequence


class TextDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len=200):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        # 将文本转换为索引序列
        sequence = text_to_sequence(text, self.vocab)

        # 截断或填充序列
        if len(sequence) > self.max_len:
            sequence = sequence[:self.max_len]
        else:
            sequence = sequence + [self.vocab['<PAD>']] * (self.max_len - len(sequence))

        return {
            'text': torch.tensor(sequence, dtype=torch.long),
            'label': torch.tensor(label, dtype=torch.long)
        }


class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers,
                 bidirectional, dropout, use_pretrained=False, embedding_matrix=None):
        super().__init__()

        # 嵌入层
        if use_pretrained and embedding_matrix is not None:
            self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=False)
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # LSTM层
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=n_layers,
            bidirectional=bidirectional,
            dropout=dropout if n_layers > 1 else 0,
            batch_first=True
        )

        # 全连接层
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Linear(lstm_output_dim, output_dim)

        # Dropout层
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        embedded = self.dropout(self.embedding(text))
        output, (hidden, cell) = self.lstm(embedded)
        if self.lstm.bidirectional:
            hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        else:
            hidden = self.dropout(hidden[-1, :, :])
        return self.fc(hidden)


# 训练函数
def train(model, iterator, optimizer, criterion):
    model.train()
    epoch_loss = 0
    epoch_acc = 0

    for batch in iterator:
        optimizer.zero_grad()

        texts = batch['text'].to(device)
        labels = batch['label'].to(device)

        predictions = model(texts).squeeze(1)

        loss = criterion(predictions, labels)

        acc = accuracy_score(labels.cpu().numpy(), predictions.argmax(dim=1).cpu().numpy())

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


# 评估函数
def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    epoch_acc = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in iterator:
            texts = batch['text'].to(device)
            labels = batch['label'].to(device)

            predictions = model(texts).squeeze(1)

            loss = criterion(predictions, labels)

            pred_labels = predictions.argmax(dim=1).cpu().numpy()
            true_labels = labels.cpu().numpy()

            acc = accuracy_score(true_labels, pred_labels)

            epoch_loss += loss.item()
            epoch_acc += acc

            all_preds.extend(pred_labels)
            all_labels.extend(true_labels)

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='binary', zero_division=0
    )

    return epoch_loss / len(iterator), epoch_acc / len(iterator), precision, recall, f1


# 加载预训练词向量
def load_pretrained_embeddings(embedding_path, vocab, embedding_dim=300):
    """加载预训练词向量并构建嵌入矩阵"""
    word_vectors = {}
    print(f"Loading pretrained embeddings from {embedding_path}...")
    with open(embedding_path, 'r', encoding='GBK') as f:
        for i, line in enumerate(f):
            if i == 0 and len(line.split()) == 2:
                continue

            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            if word in vocab:
                word_vectors[word] = vector

    print(f"Found {len(word_vectors)} words in pretrained embeddings.")
    embedding_matrix = np.zeros((len(vocab), embedding_dim))

    for word, idx in vocab.items():
        if word in word_vectors:
            embedding_matrix[idx] = word_vectors[word]
        elif word == '<PAD>':
            embedding_matrix[idx] = np.zeros(embedding_dim)
        else:
            # 使用随机初始化，但保持一致性
            embedding_matrix[idx] = np.random.normal(scale=0.6, size=(embedding_dim,))

    return torch.tensor(embedding_matrix, dtype=torch.float)


# 主函数
def main():
    print("Loading and processing data...")
    pos_folder = '.\\实验四数据\\htl_del_4000\\pos'
    neg_folder = '.\\实验四数据\\htl_del_4000\\neg'
    texts, labels = load_and_process_data(pos_folder, neg_folder)
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, test_size=0.2, stratify=labels, random_state=42
    )
    vocab = build_vocab(train_texts)
    vocab_size = len(vocab)
    print(f"Vocabulary size: {vocab_size}")
    max_len = 200
    train_dataset = TextDataset(train_texts, train_labels, vocab, max_len)
    test_dataset = TextDataset(test_texts, test_labels, vocab, max_len)
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # 实验8.1: 训练不带预训练词向量的LSTM模型
    print("\nExperiment 8.1: Training LSTM without pretrained embeddings...")
    embedding_dim = 128
    hidden_dim = 256
    output_dim = 2
    n_layers = 2
    bidirectional = True
    dropout = 0.5
    lr = 0.001
    n_epochs = 15
    model = LSTMClassifier(vocab_size, embedding_dim, hidden_dim, output_dim,
                           n_layers, bidirectional, dropout).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    best_f1 = 0
    for epoch in range(n_epochs):
        start_time = time.time()
        train_loss, train_acc = train(model, train_loader, optimizer, criterion)
        valid_loss, valid_acc, valid_precision, valid_recall, valid_f1 = evaluate(model, test_loader, criterion)
        end_time = time.time()
        epoch_mins, epoch_secs = divmod(end_time - start_time, 60)
        if valid_f1 > best_f1:
            best_f1 = valid_f1
            torch.save(model.state_dict(), 'lstm_best_model.pt')

        print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins:.0f}m {epoch_secs:.0f}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
        print(f'\t Val. Loss: {valid_loss:.3f} | Val. Acc: {valid_acc * 100:.2f}%')
        print(f'\t Val. Precision: {valid_precision:.3f} | Val. Recall: {valid_recall:.3f} | Val. F1: {valid_f1:.3f}')
    with open('vocab.pkl', 'wb') as f:
        pickle.dump(vocab, f)

    # 实验8.2: 使用预训练词向量
    print("\nExperiment 8.2: Training LSTM with pretrained embeddings...")
    embedding_path = 'W2V.txt'
    embedding_dim_pretrained = 64
    embedding_matrix = load_pretrained_embeddings(embedding_path, vocab, embedding_dim_pretrained)
    model_pretrained = LSTMClassifier(
        vocab_size,
        embedding_dim_pretrained,
        hidden_dim,
        output_dim,
        n_layers,
        bidirectional,
        dropout,
        use_pretrained=True,
        embedding_matrix=embedding_matrix
    ).to(device)
    optimizer_pretrained = optim.Adam(model_pretrained.parameters(), lr=lr)
    best_f1_pretrained = 0
    for epoch in range(n_epochs):
        start_time = time.time()

        train_loss, train_acc = train(model_pretrained, train_loader, optimizer_pretrained, criterion)
        valid_loss, valid_acc, valid_precision, valid_recall, valid_f1 = evaluate(model_pretrained, test_loader,
                                                                                  criterion)
        end_time = time.time()
        epoch_mins, epoch_secs = divmod(end_time - start_time, 60)
        if valid_f1 > best_f1_pretrained:
            best_f1_pretrained = valid_f1
            torch.save(model_pretrained.state_dict(), 'lstm_pretrained_best_model.pt')
        print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins:.0f}m {epoch_secs:.0f}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
        print(f'\t Val. Loss: {valid_loss:.3f} | Val. Acc: {valid_acc * 100:.2f}%')
        print(f'\t Val. Precision: {valid_precision:.3f} | Val. Recall: {valid_recall:.3f} | Val. F1: {valid_f1:.3f}')

    # 实验8.3: 任意输入句子分类
    print("\nExperiment 8.3: Classifying arbitrary sentences")
    with open('vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)
    model_pretrained.load_state_dict(torch.load('lstm_pretrained_best_model.pt'))
    model_pretrained.eval()
    test_sentences = [
        "这家酒店的服务真的很棒，房间干净舒适，下次还会再来！",
        "糟糕的体验，房间不干净，服务员态度很差，绝对不会再来了。",
        "地理位置不错，但房间设施有些老旧，早餐种类也不多。",
        "性价比很高，虽然不算豪华，但干净整洁，服务也很周到。",
        "简直不能更差！空调坏了，卫生间还有异味，太失望了。"
    ]

    print("\nTesting sentiment classification:")
    for sentence in test_sentences:
        result = predict_sentiment(model_pretrained, sentence, vocab, max_len)
        print(f"\n句子: {result['sentence']}")
        print(f"情感: {result['sentiment']} (置信度: {result['confidence']:.2f})")
        print(f"正面概率: {result['pos_prob']:.4f}, 负面概率: {result['neg_prob']:.4f}")

    # 用户交互模式
    print("\nEntering interactive mode. Type 'exit' to quit.")
    while True:
        user_input = input("\n请输入一句话进行情感分析: ")
        if user_input.lower() in ['exit', 'quit', '退出']:
            break

        result = predict_sentiment(model_pretrained, user_input, vocab, max_len)
        print(f"\n分析结果: {result['sentiment']} (置信度: {result['confidence']:.2f})")
        print(f"正面概率: {result['pos_prob']:.4f}, 负面概率: {result['neg_prob']:.4f}")
        print(f"详细: {result['sentence']}")


def predict_sentiment(model, sentence, vocab, max_len=200):
    """预测输入句子的情感"""
    # 分词
    words = jieba.cut(sentence)
    text = ' '.join(words)

    # 转换为序列
    sequence = text_to_sequence(text, vocab)

    # 截断或填充序列
    if len(sequence) > max_len:
        sequence = sequence[:max_len]
    else:
        sequence = sequence + [vocab['<PAD>']] * (max_len - len(sequence))

    # 转换为张量
    tensor = torch.tensor(sequence).unsqueeze(0).to(device)  # [1, seq_len]

    # 预测
    with torch.no_grad():
        output = model(tensor)
        prediction = torch.softmax(output, dim=1)
        pos_prob = prediction[0][1].item()
        neg_prob = prediction[0][0].item()

        sentiment = "pos" if pos_prob > 0.5 else "neg"

        # 返回详细结果
        result = {
            "sentence": sentence,
            "sentiment": sentiment,
            "pos_prob": pos_prob,
            "neg_prob": neg_prob,
            "confidence": max(pos_prob, neg_prob)
        }
        return result


if __name__ == "__main__":
    main()
