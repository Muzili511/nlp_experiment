import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle
from sklearn.metrics import precision_score, recall_score, f1_score
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
warnings.filterwarnings("ignore")
def process_experiment(experiment_name, texts, labels, random_state):
    print(f"\n===== 处理实验 {experiment_name} =====")
    shuffled_texts, shuffled_labels = shuffle(texts, labels, random_state=random_state)
    X = shuffled_texts
    y = np.array(shuffled_labels)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression()

    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    print(f"分类正确率: {accuracy:.4f}")

    with open(f'model_{experiment_name}.pkl', 'wb') as f:
        pickle.dump(model, f)

    y_pred = model.predict(X_test)
    print(f"精确率: {precision_score(y_test, y_pred):.4f}")
    print(f"召回率: {recall_score(y_test, y_pred):.4f}")
    print(f"F1值: {f1_score(y_test, y_pred):.4f}")


def predict_new_text(text, experiment_name):
    with open(f'model_{experiment_name}.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('sentence_vectors.txt', 'r', encoding='utf-8') as f:
        texts_new = f.read().split('\n')
        text_list = []
        for text_ in texts_new:
            try:
                text_list.append(text_.split('\t')[1])
            except:
                continue
    vectorizer = TfidfVectorizer()
    vectorizer.fit_transform(text_list)
    # 处理新文本
    tokenized = ' '.join(jieba.cut(text))
    features = vectorizer.transform([tokenized])
    prediction = model.predict(features)
    return "正面" if prediction[0] == 1 else "负面"


# 示例数据集（消极数据为0，积极数据为1）
with open('sentence_vectors.txt', 'r', encoding='utf-8') as f:
    lines = f.read().split('\n')
    texts = []
    labels = []
    for line in lines:
        try:
            text = line.split('\t')[1]
            texts.append([float(x) for x in text.split()])
            id = line.split('\t')[0]
            if int(id) < 2000:
                labels.append(0)
            else:
                labels.append(1)
        except:
            continue
process_experiment("VSM(tf-idf)", texts, labels, random_state=4)
# 测试新文本
new_text = "这次体验非常糟糕！"
print(f"\n新文本预测结果（这次体验非常糟糕！）: {predict_new_text(new_text, 'VSM(余弦距离)')}")