import math
import re
import warnings
from pathlib import Path

import jieba
import numpy as np
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings

warnings.filterwarnings('ignore')
pos_path = "实验四数据\\htl_del_4000\\pos"
neg_path = "实验四数据\\htl_del_4000\\neg"
pos = Path(pos_path)
neg = Path(neg_path)
pos_list = []
neg_list = []
for file in pos.rglob('*.txt'):
    try:
        with open(file, 'r', encoding="GBK") as f:
            content = f.read()
            pos_list.append(content)
    except:
        continue
for file in neg.rglob('*.txt'):
    try:
        with open(file, 'r', encoding="GBK") as f:
            content = f.read()
            neg_list.append(content)
    except:
        continue
for i, item in enumerate(pos_list):
    item = item.replace("\n", '')
    item = re.sub(r"[a-zA-Z\s，。（）*+-/“”\[\]《》：！；——’、·*『』?？><:：]+", "", item)
    pos_list[i] = list(jieba.cut(item, cut_all=False))

for i, item in enumerate(neg_list):
    item = item.replace("\n", '')
    item = re.sub(r"[a-zA-Z\s，。（）*+-/“”\[\]《》：！；——’、·*『』?？><:：]+", "", item)
    neg_list[i] = list(jieba.cut(item, cut_all=False))

all_list = pos_list + neg_list


def train_and_save_models(sentences):
    cbow_model = Word2Vec(
        sentences=sentences,
        vector_size=64,
        window=5,
        min_count=1,
        sg=0,
        workers=4
    )
    cbow_model.save("cbow_model.bin")
    print("CBOW模型保存完成")
    # 训练Skip-gram模型
    sg_model = Word2Vec(
        sentences=sentences,
        vector_size=64,
        window=5,
        min_count=1,
        sg=1,
        workers=4
    )
    sg_model.save("sg_model.bin")
    print("Skip-gram模型保存完成")


def load_model_and_get_vector(model_path, word):
    model = Word2Vec.load(model_path)
    return model.wv[word]


def generate_sentence_vectors(all_sentences, pos_count):
    corpus = [' '.join(sent) for sent in all_sentences]
    tfidf_vectorizer = TfidfVectorizer(
        tokenizer=lambda x: x.split(),
        lowercase=False,
        norm=None
    )
    tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)
    vocab = tfidf_vectorizer.vocabulary_
    for model_name in ["cbow_model.bin", "sg_model.bin"]:
        model = Word2Vec.load(model_name)
        output_file = model_name.replace(".bin", "_results.txt")

        with open(output_file, "w", encoding="utf-8") as f:
            for idx, sentence in enumerate(all_sentences):
                row = tfidf_matrix[idx].toarray()[0]
                total_weight = 0
                for item in row:
                    total_weight += math.e ** item
                sentence_vec = np.zeros(model.vector_size)
                if total_weight > 0:
                    for word in sentence:
                        word_idx = vocab.get(word, -1)
                        if word_idx == -1:
                            continue
                        weight = math.e ** row[word_idx] / total_weight
                        try:
                            sentence_vec += model.wv[word] * weight
                        except KeyError:
                            pass
                label = 1 if idx < pos_count else 0
                f.write(f"{idx} {label}\t{' '.join(map(str, sentence_vec))}\n")


train_and_save_models(all_list)
test_word = "不好"
cbow_vector = load_model_and_get_vector("cbow_model.txt", test_word)
print(f"CBOW'{test_word}'的词向量；\n{cbow_vector}")
sg_vector = load_model_and_get_vector("sg_model.txt", test_word)
print(f"Skip-gram'{test_word}'的词向量；\n{sg_vector}")
generate_sentence_vectors(all_list, len(pos_list))  # 需要正样本实际数量
model = Word2Vec.load("cbow_model.txt")
print(f"'{test_word}'的向量表示：{model.wv[test_word]}")
with open('W2V.txt', 'w', encoding="GBK") as f:
    for items in all_list:
        for item in items:
            f.write(f"{item} {' '.join(map(str, model.wv[item]))}\n")

