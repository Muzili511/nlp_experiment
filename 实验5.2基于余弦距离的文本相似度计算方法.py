import math
import numpy as np
from pathlib import Path
import jieba
import re

# 基础配置
FEATURE_WORDS_PATH = '4.2.txt'
STOP_WORDS_PATH = '实验四数据\\cn_stopwords.txt'
CORPUS_VECTORS_PATH = 'sentence_vectors.txt'

# 全局变量缓存
feature_words = []  # 特征词表
idf_dict = {}  # IDF字典
corpus_vectors = []  # 语料库向量
stop_words = set()  # 停用词


# ---------- 初始化加载 ----------
def load_resources():
    """加载所有必要资源"""
    global feature_words, stop_words, idf_dict, corpus_vectors

    # 加载特征词表
    with open(FEATURE_WORDS_PATH, 'r') as f:
        feature_words = [line.strip() for line in f]

    # 加载停用词
    with open(STOP_WORDS_PATH, 'r', encoding='utf-8') as f:
        stop_words = set(f.read().splitlines())

    # 加载语料库向量
    corpus_vectors = []
    with open(CORPUS_VECTORS_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            vector = list(map(float, parts[1].split()))
            corpus_vectors.append(np.array(vector))

    # 加载IDF值
    with open('idf_values.txt', 'r', encoding='utf-8') as f:
        for line in f:
            try:
                word, idf = line.strip().split(':')
                idf_dict[word] = float(idf)
            except:
                word, idf = ":", 3.6116825361e+00
                idf_dict[word] = float(idf)


# ---------- 文本预处理 ----------
def preprocess_text(text):
    """文本预处理流程"""
    # 清洗文本
    text = re.sub(r"[a-zA-Z\s，。（）*+-/“”\[\]《》：！；——’、·*『』?？]+", "", text)
    # 分词
    words = list(jieba.cut(text))
    # 去除停用词
    return [word for word in words if word not in stop_words]


# ---------- TF-IDF向量生成 ----------
def text_to_vector(words):
    """将分词后的文本转换为TF-IDF向量"""
    vector = [0] * len(feature_words)
    word_counts = {}
    total_words = len(words)

    # 统计词频
    for word in words:
        if word in feature_words:
            word_counts[word] = word_counts.get(word, 0) + 1e-10

    # 计算TF-IDF
    for i, word in enumerate(feature_words):
        if word in word_counts:
            tf = word_counts[word] / total_words
            idf = idf_dict.get(word, 0)
            vector[i] = tf * idf
    return vector


# ---------- 相似度计算 ----------
def cosine_similarity(vec_a, vec_b):
    """计算余弦相似度"""
    dot_product = np.dot(vec_a, vec_b)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 0
    return dot_product / (norm_a * norm_b)


def find_top3_similar(query_vector):
    """查找最相似的三篇文章"""
    similarities = []
    for idx, vec in enumerate(corpus_vectors):
        sim = cosine_similarity(query_vector, vec)
        similarities.append((idx, sim))

    # 按相似度降序排序
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:3]


# ---------- 主流程 ----------
def main():
    # 初始化加载资源
    load_resources()

    while True:
        input_text = input("\n请输入要查询的文本（输入q退出）: ")
        if input_text.lower() == 'q':
            break

        # 预处理文本
        processed_words = preprocess_text(input_text)
        if not processed_words:
            print("文本无效，请重新输入")
            continue

        # 生成向量
        query_vector = text_to_vector(processed_words)

        # 查找相似文章
        top3 = find_top3_similar(query_vector)

        # 输出结果
        print("\n相似度最高的三篇文章：")
        for rank, (idx, sim) in enumerate(top3, 1):
            print(f"第{rank}名 - 文章ID: {idx} 相似度: {sim:.4f}")


if __name__ == "__main__":
    main()