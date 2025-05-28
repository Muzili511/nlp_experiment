import re

import jieba
from collections import defaultdict
import torch

import warnings
warnings.filterwarnings("ignore")
from ltp import LTP
with open('人民日报语料.txt', 'r') as f:
    corpus = f.read()
    corpus = re.sub(r'\d{8}-\d{2}-\d{3}-\d{3}/m\s*', '', corpus)

# 提取原始分词结果（去掉词性标注）
def extract_original_words(line):
    words = []
    for item in line.strip().split():
        if '/' in item:
            word, pos = item.split('/', 1)
            words.append(word)
        else:
            words.append(item)
    return words


# 使用结巴分词
def jieba_segment(line):
    return list(jieba.cut(line))


# 评价分词结果
def evaluate(original_words, segmented_words):
    original_set = set(original_words)
    segmented_set = set(segmented_words)
    tp = len(original_set & segmented_set)
    fp = len(segmented_set - original_set)
    fn = len(original_set - segmented_set)
    # 计算准确率、召回率和 F1 值
    precision = tp / (tp + fp) if (tp + fp) != 0 else 0
    recall = tp / (tp + fn) if (tp + fn) != 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) != 0 else 0

    return precision, recall, f1


if __name__ == "__main__":
    # 提取原始分词结果
    original_words = []
    for line in corpus.strip().split('\n'):
        original_words.extend(extract_original_words(line))

    # 使用结巴和ltp分词
    new_corpus = "".join(original_words)
    segmented_words = jieba_segment(new_corpus)
    ltp = LTP("C:\\Users\\muzili\\PycharmProjects\\自然语言处理实验\\LTP-small\\LTP-small")
    if torch.cuda.is_available():
        ltp.cuda()
    ltpword_raw = []
    for item in new_corpus.split("。"):
        ltpword_raw.append(ltp.pipeline(item, tasks=["cws"], return_dict=False))
    ltpword = []
    for item in ltpword_raw:
        for nex in item:
            for tem in nex:
                ltpword.append(tem)
    # 分词后的结果
    # 评价分词结果
    jieba_precision, jieba_recall, jieba_f1 = evaluate(original_words, segmented_words)
    ltp_precision, ltp_recall, ltp_f1 = evaluate(original_words, ltpword)
    # 输出结果
    print("原始分词结果:", original_words)
    print("结巴分词结果:", segmented_words)
    print("ltp 分词结果:", ltpword)
    print(f"结巴准确率: {jieba_precision:.4f}")
    print(f"结巴召回率: {jieba_recall:.4f}")
    print(f"结巴F1 值: {jieba_f1:.4f}")
    print(f"ltp准确率: {ltp_precision:.4f}")
    print(f"ltp召回率: {ltp_recall:.4f}")
    print(f"ltp F1 值: {ltp_f1:.4f}")
