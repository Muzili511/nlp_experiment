import re
from collections import defaultdict
from pathlib import Path
import jieba
import math
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
    item = re.sub(r"[a-zA-Z\s，。（）*+-/“”\[\]《》：！；——’、·*『』?？]+", "", item)
    item = list(jieba.cut(item, cut_all=False))
    pos_list[i] = item
for i, item in enumerate(neg_list):
    item = item.replace("\n", '')
    item = re.sub(r"[a-zA-Z\s，。（）*+-/“”\[\]《》：！；——’、·*『』?？]+", "", item)
    item = list(jieba.cut(item, cut_all=False))
    neg_list[i] = item
with open('实验四数据\\cn_stopwords.txt', 'r', encoding='utf-8') as f:
    stop_list = set(f.read().splitlines())

for i in range(len(pos_list)):
    pos_list[i] = [word for word in pos_list[i] if word not in stop_list]

for i in range(len(neg_list)):
    neg_list[i] = [word for word in neg_list[i] if word not in stop_list]

# 读取特征词
feature_words_path = '4.2.txt'
with open(feature_words_path, 'r') as f:
    feature_words = [line.strip() for line in f]

# 合并所有文档
all_docs = pos_list + neg_list
doc_sets = [set(doc) for doc in all_docs]

# 计算文档频率
doc_freq = {}
for word in feature_words:
    count = 0
    for doc_set in doc_sets:
        if word in doc_set:
            count += 1
    doc_freq[word] = count

# 计算IDF值
N = len(all_docs)
idf_dict = {}
for word in feature_words:
    df = doc_freq.get(word, 0)
    idf_dict[word] = math.log(N / (df + 1e-10))

# 生成所有句子的向量表示（先负面评论后正面评论）
sentences = neg_list + pos_list

# 写入输出文件
with open('sentence_vectors.txt', 'w', encoding='utf-8') as f_out:
    for idx, sen in enumerate(sentences):
        vector = []
        for word in feature_words:
            if word in sen:
                # 计算TF-IDF值
                tf = sen.count(word) / len(sen)
                tfidf = tf * idf_dict[word]
            else:
                tfidf = 0.0
            vector.append(f"{tfidf:.6f}")
        # 拼接向量字符串
        vector_str = ' '.join(vector)
        f_out.write(f"{idx}\t{vector_str}\n")

idf_save_path = 'idf_values.txt'  # 可根据需要调整路径
with open(idf_save_path, 'w', encoding='utf-8') as f:
    # 按特征词表顺序保存保证一致性
    for word in feature_words:
        idf_value = idf_dict.get(word, 0.0)
        # 使用科学计数法保存避免精度丢失
        f.write(f"{word}:{idf_value:.10}\n")


print("向量生成完成，已保存到 sentence_vectors.txt")