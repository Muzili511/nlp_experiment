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

# 读取正负数据
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

# 清洗文本并分词
for i, item in enumerate(pos_list):
    item = item.replace("\n", '')
    item = re.sub(r"[a-zA-Z\s，。（）*+-/“”\[\]《》：！；——’、·*『』?？><:：]+", "", item)
    pos_list[i] = list(jieba.cut(item, cut_all=False))

for i, item in enumerate(neg_list):
    item = item.replace("\n", '')
    item = re.sub(r"[a-zA-Z\s，。（）*+-/“”\[\]《》：！；——’、·*『』?？><:：]+", "", item)
    neg_list[i] = list(jieba.cut(item, cut_all=False))

# 加载停用词并过滤
with open('实验四数据\\cn_stopwords.txt', 'r', encoding='utf-8') as f:
    stop_list = set(f.read().splitlines())

for i in range(len(pos_list)):
    pos_list[i] = [word for word in pos_list[i] if word not in stop_list]

for i in range(len(neg_list)):
    neg_list[i] = [word for word in neg_list[i] if word not in stop_list]

# 统计文档频次
pos_dict = defaultdict(int)
for doc in pos_list:
    unique_words = set(doc)
    for word in unique_words:
        pos_dict[word] += 1

neg_dict = defaultdict(int)
for doc in neg_list:
    unique_words = set(doc)
    for word in unique_words:
        neg_dict[word] += 1

# 计算互信息
N_pos = len(pos_list)
N_neg = len(neg_list)
N_total = N_pos + N_neg
mi_scores = {}

all_words = set(pos_dict.keys()).union(neg_dict.keys())

for word in all_words:
    A = pos_dict.get(word, 0)
    B = neg_dict.get(word, 0)
    C = N_pos - A
    D = N_neg - B
    # 计算互信息的四个分量
    terms = []
    if A > 0 and (A + B) > 0 and N_pos > 0:
        ratio = (A * N_total) / ((A + B) * N_pos)
        terms.append((A / N_total) * math.log2(ratio))

    if B > 0 and (A + B) > 0 and N_neg > 0:
        ratio = (B * N_total) / ((A + B) * N_neg)
        terms.append((B / N_total) * math.log2(ratio))

    if C > 0 and (C + D) > 0 and N_pos > 0:
        ratio = (C * N_total) / ((C + D) * N_pos)
        terms.append((C / N_total) * math.log2(ratio))

    if D > 0 and (C + D) > 0 and N_neg > 0:
        ratio = (D * N_total) / ((C + D) * N_neg)
        terms.append((D / N_total) * math.log2(ratio))

    mi_scores[word] = sum(terms)
# 按互信息值排序
sorted_mi = sorted(mi_scores.items(), key=lambda x: x[1], reverse=True)

print('前1000个特征词')
for word, score in sorted_mi[:1500]:
    print(f"{word}: {score:.4f}")
with open('4.2.txt', 'w') as f:
    for item in sorted_mi[:1500]:
        f.write(item[0]+"\n")