import re
from collections import defaultdict
from pathlib import Path
import jieba

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
S = set()
for items in neg_list:
    for item in items:
        S.add(item)
for items in pos_list:
    for item in items:
        S.add(item)
total_dict = defaultdict(int)
for item in S:
    for items in pos_list:
        if item in items:
            total_dict[item] += 1
    for items in neg_list:
        if item in items:
            total_dict[item] += 1
total_list = list(total_dict.items())
sorted_total_list = sorted(total_list, key=lambda x: x[1], reverse=True)
print(sorted_total_list[:1000])
with open('4.1.txt', 'w') as f:
    for item in sorted_total_list[:1000]:
        f.write(item[0]+"\n")