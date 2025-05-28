import re

import numpy as np
from gensim.models import Word2Vec
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle
from sklearn.metrics import precision_score, recall_score, f1_score
import jieba
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
    model_w2v = Word2Vec.load('sg_model.bin')
    text = jieba.cut(text)
    text_temp = []
    for item in text:
        item = re.sub(r"[a-zA-Z\s，。（）*+-/“”\[\]《》：！；——’、·*『』?？><:：]+", "", item)
        if item == "":
            continue
        text_temp.append(item)
    temp = model_w2v.wv[text_temp]
    # 处理新文本
    prediction = model.predict(temp)
    return "正面" if prediction[0] == 1 else "负面"


# 示例数据集（消极数据为0，积极数据为1）
with open('sg_model_results.txt', 'r', encoding='utf-8') as f:
    lines = f.read().split('\n')
    texts = []
    labels = []
    for line in lines:
        try:
            text = line.split('\t')[1]
            texts.append([float(x) for x in text.split()])
            temp = line.split('\t')[0]
            id = temp.split()[0]
            if int(id) < 2000:
                labels.append(1)
            else:
                labels.append(0)
        except:
            continue
process_experiment("W2V", texts, labels, random_state=4)
# 测试新文本
new_text = "这次体验非常好！"
print(f"\n新文本预测结果（这次体验非常好！）: {predict_new_text(new_text, 'W2V')}")