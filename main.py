# import torch
#
# import warnings
# warnings.filterwarnings("ignore")
# from ltp import LTP
# corpus = "19980101-01-001-001迈向充满希望的新世纪。——一九九八年新年讲话附图片张）中共中央总书记国家主席江泽民发表１９９８年新年讲话。"
# ltp = LTP("C:\\Users\\muzili\\PycharmProjects\\自然语言处理实验\\LTP-small\\LTP-small")
# if torch.cuda.is_available():
#     ltp.cuda()
# ltpword_raw = []
# for item in corpus.split("。"):
#     ltpword_raw.append(ltp.pipeline(item, tasks=["cws"], return_dict=False))
# ltpword = []
# for item in ltpword_raw:
#     for nex in item:
#         for tem in nex:
#             ltpword.append(tem)
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle
from sklearn.metrics import precision_score, recall_score, f1_score
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer

# 示例数据集（假设包含中文文本和二分类标签）
texts = [
    "这个手机非常好用，强烈推荐！",
    "质量太差，用了两天就坏了。",
    "服务态度很不好，不会再来了。",
    "非常满意，物超所值。",
    "包装破损，体验很差。"
]
labels = [1, 0, 0, 1, 0]  # 1代表正面，0代表负面


def process_experiment(experiment_name, texts, labels, random_state):
    print(f"\n===== 处理实验 {experiment_name} =====")

    # 1. 随机打乱数据
    shuffled_texts, shuffled_labels = shuffle(texts, labels, random_state=random_state)

    # 分词处理
    def tokenize(text):
        return ' '.join(jieba.cut(text))

    tokenized_texts = [tokenize(text) for text in shuffled_texts]

    # 特征提取
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(tokenized_texts)
    y = np.array(shuffled_labels)

    # 2. 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. 实例化模型
    model = LogisticRegression()

    # 4. 训练模型并评估
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    print(f"分类正确率: {accuracy:.4f}")

    # 5. 保存模型
    with open(f'model_{experiment_name}.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open(f'vectorizer_{experiment_name}.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)

    # 6. 预测并计算指标
    y_pred = model.predict(X_test)
    print(f"精确率: {precision_score(y_test, y_pred):.4f}")
    print(f"召回率: {recall_score(y_test, y_pred):.4f}")
    print(f"F1值: {f1_score(y_test, y_pred):.4f}")


# 执行两个实验（使用不同的随机种子）
process_experiment("五", texts, labels, random_state=5)
process_experiment("六", texts, labels, random_state=6)


# 7. 新文本预测示例
def predict_new_text(text, experiment_name):
    # 加载模型和特征处理器
    with open(f'model_{experiment_name}.pkl', 'rb') as f:
        model = pickle.load(f)
    with open(f'vectorizer_{experiment_name}.pkl', 'rb') as f:
        vectorizer = pickle.load(f)

    # 处理新文本
    tokenized = ' '.join(jieba.cut(text))
    features = vectorizer.transform([tokenized])
    prediction = model.predict(features)
    return "正面" if prediction[0] == 1 else "负面"


# 测试新文本
new_text = "这次购物体验非常糟糕！"
print(f"\n新文本预测结果（实验五）: {predict_new_text(new_text, '五')}")
print(f"新文本预测结果（实验六）: {predict_new_text(new_text, '六')}")