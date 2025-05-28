# 将词保存成集合形式
def getWord(str):
    list = []
    s = 0
    for word in str.split("/"):
        e = s + len(word)
        list.append((s, e - 1))
        s = e
    return set(list)

# ori是原来的分词，pred是系统预测的输出
def evaluate(ori,pred):
    predSize = len(getWord(pred))
    oriSize = len(getWord(ori))
    rightSize = len(getWord(ori) & getWord(pred))
    # Recall
    R = rightSize/oriSize
    # Precision
    P = rightSize/predSize
    # F-measure
    F = 2*P*R/(P+R)
    return R, P, F

pred_1 = "我/来到/北京/清华/大学"
pred_2 = "我/来到/北京/清华大学"
ori = "我/来到/北京/清华大学"

print(evaluate(pred_1,ori))
print(evaluate(pred_2,ori))
