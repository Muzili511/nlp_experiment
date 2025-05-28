import math
from collections import defaultdict

with open("词表带数量.txt", 'r') as f:
    temp = '1'
    word_freq = defaultdict(int)
    while temp:
        temp = f.readline().split()
        if len(temp) >= 2:
            word_freq[temp[0]] = int(temp[1])
    print(word_freq)
# 计算词的概率（归一化）
total_freq = sum(word_freq.values())
word_prob = {word: freq / total_freq for word, freq in word_freq.items()}

# 未知词的概率（平滑处理）
unknown_prob = 1e-10

# 最大词长（假设词的最大长度为5）
max_word_length = 5

def viterbi_segment(text):
    n = len(text)
    dp = [ -float('inf') ] * (n + 1)
    dp[0] = 0.0  # 空字符串的概率为1
    prev = [-1] * (n + 1)  # 记录每个位置的最佳左邻词的结束位置

    for i in range(1, n + 1):
        for j in range(max(0, i - max_word_length), i):
            word = text[j:i]
            if word in word_prob:
                current_prob = dp[j] + math.log(word_prob[word])
            else:
                current_prob = dp[j] + math.log(unknown_prob)
            if current_prob > dp[i]:
                dp[i] = current_prob
                prev[i] = j

    # 回溯得到分词结果
    words = []
    i = n
    while i > 0:
        j = prev[i]
        words.append(text[j:i])
        i = j
    words.reverse()
    return words

# 测试输入
input_text = "用户提供的词表和词频统计结果转换为一个字典"
result = viterbi_segment(input_text)
print("输入字串:", input_text)
print("分词结果:", result)