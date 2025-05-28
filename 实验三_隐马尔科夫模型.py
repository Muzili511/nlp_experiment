import math
import re
import random
from collections import defaultdict


def read_corpus(file_path):
    sentences = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # 处理复合结构（如 "[中央/n 人民/n]nt"）
            processed_line = re.sub(r'\[(.*?)\]/\w+', lambda m: m.group(1), line)
            tokens = processed_line.split()
            sentence = []
            for token in tokens:
                if '/' in token:
                    word, pos = token.rsplit('/', 1)
                    sentence.append((word, pos))
                else:
                    sentence.append((token, 'UNK'))  # 未知词性标记
            sentences.append(sentence)
    return sentences


def split_data(sentences, test_ratio=0.2):
    random.seed(42)
    random.shuffle(sentences)
    split_idx = int(len(sentences) * (1 - test_ratio))
    return sentences[:split_idx], sentences[split_idx:]


# 初始概率
def compute_initial_prob(training_sentences):
    initial_counts = defaultdict(int)
    total = 0
    for sentence in training_sentences:
        if sentence:
            first_pos = sentence[0][1]
            initial_counts[first_pos] += 1
            total += 1
    return {pos: count / total for pos, count in initial_counts.items()}


# 转移概率
def compute_transition_prob(training_sentences):
    trans_counts = defaultdict(lambda: defaultdict(int))
    pos_counts = defaultdict(int)
    for sentence in training_sentences:
        pos_seq = [pos for _, pos in sentence]
        for i in range(len(pos_seq) - 1):
            current, next_pos = pos_seq[i], pos_seq[i + 1]
            trans_counts[current][next_pos] += 1
            pos_counts[current] += 1
    trans_prob = defaultdict(dict)
    for current in trans_counts:
        for next_pos in trans_counts[current]:
            trans_prob[current][next_pos] = trans_counts[current][next_pos] / pos_counts[current]
    return trans_prob


# 发射概率
def compute_emission_prob(training_sentences):
    emit_counts = defaultdict(lambda: defaultdict(int))
    pos_counts = defaultdict(int)
    vocab = set()
    for sentence in training_sentences:
        for word, pos in sentence:
            emit_counts[pos][word] += 1
            pos_counts[pos] += 1
            vocab.add(word)
    V = len(vocab)
    emit_prob = defaultdict(dict)
    for pos in emit_counts:
        for word in emit_counts[pos]:
            emit_prob[pos][word] = (emit_counts[pos][word] + 1) / (pos_counts[pos] + V)
    return emit_prob, pos_counts, V


# viterbi算法
def viterbi(sentence_words, initial_prob, trans_prob, emit_prob, pos_counts, V, all_pos):
    n = len(sentence_words)
    if n == 0:
        return []

    # 使用对数防止下溢
    log = lambda x: math.log(x) if x > 0 else -float('inf')

    dp = [{} for _ in range(n)]
    path = {}

    # 初始化
    first_word = sentence_words[0]
    for pos in all_pos:
        init_p = initial_prob.get(pos, 0)
        emit_p = emit_prob[pos].get(first_word, 1 / (pos_counts.get(pos, 1) + V))
        dp[0][pos] = log(init_p) + log(emit_p)
        path[pos] = [pos]

    # 递推
    for t in range(1, n):
        new_path = {}
        for curr_pos in all_pos:
            max_logp = -float('inf')
            best_prev = None
            for prev_pos in all_pos:
                logp = dp[t - 1].get(prev_pos, -float('inf'))
                trans_p = trans_prob.get(prev_pos, {}).get(curr_pos, 0)
                logp += log(trans_p) if trans_p > 0 else -float('inf')
                if logp > max_logp:
                    max_logp = logp
                    best_prev = prev_pos
            emit_p = emit_prob[curr_pos].get(sentence_words[t], 1 / (pos_counts.get(curr_pos, 1) + V))
            dp[t][curr_pos] = max_logp + log(emit_p) if emit_p > 0 else -float('inf')
            new_path[curr_pos] = path.get(best_prev, []) + [curr_pos]
        path = new_path

    # 回溯最优路径
    max_logp = max(dp[-1].values(), default=-float('inf'))
    best_pos = [pos for pos, val in dp[-1].items() if val == max_logp]
    return path[best_pos[0]] if best_pos else []


# 评价程序
def evaluate(test_sentences, initial_prob, trans_prob, emit_prob, pos_counts, V, all_pos):
    correct = 0
    total = 0
    for sentence in test_sentences:
        words = [word for word, _ in sentence]
        true_pos = [pos for _, pos in sentence]
        pred_pos = viterbi(words, initial_prob, trans_prob, emit_prob, pos_counts, V, all_pos)
        if len(pred_pos) != len(true_pos):
            continue
        correct += sum(1 for t, p in zip(true_pos, pred_pos) if t == p)
        total += len(true_pos)
    return correct / total if total > 0 else 0


def main():
    sentences = read_corpus('人民日报语料.txt')
    train, test = split_data(sentences)  # train和test中每句话都是元组形式，（词，词性）
    initial_prob = compute_initial_prob(train)  # 计算每句话中第一个词的初始概率
    trans_prob = compute_transition_prob(train)
    emit_prob, pos_counts, V = compute_emission_prob(train)
    all_pos = list(pos_counts.keys())
    accuracy = evaluate(test, initial_prob, trans_prob, emit_prob, pos_counts, V, all_pos)
    print(f"词性标注准确率: {accuracy:.2%}")


if __name__ == '__main__':
    main()
