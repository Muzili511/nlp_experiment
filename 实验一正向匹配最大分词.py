with open("词库.txt", 'r') as f:
    data = f.read()
    data_list = data.split("\n")
    cidian_list = []
    for item in data_list:
        if item:
            cidian_list.append(item)

# target = input("请输入您的所需要分词的句子")
target = "他是研究生物化学的"
i = 0
n = len(target) - 1
cidian_max_length = 0
slice_position = []
for item in cidian_list:
    cidian_max_length = max(cidian_max_length, len(item))
while n != 1:
    m = cidian_max_length
    if n < m:
        m = n
    wi = target[i: i+m+1]
    if wi in cidian_list:
        slice_position.append(i + m)
        if i == len(target) - 1:
            break
        else:
            i += m
            n -= m
    flag = False
    while wi not in cidian_list and len(wi) > 1:
        wi = wi[:-1]
        if wi in cidian_list:
            flag = True
            slice_position.append(i + len(wi))
            if i == len(target) - 1:
                break
            else:
                i += len(wi)
                n -= len(wi)
    if len(wi) == 1 and not flag:
        slice_position.append(i + len(wi))
        cidian_list.append(wi)
        if i == len(target) - 1:
            break
        else:
            i += len(wi)
            n -= len(wi)
string_list = []
for item in target:
    string_list.append(item)
for i in range(len(slice_position)):
    string_list.insert(slice_position[i], '/')
    for i in range(len(slice_position)):
        slice_position[i] += 1
print("".join(string_list))
