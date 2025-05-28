import re
from collections import defaultdict

with open("人民日报语料.txt", "r") as f:
    data = f.read()
    data_list = data.split('/')
    result_list = []
    for item in data_list:
        result = re.sub(r"[a-zA-Z\s，。（）*+-/“”\[\]《》：！；——’、·*『』?？]+", "", item)
        result_list.append(result)
    result_dict = defaultdict(int)
    for item in result_list:
        if item:
            result_dict[item]+=1
    result_list = [(k, v) for k, v in result_dict.items()]
    result_list = sorted(result_list, key=lambda x: x[1])
    print(f'总词数：{len(result_list)}')
    for item in result_list:
        print(f"{item[0]}:{item[1]}")

with open("词库.txt", "w") as f:
    for item in result_list:
        f.write(f"{item[0]}\n")
with open("词表带数量.txt", 'w') as f:
    for item in result_list:
        f.write(f"{item[0]} {item[1]}\n")