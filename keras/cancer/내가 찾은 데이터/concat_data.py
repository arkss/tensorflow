import pandas as pd
from pandas import DataFrame

data1 = pd.read_excel('data/test1.xlsx')
data2 = pd.read_excel('data/test2.xlsx')

data1_list = data1.values.tolist()
data2_list = data2.values.tolist()

data1_char_list = []
data2_char_list = []

for data in data1_list:
    data1_char_list.append(data[0])

for data in data2_list:
    data2_char_list.append(data[0])

# 교집합 구하기
intersect_char_list = [data for data in data1_char_list if data in data2_char_list]


# 각 데이터에서 교집합이 아닌 부분 제거
for data in data1_list:
    if not data[0] in intersect_char_list:
        data1_list.remove(data)

for data in data2_list:
    if not data[0] in intersect_char_list:
        data2_list.remove(data)

# print(data1_list)
# print(data2_list)

# 두 데이터 합치기
result_list = []
for data1 in data1_list:
    for data2 in data2_list:
        if data1[0] == data2[0]:
            result_list.append(data1+data2[1:])

# transpose
transpose_result_list = [list(i) for i in zip(*result_list)]

print(transpose_result_list)

result_dict = {}
for i,transpose_result in enumerate(transpose_result_list):
    result_dict['col'+str(i)] = transpose_result

df = DataFrame(result_dict)

writer = pd.ExcelWriter('test.xlsx', engine='xlsxwriter')

df.to_excel(writer, sheet_name='sheet1')

writer.close()

print("완료")