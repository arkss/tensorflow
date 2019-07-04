# 각 데이터를 500개로 맞추는 작업

import numpy as np
import pandas as pd
from random import *

all_data = pd.read_csv("./all_data_500/total_data_13_500.csv" )

all_data_list = all_data.values.tolist()

# 각 cancer 의 개수를 저장할 list
count_list = [0 for i in range(14)]

# 각 cancer의 개수 카운팅
for row in all_data_list:
    count_list[int(row[0])] += 1

# 500개씩의 데이터를 만들어 추가해줄 df
all_data_500 = pd.DataFrame()

copy_index = 0
new_index = 0
for label in range(1,14):
    # label에 해당하는 df을 filtering
    sub_data = all_data[all_data['label'].isin([''+str(label)])]
    print(label, "번째 label 시행 중 ..")
    # 길이가 500이 될 때 까지 반복해준다.
    copy_index += count_list[label-1] # 어느 행을 복사하여 추가할지 고르는 변수
    new_index += count_list[label]
    temp_new_index = new_index
    temp_copy_index = copy_index
    while len(sub_data) != 500:
        random = uniform(0.9, 1.1)
        sub_data.loc[temp_new_index] = sub_data.loc[temp_copy_index] * random
        temp_new_index += 1
        temp_copy_index += 1
        
    all_data_500 = pd.concat([all_data_500, sub_data], ignore_index=True)

print(all_data_500)

    





