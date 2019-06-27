import numpy as np
import pandas as pd
from random import *

all_data = pd.read_csv("./total_data_13_500.csv" )

print(all_data)

all_data_list = all_data.values.tolist()








# 각 cancer 의 개수를 저장할 list
count_list = [0 for i in range(14)]

# 각 cancer의 개수 카운팅
for row in  all_data_list:
    count_list[int(row[0])] += 1



index = 0
for i in range(1,14):
    # print(i)
    for j in range(count_list[i],501):
        random_rate = uniform(0.9, 1.1)
        # print(random_rate)

        generate_data_list = []
        generate_data_list.append(i)
        for k,data in enumerate(all_data_list[index]):
            if k == 0:
                continue
            generate_data_list.append(data*random_rate)
            all_data.append(generate_data_list)
    
    index += count_list[i]

print(all_data)
            
        
    





