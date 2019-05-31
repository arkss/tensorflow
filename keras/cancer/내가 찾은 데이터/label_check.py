
import pandas as pd
from pandas import DataFrame

data = pd.read_excel('label/label_3.xlsx')

data_list = data.values.tolist()

clean_data_list = []
count_dict = {}

for data in data_list:
    # 데이터 마다 짤라야 하는 부분이 다르기 때문에 데이터를 확인하고 수정하자.
    clean_data = " ".join(data[0].split()[1:])

    if (clean_data in count_dict.keys()):
        count_dict[clean_data]+=1
    else:
        count_dict[clean_data] = 1
 
    clean_data_list.append(clean_data)

cancer_list = list(count_dict.keys())
cancer_count_list = list(count_dict.values())

df1 = DataFrame({"cancer_name":cancer_list})
df2 = DataFrame({"count":cancer_count_list})

df_c = pd.concat([df1,df2], axis=1)


writer = pd.ExcelWriter("data_count/cancer3.xlsx", engine='xlsxwriter')

df_c.to_excel(writer,sheet_name='sheet1')

writer.close()

print("완료")
