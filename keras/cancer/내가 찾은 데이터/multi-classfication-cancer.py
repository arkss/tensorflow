import keras
import numpy as np
import pandas as pd 

""" 데이터 불러오기 """
all_data = pd.read_csv("./total_data.csv" )


# label 만 제거만 데이터
data = all_data.drop(columns = ['label'])
# label 만 남은 데이터
label_data = all_data[['label']]

# print(all_data)
# [6237 rows x 2257 columns]

""" train, test data 로 나누기 """
all_data_list = all_data.values.tolist()
data_list = data.values.tolist()

count_list = [0 for i in range(1,24)]


for data_list in all_data_list:
    count_list[int(data_list[0])-1] += 1
    
train_data = []
test_data = []

before_test_index = 0
train_index = 0
test_index = 0

for count in count_list:
    # 0.7 / 0.3 , training / test
    train_index = int(count*0.7) + before_test_index
    test_index = int(count*0.3) + train_index

    train_data += all_data_list[before_test_index:train_index]
    test_data += all_data_list[train_index:test_index]

    before_test_index = test_index


""" label 과 train 나누기 및 label에 대한 one-hot-encoding """

train_label = []
test_label = []
for data in train_data:
    train_label.append(int(data[0])-1)
for data in test_data:
    test_label.append(int(data[0])-1)

from keras.utils.np_utils import to_categorical

one_hot_train_labels = to_categorical(train_label)
one_hot_test_labels = to_categorical(test_label)


from keras import models
from keras import layers

model = models.Sequential()
model.add(layers.Dense(200, activation='relu'))
model.add(layers.Dense(200, activation='relu'))
model.add(layers.Dense(200, activation='relu'))
model.add(layers.Dense(200, activation='relu'))
model.add(layers.Dense(200, activation='relu'))
model.add(layers.Dense(23, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])


train_data = np.array(train_data)
test_data = np.array(test_data)


history = model.fit(train_data, one_hot_train_labels, validation_split=0.1, epochs=500)

results = model.evaluate(test_data, one_hot_test_labels)

print("result=",results)
print("끝")


