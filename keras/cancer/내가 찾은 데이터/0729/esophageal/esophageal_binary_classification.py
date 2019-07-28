import keras
import numpy as np
import pandas as pd 
from random import *

""" 데이터 불러오기 """
all_data = pd.read_csv("./esophageal_cancer.csv" )


# label 만 제거만 데이터
data = all_data.drop(columns = ['label'])
# label 만 남은 데이터
label_data = all_data[['label']]
label_data_list = label_data.values.tolist()


# print(all_data)
# [6237 rows x 2257 columns]



all_data_list = all_data.values.tolist()


""" 데이터 증식 """
count_list = [0 for i in range(1,4)]

# 맨 앞은 0으로 비워 놓는다.
# count_list = [0,0,0]

for data_list in all_data_list:
    count_list[int(data_list[0])] += 1

all_data_500 = pd.DataFrame()

copy_index = 0
new_index = 0

for label in range(1,3):
    sub_data = all_data[all_data['label'].isin([''+str(label)])]
    print(label,"번째 label 시행중..")
    copy_index += count_list[label-1] # 어느 행을 복사하여 추가할지 고르는 변수
    new_index += count_list[label]
    
    temp_new_index = new_index
    temp_copy_index = randint(copy_index,new_index)
    while len(sub_data) != 1000:
        sub_data.loc[temp_new_index] = sub_data.loc[temp_copy_index] 
        temp_new_index += 1
        
    all_data_500 = pd.concat([all_data_500, sub_data], ignore_index=True)

print(all_data_500)
    
from sklearn.utils import shuffle
all_data_500 = shuffle(all_data_500)

""" train, test data 로 나누기 """
train_data = []
test_data = []

before_test_index = 0
train_index = 0
test_index = 0

for count in count_list:
    # 0.7 / 0.3 , training / test
    train_index = int(count*0.8) + before_test_index
    test_index = int(count*0.2) + train_index

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
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dropout(0.3))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.3))
model.add(layers.Dense(2, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])


train_data = np.array(train_data)
test_data = np.array(test_data)


history = model.fit(train_data, one_hot_train_labels, validation_split=0.2,shuffle=True, epochs=100)
 

results = model.evaluate(test_data, one_hot_test_labels)

predict_classes = model.predict_classes(train_data)


print("#####################")
print(len(predict_classes)) # 4358
print(len(train_label))     # 4358

# answer_matrix 는 행에는 정답이, 열에는 예측한 수가 위치한다. 즉 diagonal 한 원소에 값이 클수록 정확도가 높다고 예측할 수 있다. 
answer_matrix = [[0]*2 for i in range(2)]
for predict,label in zip(predict_classes, train_label):
    answer_matrix[label][predict] += 1
    
print("@@@@@@@@@@@@@@@@@@@@@@")
print(answer_matrix)

# print("predict_classes=", model.predict_classes(train_data))
# print("predict=", model.predict(train_data))
print("result=",results)
print("끝")


import matplotlib
matplotlib.use('agg')
# RuntimeError: Invalid DISPLAY variable 오류를 제거하기 위한 코드
import matplotlib.pyplot as plt


acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1,len(acc)+1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.savefig("./acc_binary_esophageal.png")

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.savefig("./loss_binary_esophageal.png")

