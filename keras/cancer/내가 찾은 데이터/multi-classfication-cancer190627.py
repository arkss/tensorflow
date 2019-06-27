import keras
import numpy as np
import pandas as pd 

""" 데이터 불러오기 """
all_data = pd.read_csv("./total_data_13_500.csv" )


# label 만 제거만 데이터
data = all_data.drop(columns = ['label'])
# label 만 남은 데이터
label_data = all_data[['label']]
label_data_list = label_data.values.tolist()


# print(all_data)
# [6237 rows x 2257 columns]




from sklearn.utils import shuffle
all_data = shuffle(all_data)


""" train, test data 로 나누기 """
all_data_list = all_data.values.tolist()


# count_list = [0 for i in range(1,24)]
count_list = [0 for i in range(1,14)]


for data_list in all_data_list:
    count_list[int(data_list[0])-1] += 1
    
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
model.add(layers.Dense(13, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])


train_data = np.array(train_data)
test_data = np.array(test_data)


history = model.fit(train_data, one_hot_train_labels, validation_split=0.2,shuffle=True, epochs=100)
 

results = model.evaluate(test_data, one_hot_test_labels)

predict_classes = model.predict_classes(train_data)


print("#####################")
print(len(predict_classes)) # 4358
print(len(train_label))     # 4358

# answer_matrix 는 행에는 정답이, 열에는 예측한 수가 위치한다. 즉 diagonal 한 원소에 값이 클수록 정확도가 높다고 예측할 수 있다. 
answer_matrix = [[0]*13 for i in range(13)]
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

plt.savefig("./graph_images/acc/acc_200Dense_4_dropdout0.3_2_edit13.png")

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.savefig("./graph_images/loss/loss_200Dense_4_dropdout0.3_2_2_edit13.png")



[[97, 0, 21, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 66, 48, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [2, 0, 2352, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 55, 39, 111, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [1, 11, 4, 9, 97, 1, 2, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 8, 0, 105, 0, 0, 0, 0, 0, 0, 0],
 [0, 5, 0, 2, 3, 0, 134, 0, 0, 0, 0, 0, 0],
 [0, 30, 2, 31, 0, 1, 0, 37, 0, 0, 1, 0, 0],
 [0, 5, 0, 0, 23, 1, 62, 2, 38, 0, 0, 0, 0],
 [0, 1, 3, 0, 1, 0, 0, 0, 0, 188, 1, 0, 0],
 [0, 0, 1, 1, 0, 0, 8, 0, 0, 0, 629, 0, 2],
 [9, 0, 4, 8, 2, 37, 1, 0, 3, 0, 0, 58, 9],
 [0, 0, 3, 19, 0, 18, 1, 5, 1, 1, 21, 4, 193]]

[[118, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [1, 105, 0, 3, 0, 1, 1, 2, 0, 0, 0, 0, 0],
 [0, 0, 404, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
 [1, 78, 2, 120, 2, 2, 0, 5, 0, 0, 0, 0, 0],
 [1, 0, 0, 1, 118, 0, 2, 1, 5, 1, 0, 0, 0],
 [1, 0, 0, 4, 0, 99, 0, 1, 0, 0, 0, 2, 1],
 [0, 0, 0, 0, 1, 0, 139, 0, 1, 0, 0, 0, 0],
 [0, 0, 0, 4, 2, 3, 2, 83, 0, 0, 0, 0, 2],
 [0, 0, 0, 1, 4, 0, 17, 1, 115, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 196, 0, 0, 0],
 [0, 0, 0, 0, 0, 1, 13, 1, 2, 1, 370, 2, 6],
 [1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 129, 2],
 [0, 0, 0, 0, 0, 2, 0, 2, 0, 3, 0, 2, 241]]

 