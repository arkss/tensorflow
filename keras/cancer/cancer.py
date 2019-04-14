import keras

import numpy as np
import pandas as pd 

all_data = pd.read_csv("./total.csv" )

# print(all_data.shape) # (3967, 2541)

# label 만 제거만 데이터
data = all_data.drop(columns = ['label'])

# label 만 남은 데이터
label_data = all_data[['label']]

# 70 : 15 : 15

train_index = int(len(data) * 0.85)
test_index = int(len(data) * 0.15) + train_index

train_data = data[:train_index]
test_data = data[train_index:test_index]

from keras.utils.np_utils import to_categorical

train_labels = label_data[:train_index]
one_hot_train_labels = to_categorical(train_labels)
test_labels = label_data[train_index:test_index]
one_hot_test_labels = to_categorical(test_labels)

print(len(train_data), len(test_data)) # 3371 595

from keras import models
from keras import layers

model = models.Sequential()
model.add(layers.Dense(200, activation='relu'))
model.add(layers.Dense(200, activation='relu'))
model.add(layers.Dense(200, activation='relu'))
model.add(layers.Dense(200, activation='relu'))
model.add(layers.Dense(2, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

history = model.fit(train_data.values, one_hot_train_labels, validation_split=0.2, epochs=10)

results = model.evaluate(test_data, one_hot_test_labels)
print("result=",results)

import matplotlib.pyplot as plt

history_dict = history.history
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1,len(loss)+1)

plt.plot(epochs, loss , 'bo', label='training loss')
plt.plot(epochs, val_loss , 'b', label='validation loss')
plt.title('training and validation loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()

plt.show()

plt.clf()
acc = history_dict['acc']
val_acc = history_dict['val_acc']

plt.plot(epochs, acc, 'bo', label='training acc')
plt.plot(epochs, val_acc, 'b', label='validation acc')
plt.title('training and validation accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()

plt.show()





