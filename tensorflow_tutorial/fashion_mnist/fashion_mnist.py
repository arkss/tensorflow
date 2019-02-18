from __future__ import absolute_import, division, print_function, unicode_literals
"""
호환성있는 코드를 짜기 위한 방법
absoulute_import : 여러 모듈이 같은 이름을 가질 때 문제 없이 import 하는 것을 도와줌
division : 파이썬 2와 3은 / 의 기능이 조금 다르다. 이를 3에 맞춰서 계산
print_function : 2는 출력시 ()사용 안했는데 비해, 3은 사용
unicode_literals : 인용된 문자열을 바이트가 아닌 유니코드 시퀀스로 인코딩
"""
# tensorflow와 tf.keras import
import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__) # 1.9.0



# data download
fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# 레이블은 0~9까지의 정수 배열이고, 각각 한 가지 옷의 종류에 매칭되는데 이를 리스트로 만들어놓자.

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# 10개 카테고리, 70,000 개의 패션 흑백 이미지 
# train : 60,000, 28*28 pixel , test : 10,000, 28*28 pixel

# print('train_images.shape : ', train_images.shape) # (60000,28,28)
# print('len(train_labels) : ',len(train_labels)) # 60000
# print('train_label : ',train_labels) # [9 0 0 ... 3 0 5]
# print('test_images.shape : ', test_images.shape) #(10000, 28, 28)
# print('test_labels : ', test_labels) # [9 2 1 ... 8 1 5]

# 이미지 확인 
# plt.figure()
# plt.imshow(train_images[10])
# plt.colorbar() # 우측에 색깔 막대
# plt.grid(False) # 격자제거
# plt.show()

# 이미지를 보면 픽셀 값의 범위가 0~255 사이라는 것을 알 수 있음
# 이를 0~1 사이로 수로 조정

train_images = train_images / 255.0
test_images = test_images / 255.0


# test_images 와 labels 확인하기
plt.figure(figsize=(10,10)) # size 조절
for i in range(25): # 처음 25개 확인
    plt.subplot(5,5,i+1) # (행,렬,순서-0이 아닌 1부터 시작) 에 맞게 출력 칸 생성
    plt.xticks([]) # x축과 y축 눈금 표시
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()

# layer 구성

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

"""
28*28 2차원 배열 -> 784 1차원 배열로 변환

128개의 노드, 활성화 함수로 relu

10개의 노드, softmax, 전체의 합은 1, 10개 중 각각에 속할 확률로 표시
"""

# 훈련하기 전 설정

model.compile(optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])

"""
optimizer 방법과 loss함수를 결정,
측정항목의 목록 리스트화
"""

# 모델 훈련

model.fit(train_images, train_labels, epochs=5)
#loss: 0.2958 - acc: 0.8908
"""
training data와 epochs의 수를 설정해주면 훈련가능
"""

# 정확도 평가

test_loss, test_acc = model.evaluate(test_images, test_labels)

print('loss : ',test_loss) # 0.3587061176776886
print('정확도 : ',test_acc)  # 0.8703 

"""
test의 정확도가 model보다 조금 낮음.
overfitting 때문으로 예상
"""

# 예측 만들기

# 위에서 훈련될 model을 사용하여 이미지에 대한 예측을 만들 수 있다.
predictions = model.predict(test_images)

print('predictions[0] : ', predictions[0]) 
"""
 array([4.2575375e-05, 1.0634101e-07, 4.3224929e-07, 1.0461436e-06, 2.2473921e-06, 5.3110592e-02, 8.6338503e-07, 1.5696593e-02, 5.1282190e-05, 9.3109423e-01], dtype=float32)

각 수가 나타내는 것은 각 index에 해당하는 class의 신뢰도
 """
print(np.argmax(predictions[0])) # 9, 가장 큰 값은 9
print(test_labels[0]) # 9, 실제 labels도 9


