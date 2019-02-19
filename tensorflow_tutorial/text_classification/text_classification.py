"""
영화 리뷰 텍스트를 긍정 또는 부정으로 분류

50,000개의 영화 리뷰 텍스트를 담음 데이터 셋 사용
"""

from __future__ import absolute_import, division,print_function, unicode_literals
import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

# 데이터 다운로드

imdb = keras.datasets.imdb

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
# num_words = 10000 은 훈련데이터에서 가장 많이 등장하는 상위 10,000 개의 단어를 선택. 데이터의 크기를 적당하게 유지하기 위해 드물게 등장하는 단어는 제외

print("훈련 샘플: {}, 레이블: {}".format(len(train_data),len(train_labels)))
# 훈련 샘플: 25000, 레이블: 25000

print(train_data[0])
# 어휘 사전에 있는 단어들을 정수에 매칭
# [1, 14, 22, 16, 43, 530, 973, 1622, 1385, 65, 458, 4468, 66, 3941, 4, 173, 36, 256, 5, 25, 100, 43, 838, 112, 50, 670, 2, 9, 35, 480, 284, 5, 150, 4, 172, 112, 167, 2, 336, 385, 39, 4, 172, 4536, 1111, 17, 546, 38, 13, 447, 4, 192, 50, 16, 6, 147, 2025, 19, 14, 22, 4, 1920, 4613, 469, 4, 22, 71, 87, 12, 16, 43, 530, 38, 76, 15, 13, 1247,4, 22, 17, 515, 17, 12, 16, 626, 18, 2, 5, 62, 386, 12, 8, 316, 8, 106, 5, 4, 2223, 5244, 16, 480, 66, 3785, 33, 4, 130, 12, 16, 38, 619, 5, 25, 124, 51, 36, 135, 48, 25, 1415, 33, 6, 22, 12, 215, 28, 77, 52, 5, 14, 407, 16, 82, 2, 8, 4, 107, 117, 5952, 15, 256, 4, 2, 7,3766, 5, 723, 36, 71, 43, 530, 476, 26, 400, 317, 46, 7, 4, 2, 1029, 13, 104, 88, 4, 381, 15, 297, 98, 32, 2071, 56, 26, 141, 6, 194, 7486, 18, 4, 226, 22, 21, 134, 476, 26, 480, 5, 144, 30, 5535, 18, 51, 36, 28, 224, 92, 25, 104, 4, 226, 65, 16, 38, 1334, 88, 12, 16, 283, 5, 16, 4472, 113, 103, 32, 15, 16, 5345, 19, 178, 32]

# print(len(train_data[0]),len(train_data[1])) # 218,189


# 정수를 단어로 다시 변환하기

word_index = imdb.get_word_index()
# {'contends': 40832, 'copywrite': 88584, 'geysers': 52006, 'artbox': 88585, 'cronyn': 52007, 'hardboiled': 52008, "voorhees'": 88586, '35mm': 16818, "'l'": 88587, 'paget': 18512, 'expands': 20600}

word_index = {k:(v+3) for k,v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
# key와 value 값이 뒤바뀜

# 리뷰 텍스트 출력하는 함수
def decode_review(text):
    return ' '.join([reverse_word_index.get(i,'?') for i in text])
"""
get 함수
key 값으로 value 얻기
.get('키이름') 로 value를 뽑아낼 때 없으면 none 이 뜬다.
뒤에 인자를 하나 더 추가하여 .get('키이름','대체 값') 로 value를 뽑아낼 때 해당되는 키가 없으면 none 대신 대체 값이 나오게 된다.
"""
print(decode_review(train_data[0]))
"""
"<START> this film was just brilliant casting location ...
life after all that was shared with us all"
"""

#데이터 준비
"""
정수 배열의 길이가 모두 같도록 패딩(padding)을 추가해 max_length * num_reviews 크기의 정수 텐서를 만듭니다. 이런 형태의 텐서를 다룰 수 있는 임베딩(embedding) 층을 신경망의 첫 번째 층으로 사용할 수 있습니다.
"""
train_data = keras.preprocessing.sequence.pad_sequences(train_data,
value=word_index["<PAD>"],
padding="post",
maxlen=256)
test_data = keras.preprocessing.sequence.pad_sequences(test_data,
value=word_index["<PAD>"],
padding="post",
maxlen=256)
"""
pad_sequences 함수에 대해서 알아보자.
첫번째 인자로 sequences
value 값으로 padding 되는 자리에 어떤 값이 들어가는지 들어간다.
Padding 을 어느 방식으로 할지 결정한다. pre: 앞, post: 뒤
truncating 은 어디서 자를지 결정한다. pre: 앞, post: 뒤
maxlen 은 최대길이를 지정한다.
"""
# padding 결과 확인

print(train_data[0])
"""
[  1   14   22   16   43  530  973 1622 1385   65 
  ... 
   0    0    0    0    0    0    0    0    0    0  ]
"""
# 모델 구성
"""
모델에서 얼마나 많은 층을 사용할 것인가,
각 층에서 얼마나 많은 은닉유닛을 사용할 것인가
"""
vocab_size = 10000

model = keras.Sequential()
"""
모델을 만드는 방식이 두 가지이다.
1. model = Sequential([ 이 안에 layer들을 담아준다. ])
2. model = sequential()
model.add(모델)
model.add(모델)
"""

model.add(keras.layers.Embedding(vocab_size, 16, input_shape=(None,)))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16,activation=tf.nn.relu))
model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))
"""
1.embedding
모델의 첫번째 레이어로만 사용
embedding (input_dim, output_dim, input_shape)
입력모양 : (batch_size, sequence_length)
출력형태 : (batch_size, sequence_length, output_dim)
해당 레이어는 숫자로 인코딩 되어있는 각 단어를 사용하며 각 단어 인덱스에 대한 벡터를 찾는다.

2. globalaveragepooling1d
sequence 차원을 평균을 계산하여 고정된 길이의 벡터를 출력한다.
가변적인 길이의 입력을 간단하게 처리할 수 있다.

3. 첫번째 dense layer를 통해 고정길이로 출력된 vector 값을 통해 16개의 hidden unit을 가진 fully-connected layer를 통과
   두번째 dense layer를 통해 시그모이드를 통해 0~1 값을 가지게 한다.

"""
model.summary()
"""
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
embedding (Embedding)        (None, None, 16)          160000    
_________________________________________________________________
global_average_pooling1d (Gl (None, 16)                0         
_________________________________________________________________
dense (Dense)                (None, 16)                272       
_________________________________________________________________
dense_1 (Dense)              (None, 1)                 17        
=================================================================
Total params: 160,289
Trainable params: 160,289
Non-trainable params: 0
_________________________________________________________________
"""

# 손실함수와 옵티마이저

model.compile(optimizer=tf.train.AdamOptimizer(),
                loss = 'binary_crossentropy',
                metrics = ['accuracy'])

# 검증 세트 만들기
# val 이 붙은 변수들은 검증 데이터
x_val = train_data[:10000]
partial_x_train = train_data[10000:]

y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]

# 모델 훈련
"""
512개의 샘플로 이루어진 미니배치에서 40번의 epoch동안 훈련,
모든 샘플에 대해서 40번 반복
"""
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=40,
                    batch_size=512,
                    validation_data = (x_val,y_val),
                    verbose=1)

#loss:0.1944 - acc: 0.9317 - val_loss: 0.2915 - val_acc: 0.8846

# 모델 평가

result = model.evaluate(test_data, test_labels)

print(result)
#[0.3208913328838348, 0.87344] 손실과 정확도

# 정확도와 손실 그래프 그리기
"""
모델 훈련시 사용했던 fit() 메소드는 history 객체를 반환.
훈련하는 동안의 정보가 담긴 딕셔너리가 들어있음.
"""

history_dict = history.history
print(history_dict.keys()) # dict_keys(['acc', 'val_loss', 'val_acc', 'loss'])

acc = history_dict['acc'] # 훈련정확도
val_acc = history_dict['val_acc'] # 검증손실
loss = history_dict['loss'] # 검증정확도
val_loss = history_dict['val_loss'] # 훈련손실

epochs = range(1, len(acc)+1)
# bo: 파란색 점 , b: 파란 실선
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='validation loss')
plt.title('training and validation loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend() # 범례 표시하기
 
# plt.show()

plt.clf() # 그림 초기화
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('training and validation accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()

# plt.show()

"""
훈련 손실은 epochs 마다 감소하고 정확도는 증가
검증 손실과 검증 정확도는 20epochs 부터 overfitting
"""
