"""
일정 에폭 동안 훈련시 모델의 성능이 최고점에 도달한 다음 감소하기 시작,
train에서는 높은 성능을 얻지만 test에서는 성능이 떨어짐
-> overfitting(과대적합)

반대말로는 underfitting(과소적합),
성능이 아직 향상될 여지가 남아있을 때(모델이 너무 단순하거나, 규제가 너무 많거나, 충분히 오래 훈련하지 않은 경우)

과대적합를 막는 가장 좋은 방법은 더 많은 훈련 데이터

이것이 불가능 할 때는 regularization(규제)가 필요
모델이 저장할 수 있는 정보의 양과 종류에 제약을 부과
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

"""
임베딩을 사용안하고 멀티-핫 인코딩 방법으로 사용
[3,5] -> [0,0,1,0,1,0, ... ,0] 
"""
NUM_WORDS = 10000

(train_data, train_labels),(test_data, test_labels) = keras.datasets.imdb.load_data(num_words=NUM_WORDS)



def multi_hot_sequences(sequences, dimension):
    # 0으로 채워진 (len(sequences), dimension) 크기의 행렬
    results = np.zeros((len(sequences), dimension))
    for i, word_indices in enumerate(sequences):
        results[i,word_indices] = 1.0
    return results

train_data = multi_hot_sequences(train_data, NUM_WORDS)
test_data = multi_hot_sequences(test_data, NUM_WORDS)

plt.plot(train_data[0])
# plt.show()

# 모델 만들기
"""
overfitting 을 막기 위해서는 중도를 찾는 것이 중요하다.
기준 모델을 만들고 여러 방법으로 바꾸면서 적합한 모델을 찾아보자.
"""
# 기준 모델
baseline_model = keras.Sequential([
    keras.layers.Dense(16, activation=tf.nn.relu, input_shape=(NUM_WORDS,)),
    keras.layers.Dense(16, activation=tf.nn.relu),
    keras.layers.Dense(1, activation=tf.nn.sigmoid)
])

baseline_model.compile(optimizer='adam',
loss = 'binary_crossentropy',
metrics=['accuracy','binary_crossentropy'])

# baseline_model.summary()
baseline_history = baseline_model.fit(train_data,
                                      train_labels,
                                      epochs=20,
                                      batch_size=512,
                                      validation_data=(test_data, test_labels),
                                      verbose=2)
# 작은 모델

smaller_model = keras.Sequential([
    keras.layers.Dense(4, activation=tf.nn.relu, input_shape=(NUM_WORDS,)),
    keras.layers.Dense(4, activation=tf.nn.relu),
    keras.layers.Dense(1, activation=tf.nn.sigmoid)
])

smaller_model.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy', 'binary_crossentropy'])

# smaller_model.summary()
smaller_history = smaller_model.fit(train_data,
                                    train_labels,
                                    epochs=20,
                                    batch_size=512,
                                    validation_data=(test_data, test_labels),
                                    verbose=2)

# 큰 모델

bigger_model = keras.Sequential([
    keras.layers.Dense(512, activation=tf.nn.relu, input_shape=(NUM_WORDS,)),
    keras.layers.Dense(512, activation=tf.nn.relu),
    keras.layers.Dense(1, activation=tf.nn.sigmoid)
])

bigger_model.compile(optimizer='adam',
                     loss='binary_crossentropy',
                     metrics=['accuracy','binary_crossentropy'])

# bigger_model.summary()
bigger_history = bigger_model.fit(train_data,                                            
                                  train_labels,
                                  epochs=20,
                                  batch_size=512,
                                  validation_data=(test_data, test_labels),
                                  verbose=2)

def plot_history(histories, key='binary_crossentropy'):
  plt.figure(figsize=(16,10)) # 크기 고정
    
  for name, history in histories:
    # plt.plot(가로, 세로, 선의 모양 및 색, label 이름)
    # 첫 번째를 val 변수에 담은 이유는 아래에서 get_color 로 색을 뽑아와 같은 model 끼리 색을 통일하기 위해서
    val = plt.plot(history.epoch, history.history['val_'+key],
                   '--', label=name.title()+' Val')
    plt.plot(history.epoch, history.history[key], color=val[0].get_color(),
             label=name.title()+' Train')

  plt.xlabel('Epochs')
  plt.ylabel(key.replace('_',' ').title()) # _ 만 공백으로 바꾸기
  plt.legend()

  plt.xlim([0,max(history.epoch)])


plot_history([('baseline', baseline_history),
              ('smaller', smaller_history),
              ('bigger', bigger_history)])
"""
모델이 작을수록 overfitting이 덜 발생한다.
"""

# regularization

"""
l1 과 l2 방식이 있다.
L1 규제는 가중치의 절댓값에 비례하는 비용이 추가
L2 규제는 가중치의 제곱에 비례하는 비용이 추가

사용 방법은 regularizers 메소드의 l1,l2를 추가해준다.
그 뒤에 regularization 상수 값이 들어간다.
"""

l2_model = keras.models.Sequential([
    keras.layers.Dense(16, kernel_regularizer=keras.regularizers.l2(0.001),
                       activation=tf.nn.relu, input_shape=(NUM_WORDS,)),
    keras.layers.Dense(16, kernel_regularizer=keras.regularizers.l2(0.001),
                       activation=tf.nn.relu),
    keras.layers.Dense(1, activation=tf.nn.sigmoid)
])

l2_model.compile(optimizer='adam',
                 loss='binary_crossentropy',
                 metrics=['accuracy', 'binary_crossentropy'])

l2_model_history = l2_model.fit(train_data,
                                train_labels,
                                epochs=20,
                                batch_size=512,
                                validation_data=(test_data, test_labels),
                                verbose=2)

plot_history([('baseline', baseline_history),
               'l2', l2_model_history])

"""
overfitting이 덜 발생함
"""

# dropout 추가하기


dpt_model = keras.models.Sequential([
    keras.layers.Dense(16, activation=tf.nn.relu, input_shape=(NUM_WORDS,)),
    keras.layers.Dropout(0.5), # 0으로 만들어주는 비율이 들어감
    keras.layers.Dense(16, activation=tf.nn.relu),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(1, activation=tf.nn.sigmoid)
])

dpt_model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy','binary_crossentropy'])

dpt_model_history = dpt_model.fit(train_data,                                            
                                  train_labels,
                                  epochs=20,
                                  batch_size=512,
                                  validation_data=(test_data, test_labels),
                                  verbose=2)

"""
역시나 기존 모델보다 훨씬 overfitting 이 덜 일어난다.
"""