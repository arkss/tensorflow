import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/",one_hot = True)

img = mnist.train.images[0].reshape(28,28)
plt.imshow(img,cmap ='gray')
plt.show()

sess = tf.InteractiveSession()

img = img.reshape(-1,28,28,1)
W1 = tf.Variable(tf.random_normal([3,3,1,5], stddev = 0.01))
# [3,3,1,5]의 의미를 살펴보면
# 3 3 은 filter의 크기 / 1 은 색깔(따라서 img의 마지막 원소와 동일해야한다.) / 5는 필터의 개수
# random_normal 로 설정할 떄 stddev로 정규분포의 표준편차를 지정해줄 수 있다.

conv2d = tf.nn.conv2d(img,W1,strides=[1,2,2,1], padding='SAME')
# strides 각 자리의 의미 : [batch, width, height, depth] 
# [0],[3] 은 통상적으로 1을 사용하고 [1],[2] 같은 값을 사용
# 기존 28*28 size에서 14*14 size로 바뀜
print(conv2d)

pool = tf.nn.max_pool(conv2d, ksize=[1,2,2,1],strides=[1,2,2,1],padding ='SAME')
# 기존 14*14 size에서 7*7 size로 바뀜
print(pool)

