
import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt 
import random
# 사진을 불러오기 위해서 두 개를 import


from tensorflow.examples.tutorials.mnist import input_data
# https://www.tensorflow.org/get_started/mnist/beginners

mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)
# MINST_data 를 경로에 다운받는다. 
# one_hot = true 로 하면 Y 데이터를 읽어올 때 자동으로 onehotencoding 해준다.
nb_classes = 10
# 0~9까지 10개의 classfication 이므로 10

X = tf.placeholder(tf.float32, [None,784])
# input data 의 숫자 이미지는 28*28 = 784
Y = tf.placeholder(tf.float32, [None, nb_classes])

W = tf.Variable(tf.random_normal([784, nb_classes]))
b = tf.Variable(tf.random_normal([nb_classes]))

hypothesis = tf.nn.softmax(tf.matmul(X,W)+b)

cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis = 1))
# cross entropy Y는 현재 one hot encoding
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.1).minimize(cost)

is_correct = tf.equal(tf.arg_max(hypothesis, 1), tf.arg_max(Y,1))
# hypothesis 와 arg_max 의 값을 바교해서 같으면 True 다르면 False
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
# 말그대로 cast 연산자이다. is_correct의 값이 현재 boolean 값으로 들어있는데
# 그를 float 형태로 cast 해준다.
# 그 후 평균을 구하면 이것이 정확도가 된다.

training_epochs = 15
batch_size = 100

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # 변수 초기화

    
    for epoch in range(training_epochs): # 15번 반복한다.
        avg_cost = 0
        total_batch = int(mnist.train.num_examples / batch_size)
        # 전체 사이즈 / 배치수 를 함으로서 몇 번 돌아야하는지 구한다.
        
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            # 데이터를 한 번에 다 올리면 공간이 많이 차지하므로
            # batch를 이용해서 100개씩 올린다.
            c, _ = sess.run([cost,optimizer], feed_dict={X:batch_xs, Y:batch_ys})
            avg_cost += c / total_batch
# epoch : 전체 데이터를 한 번 학습시킨 걸 1 epoch 라고 한다.
# 1000의 training data가 있을 때 500의 크기로 batch 한다고 하면
# 1epoch가 되기 위해선 2iteration 해야한다.
        print('Epoch:', '%04d' % (epoch+1),
        'cost = ','{:.9f}'.format(avg_cost))

    print("Accuracy: ", accuracy.eval(session=sess,
         feed_dict={X:mnist.test.images, Y:mnist.test.labels}))
    # 그 전 training과 관련없는 test 데이터를 가져온다.

    r = random.randint(0, mnist.test.num_examples - 1)
    # random 을 import 한 후 사용가능하고 두 개의 매개변수를 받아 
    # 두 수를 포함한 사이의 수 중 random하게 수를 return 한다.
    
    print("mnist.test.num_examples:",mnist.test.num_examples) # 10000
    print("mnist.train.num_examples:",mnist.train.num_examples) # 55000
    print("mnist.validation.num_examples:",mnist.validation.num_examples) # 5000
    # validation 은 training data 중에서 
    # learning rate 나 regularization strength을 잘 찾기 위한 데이터

    print("r:",r)

    print ("np.shape(mnist.train.images):", np.shape(mnist.train.images))  # (55000, 784)
    print ("np.shape(mnist.train.labels):",np.shape(mnist.train.labels))  # (55000, 10)
    # 데이터는 784(28x28)개의 픽셀을 가지는 이미지와
    # 10(0~9)개 클래스를 가지는 one hot 인코드된 레이블을 가지고 있음

    print("Labels:", sess.run(tf.argmax(mnist.test.labels[r:r+1],1)))
    print("Prediction:", sess.run(tf.argmax(hypothesis,1),
            feed_dict = {X:mnist.test.images[r:r+1]}))

    plt.imshow(mnist.test.images[r:r+1].
        reshape(28,28),cmap='Greys', interpolation='nearest')
        # interpolation='nearest' 는 디스플레이 해상도가 이미지 해상도와 같지 않은 경우
        #  픽셀 사이를 보간하지 않고 이미지를 표시합니다 (가장 자주 발생하는 경우).
        #  픽셀이 여러 픽셀의 사각형으로 표시되는 이미지가 생성됩니다.
    plt.show()