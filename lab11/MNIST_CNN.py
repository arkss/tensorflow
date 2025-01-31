import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/",one_hot = True)

# conv layer1

X = tf.placeholder(tf.float32,[None,784])
X_img = tf.reshape(X,[-1,28,28,1])
Y = tf.placeholder(tf.float32, [None, 10])

# W1 은 filter
W1 = tf.Variable(tf.random_normal([3,3,1,32], stddev = 0.01))
# 표준편차 0.01 로 설정
# 3*3 개의 크기 / 색상은 1개 (black & white) / 32개의 weight,filter

L1 = tf.nn.conv2d(X_img, W1, strides = [1,1,1,1], padding = 'SAME')
# strides 각 자리의 의미 : [batch, width, height, depth] 
# shape = (?,28,28,32)
L1 = tf.nn.relu(L1)
# shape = (?,28,28,32)
L1 = tf.nn.max_pool(L1, ksize=[1,2,2,1], strides = [1,2,2,1], padding = 'SAME')
# ksize는 커널의 사이즈



# conv layer2

# W2 은 filter
W2 = tf.Variable(tf.random_normal([3,3,32,64], stddev = 0.01))
# filter의 크기는 3*3 / 32는 input data depth와 맞춰줌 / 64는 filter의 개수

L2 = tf.nn.conv2d(L1 ,W2,strides = [1,1,1,1], padding = 'SAME')
# shape = (?,14,14,64)
L2 = tf.nn.relu(L2)
# shape = (?,14,14,64)
L2 = tf.nn.max_pool(L2, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')
# shape = (?,7,7,64)
L2 = tf.reshape(L2, [-1, 7*7*64])
# shape = (?, 3136)
# reshape 해서 펼쳐주는 과정

# fully connected layer

W3 = tf.get_variable("W3", shape = [7*7*64,10],
    initializer = tf.contrib.layers.xavier_initializer())
b = tf.Variable(tf.random_normal([10]))
hypothesis = tf.matmul(L2,W3)+b

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = hypothesis , labels =Y))
optimizer = tf.train.AdamOptimizer(learning_rate = 0.01).minimize(cost)


# training and evaluation

sess = tf.Session()
sess.run(tf.global_variables_initializer())

print("learning started. it takes sometime")

training_epochs = 15
batch_size = 100
for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = int(mnist.train.num_examples / batch_size)
    for i in range(total_batch):
        batch_xs , batch_ys = mnist.train.next_batch(batch_size)
        feed_dict = {X:batch_xs, Y:batch_ys}
        c, _ = sess.run([cost,optimizer], feed_dict = feed_dict)
        avg_cost += c / total_batch
    print("epoch :", '%04d' %(epoch+1), 'cost =', '{:.9f}'.format(avg_cost))

print('learning finished!')

correct_prediction = tf.equal(tf.argmax(hypothesis,1), tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print('accuracy:', sess.run(accuracy, feed_dict = {X:mnist.test.images, Y : mnist.test.labels}))
