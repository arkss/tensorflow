import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/",one_hot = True)



X = tf.placeholder(tf.float32,[None,784])
X_img = tf.reshape(X,[-1,28,28,1])
Y = tf.placeholder(tf.float32, [None, 10])

keep_prob = tf.placeholder(tf.float32)

### conv layer1

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
# shape = (?,14,14,32)
L1 = tf.nn.dropout(L1, keep_prob = keep_prob)
# shape = (?,14,14,32)


### conv layer2

# W2 은 filter
W2 = tf.Variable(tf.random_normal([3,3,32,64], stddev = 0.01))
# filter의 크기는 3*3 / 32는 input data depth와 맞춰줌 / 64는 filter의 개수

L2 = tf.nn.conv2d(L1 ,W2,strides = [1,1,1,1], padding = 'SAME')
# shape = (?,14,14,64)
L2 = tf.nn.relu(L2)
# shape = (?,14,14,64)
L2 = tf.nn.max_pool(L2, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')
# shape = (?,7,7,64)

L2 = tf.nn.dropout(L2, keep_prob = keep_prob)
# shape = (?,14,14,32)

### con layer3

# W3 는 filter
W3 = tf.Variable(tf.random_normal([3,3,64,128], stddev = 0.01))
L3 = tf.nn.conv2d(L2, W3, strides= [1,1,1,1], padding = 'SAME')
# shape = (?,7,7,128)
L3 = tf.nn.relu(L3)
# shape = (?,7,7,128)
L3 = tf.nn.max_pool(L3, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')
# shape = (?,4,4,128)
L3 = tf.nn.dropout(L3, keep_prob = keep_prob)
# shape = (?,4,4,128)

L3_flat = tf.reshape(L3, [-1, 128 * 4 * 4])
# shape = (?, 2048)
# reshape 해서 펼쳐주는 과정


### fully connected layer : 4*4*128 inputs -> 625 outputs
W4 = tf.get_variable("W4",shape = [128*4*4,625],
        initializer = tf.contrib.layers.xavier_initializer())
b4 = tf.Variable(tf.random_normal([625]))
L4 = tf.nn.relu(tf.matmul(L3_flat, W4)+b4)
L4 = tf.nn.dropout(L4, keep_prob = keep_prob)

### fully connected layer : 625 inputs -> 10 outputs

W5 = tf.get_variable("W5", shape = [625,10],
    initializer = tf.contrib.layers.xavier_initializer())
b5 = tf.Variable(tf.random_normal([10]))
hypothesis = tf.matmul(L4,W5)+b5
# shape = (?, 10)

#cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits , labels =Y))
#optimizer = tf.train.AdamOptimizer(learning_rate = 0.001).minimize(cost)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=hypothesis, labels=Y))

optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

### training and evaluation

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
        feed_dict = {X:batch_xs, Y:batch_ys, keep_prob : 0.7}
        c, _ = sess.run([cost,optimizer], feed_dict = feed_dict)
        avg_cost += c / total_batch
    print("epoch :", '%04d' %(epoch+1), 'cost =', '{:.9f}'.format(avg_cost))


print('learning finished!')

correct_prediction = tf.equal(tf.argmax(hypothesis,1), tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print('accuracy:', sess.run(accuracy, feed_dict = {X:mnist.test.images, Y : mnist.test.labels, keep_prob : 1}))


"""

SAME 패딩의 경우, 출력 높이와 폭은 다음과 같이 계산됩니다.
out_height = ceil (float (in_height) / float (strides [1]))

out_width = ceil (float (in_width) / float (strides [2]))

과

 VALID 패딩의 경우, 출력 높이와 폭은 다음과 같이 계산됩니다.
out_height = ceil (float (in_height - filter_height + 1) / float (strides [1]))

out_width = ceil (float (in_width - filter_width + 1) / float (strides [2]))
"""