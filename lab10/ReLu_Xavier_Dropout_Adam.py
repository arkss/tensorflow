import tensorflow as tf 
import numpy as np
import random
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)

nb_classes = 10

X = tf.placeholder(tf.float32, [None,784])
Y = tf.placeholder(tf.float32, [None,nb_classes])

keep_prob = tf.placeholder(tf.float32)
# keep_prob 을 다음과 같이 변수로 설정한 이유는 train 할 때와 test 할 때의 값이 달라야 하기 때문이다.
# 대체적으로 train은 0.5~0.7 로 하고 test 할 때는 무조건 1로 해야한다.

#W1 = tf.Variable(tf.random_normal([784,256]))
# 초기화 방식을 샤비에 방식으로 변경
# 변경하기 위해서는 get_variable 을 사용한다.
# 이는 매개변수로 (name, shape, initializer)을 받기 때문에 초기화를 지정해줄 수 있다.
W1 = tf.get_variable("W1", shape = [784,512], initializer = tf.contrib.layers.xavier_initializer())
b1 = tf.Variable(tf.random_normal([512]))
L1 = tf.nn.relu(tf.matmul(X,W1)+b1)
L1 = tf.nn.dropout(L1, keep_prob=keep_prob)
# 활성화 함수로 relu를 사용. 0보다 작을 때는 0의 값을 가지고 0보다 클 떄는 x의 값을 가지는 함수
# dropout 을 하는 방법은 layer를 다음과 같이 하나 더 추가해준다.

#W2 = tf.Variable(tf.random_normal([256,256]))
W2 = tf.get_variable("W2", shape = [512,512], initializer = tf.contrib.layers.xavier_initializer())
b2 = tf.Variable(tf.random_normal([512]))
L2 = tf.nn.relu(tf.matmul(L1,W2)+b2)

# 층을 두 개 추가하고 전체적으로 넓게 해준다.
W3 = tf.get_variable("W3", shape = [512,512], initializer = tf.contrib.layers.xavier_initializer())
b3 = tf.Variable(tf.random_normal([512]))
L3 = tf.nn.relu(tf.matmul(L2,W3)+b3)

W4 = tf.get_variable("W4", shape = [512,512], initializer = tf.contrib.layers.xavier_initializer())
b4 = tf.Variable(tf.random_normal([512]))
L4 = tf.nn.relu(tf.matmul(L3,W4)+b4)

#W3 = tf.Variable(tf.random_normal([256,nb_classes]))
W5 = tf.get_variable("W5", shape =[512,nb_classes], initializer = tf.contrib.layers.xavier_initializer())
b5 = tf.Variable(tf.random_normal([nb_classes]))
hypothesis = tf.matmul(L4,W5)+b5

# 다음과 같은 cost 함수를 쓸 때는 hypothesis에 활성화함수가 붙지 않는다.
# 그 이유는 저 cost 함수자체에 softmax를 포함하고 있기 때문이다.
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=hypothesis, labels=Y))
# cost = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(hypothesis),axis=1))
optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)
# optimizer 를 Adam으로 변경

is_correct = tf.equal(tf.argmax(hypothesis,1 ), tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

training_epochs = 15
batch_size = 100

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples / batch_size)

        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            c, _ = sess.run([cost,optimizer], feed_dict={X:batch_xs, Y:batch_ys, keep_prob : 0.7})
            # train 할 때는 keep_prob을 0.7로 설정해준다.
            avg_cost += c / total_batch

        print('Epoch:', '%04d' % (epoch+1),
            'cost = ','{:.9f}'.format(avg_cost))    


    print("Accuracy: ", accuracy.eval(session=sess,
         feed_dict={X:mnist.test.images, Y:mnist.test.labels, keep_prob : 1}))
        #   test 할 때는 keep_prob을 1로 설정해준다.
    r = random.randint(0, mnist.test.num_examples - 1)

    print("labels:", sess.run(tf.argmax(mnist.test.labels[r:r+1],1)))
    print("prediction:", sess.run(tf.argmax(hypothesis,1),
            feed_dict= {X:mnist.test.images[r:r+1]}))
