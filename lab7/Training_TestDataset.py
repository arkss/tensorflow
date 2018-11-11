import tensorflow as tf  
import numpy as np 
# 학습데이터와 테스트 데이터 구분
x_data = [[1, 2, 1],
          [1, 3, 2],
          [1, 3, 4],
          [1, 5, 5],
          [1, 7, 5],
          [1, 2, 5],
          [1, 6, 6],
          [1, 7, 7]]
y_data = [[0, 0, 1],
          [0, 0, 1],
          [0, 0, 1],
          [0, 1, 0],
          [0, 1, 0],
          [0, 1, 0],
          [1, 0, 0],
          [1, 0, 0]]

x_test = [[2, 1, 1],
          [3, 1, 2],
          [3, 3, 4]]
y_test = [[0, 0, 1],
          [0, 0, 1],
          [0, 0, 1]]


X = tf.placeholder("float",[None,3])
Y = tf.placeholder("float",[None,3])
W = tf.Variable(tf.random_normal([3,3]))
b = tf.Variable(tf.random_normal([3]))

hypothesis = tf.nn.softmax(tf.matmul(X,W)+b)
cost = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(hypothesis), axis = 1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.1).minimize(cost)
# learning rate 는 적절한 값을 줘야한다. 
# cost 함수과 inf 로 출력되고 W 가 nan 으로 출력되는 경우가 있는데 이는 learning rate 가 너무 큰 경우, 이를 잘 조절해줘야한다.

prediction = tf.arg_max(hypothesis,1)
is_correct = tf.equal(prediction, tf.arg_max(Y,1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(201):
        cost_val, W_val, _ = sess.run([cost,W,optimizer],
            feed_dict = {X:x_data , Y :y_data})
        print(step, cost_val, W_val)


    print("prediction:",sess.run(prediction, feed_dict={X:x_test}))
    print("Accuracy:", sess.run(accuracy, feed_dict={X:x_test, Y: y_test}))