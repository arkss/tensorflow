
import tensorflow as tf   
import numpy as np
tf.set_random_seed(777)

xy = np.loadtxt('data-04-zoo.csv',delimiter=',',dtype=np.float32)
x_data = xy[:,0:-1]
y_data = xy[:,[-1]]

nb_classes = 7
# 0~6까지 7개의 class

X = tf.placeholder(tf.float32, [None,16]) # 하나의 데이터가 16가지의 특징을 가지고
Y = tf.placeholder(tf.int32, [None,1]) # 0~6까지 중 하나를 고를테니 1

Y_one_hot = tf.one_hot(Y, nb_classes)
Y_one_hot = tf.reshape(Y_one_hot,[-1,nb_classes])
"""
one hot incoding 에 대해서

이 과정을 거치기 전에 Y의 데이터 형태는 [[0],[3],[0], ... ,[0]]
shape = (?, 1) (몇 개의 데이터가 있을지 모르니 ?라고 명시하자.)
one_hot 함수는 기본적으로 (데이터, class의 수) 를 받는다.
해당되는 수는 1 아닌 수는 0으로 바꿔주는데 이는 on_value=3, off_value=2 이런식으로 바꿔줄수도 있다.
그러면 Y_one_hot 은 다음과 같은 형태를 가진다.
[[[1,0,0,0,0,0,0],[0,0,1,0,0,0,0],[1,0,0,0,0,0,0], ... , [1,0,0,0,0,0,0]]]
즉 rank가 1증가하게 된다.
shape = (?,1,7)
따라서 reshape이라는 과정을 거쳐야 한다.
reshape 함수의 예시
tensor 't' is [1, 2, 3, 4, 5, 6, 7, 8, 9]
tensor 't' has shape [9]
reshape(t, [3, 3]) ==> [[1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 9]]
reshape 는 (데이터 , 원하는 shape)를 인자로 받아 모양을 바꿔준다.
이 때 -1은 특별한 역할을 해주는데 -1이라고 명시를 하면 어떤 수든지 올 수 있다. 
위의 식에서도 [-1,7] 이 되면 shape가 (?,7)이 되게끔 만들어준다.
"""

W = tf.Variable(tf.random_normal([16,nb_classes]),name = 'weight')
b = tf.Variable(tf.random_normal([nb_classes]),name = 'bias')

logits = tf.matmul(X,W)+b
hypothesis = tf.nn.softmax(logits)

cost_i = tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = Y_one_hot)
cost = tf.reduce_mean(cost_i)
# cost = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(hypothesis),axis=1))
# cost 함수를 구하는 방법은 다음과 동일하다.

Optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.1).minimize(cost)

prediction = tf.argmax(hypothesis,1)
# hypothesis 는 softmax를 거쳤기 때문에 0~1사이의 값을 가진다.
# 이를 다시 0~6사이의 값으로 바꿔준다.
# 바꾸는 과정은 argmax( ,1)을 통해서 열기준 가장 큰 index를 return 한다.

correct_prediction = tf.equal(prediction, tf.argmax(Y_one_hot,1))
# Y_one_hot에 대하여 argmax(,1)을 해주니 원래의 Y값이 나오고
# 이것이 prediction 과 비교하며 값을 return 한다.

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# 이를 평균내면 정확도가 나온다.

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(2000):
        sess.run(Optimizer, feed_dict={X:x_data, Y:y_data})
        if step % 100 == 0:
            loss , acc = sess.run([cost,accuracy], feed_dict={
                X: x_data, Y: y_data})
            print("step:{:5}\tLoss:{:3f}\tAcc:{:.2%}".format(
                step, loss, acc))

    pred = sess.run(prediction, feed_dict={X:x_data})
    for p, y in zip(pred, y_data.flatten()):
        # flatten 은 말 그대로 평평화 작업이다.
        # ex) [[1],[0]] --> [1,0]
        # zip 은 동일한 개수로 이루어진 자료형을 묶어주는 역할을 한다.
        # ex) list(zip([1, 2, 3], [4, 5, 6], [7, 8, 9])) --> [(1, 4, 7), (2, 5, 8), (3, 6, 9)]
        print("[{}] prediction : {} True Y: {}".format(p==int(y),p,int(y)))

