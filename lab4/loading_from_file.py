import tensorflow as tf
import numpy as np


xy = np.loadtxt('data-01-test-score.csv',delimiter=',',dtype = np.float32)
# 다음과 같은 방식으로 파일을 불러올 수 있지만 단점은 타입을 하나로 밖에 지정을 못한다.
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]
# 불러온 데이터를 읽는 방법을 슬라이싱을 이용해서 한다.
# , 를 기준으로 앞은 모든 행의 데이터를 가져온다는 뜻이고 
# , 를 기준으로 뒤는 한 행에 대해서 데이터를 어떻게 가져올지
print(x_data.shape, x_data, len(x_data))
print(y_data.shape, y_data)

X = tf.placeholder(tf.float32, shape=[None,3])
Y = tf.placeholder(tf.float32, shape=[None,1])

W = tf.Variable(tf.random_normal([3,1]), name = 'weight')
b = tf.Variable(tf.random_normal([1]), name = 'bias')

hypothesis = tf.matmul(X, W) + b

cost = tf.reduce_mean(tf.square(hypothesis - Y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate = 1e-5)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(2001):
    cost_val, hy_val, _ = sess.run(
        [cost, hypothesis, train],
        feed_dict = {X: x_data, Y: y_data})
    if step % 100 == 0:
        print(step, "cost: ", cost_val, "\nprediction:\n",hy_val)

# 이를 기준으로 새로운 데이터가 들어왔을 때 어떤 결과를 가질 지 확인해보자.

print("your score will be", sess.run(hypothesis, feed_dict={X:[[100,70,101]]}))