import tensorflow as tf

x_data = [[1,2,1,1],[2,1,3,2],[3,1,3,4],[4,1,5,5],[1,7,5,5],[1,2,5,6],[1,6,6,6],[1,7,7,7]]
y_data = [[0,0,1],[0,0,1],[0,0,1],[0,1,0],[0,1,0],[0,1,0],[1,0,0],[1,0,0]]
# y데이터의 값을 one hot incoding 하였다.
# 이럴 경우 y데이터의 안 쪽 리스트 안에 원소의 갯수가 label의 개수와 같다.
X = tf.placeholder("float",[None,4])
Y = tf.placeholder("float",[None,3])
nb_classes = 3

W = tf.Variable(tf.random_normal([4,nb_classes]), name = 'weight')
b = tf.Variable(tf.random_normal([nb_classes]), name = 'bias')

hypothesis = tf.nn.softmax(tf.matmul(X,W)+b)

cost = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(hypothesis), axis = 1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.1).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(2001):
        sess.run(optimizer, feed_dict={X: x_data, Y:y_data})
        if step % 200 == 0:
            print(step, sess.run(cost, feed_dict={X:x_data, Y:y_data}))

#testing

a = sess.run(hypothesis , feed_dict={X:[[1,11,7,9]]})
print(a, sess.run(tf.arg_max(a,1)))

# a값은 0~1 사이의 값으로 리스트안 4개의 원소의 합이 1이 되게끔 나온다.
# arg_max 는 그 중 가장 큰 값을 찾아 index 을 return 해준다. 여기서는 1이 된다.