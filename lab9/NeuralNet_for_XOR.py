import tensorflow as tf
import numpy as np

x_data = np.array([[0,0],[1,0],[0,1],[1,1]], dtype= np.float32)
y_data = np.array([[0],[1],[1],[0]], dtype = np.float32)

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)
# XOR은  다음과 같이 층을 2개로 만들어서 구현해야한다.
# 이 때 주의해야할 점은 각 W, b의 크기이다.
W1 = tf.Variable(tf.random_normal([2,10]), name = 'weight1')
b1 = tf.Variable(tf.random_normal([10]), name = 'bias1')
layer1 = tf.sigmoid(tf.matmul(X,W1)+b1)
# 여기서 10은 사용자가 임의로 지정해주면 된다.
# 단 주의해야할 점은 10으로 나갔으면 W2에서 받을 때도 10으로 받기만 하면 된다.
# 지금은 10의 wide, 2의 layer로 학습을 시켰지만 wide를 넓게 하고 layer를 깊게 할수록 더 정확하게 학습이 되는걸 볼 수 있다.
W2 = tf.Variable(tf.random_normal([10,1]), name = 'weight2')
b2 = tf.Variable(tf.random_normal([1]), name = 'bias2')
hypothesis = tf.sigmoid(tf.matmul(layer1,W2)+b2)

cost = -tf.reduce_mean(Y*tf.log(hypothesis)+ (1-Y)*tf.log(1-hypothesis))
train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

predicted = tf.cast(hypothesis > 0.5, dtype= tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted,Y), dtype=tf.float32))
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(10001):
        sess.run(train, feed_dict={X:x_data, Y:y_data})
        if step % 100 == 0:
            print(step, sess.run(cost, feed_dict={X:x_data, Y:y_data}), sess.run([W1,W2]))

    h,c,a = sess.run([hypothesis, predicted, accuracy], feed_dict={X:x_data, Y:y_data})
    print("\nhypothesis:",h,"\ncorrect:",c,"\naccuracy:",a)