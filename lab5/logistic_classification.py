import tensorflow as tf

x_data = [[1,2],[2,3],[3,1],[4,3],[5,3],[6,2]]
y_data = [[0],[0],[0],[1],[1],[1]]

X = tf.placeholder(tf.float32, shape=[None,2])
Y = tf.placeholder(tf.float32, shape=[None,1])
W = tf.Variable(tf.random_normal([2,1]),name = 'weight')
b = tf.Variable(tf.random_normal([1]),name = 'bias')

hypothesis = tf.sigmoid(tf.matmul(X,W)+b)
# 다음과 같이 sigmoid 라고 적을 수 있고 원래의 식 형태로 적을 수도 있다.
# tf.div(1. , 1. + tf.exp(tf.matmul(X,W)+b))

cost = -tf.reduce_mean(Y*tf.log(hypothesis)+ (1-Y)*tf.log(1-hypothesis))

train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
# hypothesis 가 0.5 보다 크면 True 그렇지 않으면 False 를 반환
# 그리고 dtype 을 float32 로 지정해주면 각 1,0 으로 바뀐다.

accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted,Y), dtype=tf.float32))
# predicted 와 Y 를 비교해서 같으면 True 다르면 False 반환
# 마찬가지로 1,0 으로 바꿔준 후 평균을 구해주면 정확도를 예측가능하다.

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(10001):
        cost_val, _ = sess.run([cost,train],feed_dict={X:x_data, Y: y_data})
        if step % 200 == 0:
            print(step,cost_val)

    h,c,a = sess.run([hypothesis, predicted, accuracy],
        feed_dict={X:x_data, Y:y_data})

    print("\nhypothesis:",h,"\ncorrect (Y):",c,"\nAccuracy: ", a)