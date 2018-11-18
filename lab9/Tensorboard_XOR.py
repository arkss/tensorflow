# 기존에는 print 로 값을 출력하면서 결과를 확인했지만 tensorboard를 사용하면 이를 그래프로 나타낼 수 있다.


import tensorflow as tf 
import numpy as np 

x_data = np.array([[0,0],[1,0],[0,1],[1,1]],dtype = np.float32)
y_data = np.array([[0],[1],[1],[0]],dtype = np.float32)

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

with tf.name_scope("layer1"):
    W1 = tf.Variable(tf.random_normal([2,10]),name = 'weight1')
    b1 = tf.Variable(tf.random_normal([10]), name = 'bias1')
    layer1 = tf.sigmoid(tf.matmul(X,W1)+b1)

    W1_hist = tf.summary.histogram("weights1",W1)
    b1_hist = tf.summary.histogram("biases1", b1)
    layer1_hist = tf.summary.histogram("layer1",layer1)

with tf.name_scope("layer2"):
    W2 = tf.Variable(tf.random_normal([10,1]), name = 'weight2')
    b2 = tf.Variable(tf.random_normal([1]), name = 'bias2')
    hypothesis = tf.sigmoid(tf.matmul(layer1,W2)+b2)

    W2_hist = tf.summary.histogram("weights2",W2)
    b2_hist = tf.summary.histogram("biases2",b2)
    hypothesis_hist = tf.summary.histogram("hypothesis",hypothesis)

with tf.name_scope("cost"):
    cost = -tf.reduce_mean(Y*tf.log(hypothesis)+(1-Y)*tf.log(1-hypothesis))
    cost_summ = tf.summary.scalar("cost",cost)

with tf.name_scope("train"):
    train = tf.train.GradientDescentOptimizer(learning_rate = 0.1).minimize(cost)

predicted = tf.cast(hypothesis > 0.5 , dtype = tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted,Y), dtype = tf.float32))
accuracy_summ = tf.summary.scalar("accuracy", accuracy)

with tf.Session() as sess:

    merged_summary = tf.summary.merge_all()
    # merge 작업은 Session을 열고 하던 그 전에 하던 큰 상관없다.

    #writer = tf.summary.FileWriter("./logs/xor_logs_r0_01")
    writer = tf.summary.FileWriter("./logs/xor_logs_r0_1")
    writer.add_graph(sess.graph)

    sess.run(tf.global_variables_initializer())

    for step in range(10001):
        summary,_ = sess.run([merged_summary,train], feed_dict = {X:x_data, Y:y_data})
        writer.add_summary(summary, global_step=step)

        if step % 100 == 0:
            print(step, sess.run(cost, feed_dict={X:x_data , Y:y_data}),
                '\n', sess.run([W1,W2]))

    h,c,a = sess.run([hypothesis, predicted, accuracy], feed_dict={X:x_data, Y:y_data})
    print("\nhypothesis:",h,"\ncorrect:",c,"\naccuracy:",a)


# tensorboard 실행시키는 방법
#python3 -m tensorflow.tensorboard --logdir=./logs/xor_logs
# = 뒤로는 현재 terminal 에서 있는 곳으로 부터 데이터가 있는 곳으로의 경로
# 데이터는 다음 py 파일이 있는 곳과 같은 디렉토리에 생성 된다. 
# 접속하는 방법은 localhost:6006

# 두 개의 데이터에 대하여 run 해주기 위해서는
# python3 -m tensorflow.tensorboard --logdir=./logs
# logs 안에서는 running rate 를 0.01 / 0.1 로 다르게 설정해준 데이터가 각각 들어있다.