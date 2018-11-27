import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_Data/", one_hot = True)

# hyper parameters 설정
# hyper parameters : 학습 프로세스가 시작되기 전에 값이 설정되는 매개 변수
learning_rate = 0.001
training_epochs = 15
batch_sizes = 100

class Model:

    def __init__(self, sess, name):
        self.sess = sess
        self.name = name
        self._build_net()

    def _build_net(self):
        with tf.variable_scope(self.name):
            self.keep_prob = tf.placeholder(tf.float32)

            self.X = tf.placeholder(tf.float32, [None, 784])
            X_img = tf.reshape(self.X, [-1,28,28,1])
            self.Y = tf.placeholder(tf.float32, [None, 10])
            
            # conv layer1
            W1 = tf.Variable(tf.random_normal([3,3,1,32], stddev =0.01))
            # filter 설정
            # 3*3 의 크기 / 색상은 흑백 : 1 / 32개의 다른 filter 지정
            L1 = tf.nn.conv2d(X_img,W1, strides = [1,1,1,1], padding = 'SAME')
            # shape = (?, 28,28,32)
            L1 = tf.nn.relu(L1)
            # relu 는 shape 에 변화를 주지 않는다.
            L1 = tf.nn.max_pool(L1, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')
            # ksize 2*2 strides 2*2 
            # shape = (?,14,14,32)
            L1 = tf.nn.dropout(L1, keep_prob = self.keep_prob)

            # conv layer2
            W2 = tf.Variable(tf.random_normal([3,3,32,64],stddev = 0.01))
            # 위와 크기 동일 depth 는 input data인 L1과 맞춰줘야함
            L2 = tf.nn.conv2d(L1, W2 , strides = [1,1,1,1], padding = 'SAME')
            # shape = (?,14,14,64)
            L2 = tf.nn.relu(L2)
            L2 = tf.nn.max_pool(L2, ksize = [1,2,2,1],
                                strides = [1,2,2,1], padding = 'SAME')
            # shape = (?,7,7,64)
            L2 = tf.nn.dropout(L2, keep_prob = self.keep_prob)

            # conv layer3
            W3 = tf.Variable(tf.random_normal([3,3,64,128], stddev = 0.01))
            L3 = tf.nn.conv2d(L2, W3, strides = [1,1,1,1], padding = 'SAME')
            # shape = (?,7,7,128)
            L3 = tf.nn.relu(L3)
            L3 = tf.nn.max_pool(L3, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')
            # shape = (?,4,4,128)
            L3 = tf.nn.dropout(L3, keep_prob = self.keep_prob)

            L3_flat = tf.reshape(L3, [-1, 128*4*4])
            # 평탄화 작업
            # shape = (?, 4*4*128)

            # fully connected layer1 : 4*4*128 -> 625
            W4 = tf.get_variable("W4", shape = [128*4*4, 625],
                                initializer = tf.contrib.layers.xavier_initializer())
            b4 = tf.Variable(tf.random_normal([625]))
            L4 = tf.nn.relu(tf.matmul(L3_flat, W4)+b4)
            L4 = tf.nn.dropout(L4, keep_prob = self.keep_prob)

            # fully connected layer2 : 625 -> 10
            W5 = tf.get_variable("W5", shape = [625, 10],
                                initializer = tf.contrib.layers.xavier_initializer())
            b5 = tf.Variable(tf.random_normal([10]))
            self.logits = tf.matmul(L4, W5)+ b5

            self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                logits = self.logits, labels = self.Y))
            self.optimizer = tf.train.AdamOptimizer(
                learning_rate = learning_rate).minimize(self.cost)

            correct_prediction = tf.equal(
                tf.argmax(self.logits,1), tf.argmax(self.Y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def predict(self, x_test, keep_prop=1.0):
        return self.sess.run(self.logits, feed_dict = {self.X:x_test, self.keep_prob: keep_prop})

    def get_accuracy(self, x_test, y_test, keep_prop = 1.0):
        return self.sess.run(self.accuracy, feed_dict = {self.X:x_test, self.Y : y_test, self.keep_prob:keep_prop})

    def train(self, x_data, y_data, keep_prop = 0.7):
        return self.sess.run([self.cost,self.optimizer], feed_dict= {
            self.X: x_data, self.Y:y_data, self.keep_prob: keep_prop})

sess = tf.Session()
m1 = Model(sess, "m1")

#sess.run(tf.global_variables_intializer())
sess.run(tf.global_variables_initializer())

print("learning started!")

for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = int(mnist.train.num_examples / batch_sizes)

    for i in range(total_batch):           
        batch_xs, batch_ys = mnist.train.next_batch(batch_sizes)
        c, _ = m1.train(batch_xs, batch_ys)
        avg_cost += c / total_batch

    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))

print('learning finished')
print('Accuracy:', m1.get_accuracy(mnist.test.images, mnist.test.labels))
