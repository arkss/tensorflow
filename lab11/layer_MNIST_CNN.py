# 기존의 신경망을 layer 라는 새로운 패키지를 통해서 구현
# 대부분의 내용은 class_MNIST_CNN 과 동일

import tensorflow as tf 

from tensorflow.examples.tutorials.mnist import input_data 

mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)

learning_rate = 0.01
training_epochs = 15
batch_sizes = 100

class Model:

    def __init__(self, sess, name):
        self.sess = sess
        self.name = name
        self._build_net()

    def _build_net(self):
        with tf.variable_scope(self.name):
            self.training = tf.placeholder(tf.bool)
            # training 이라는 bool 값을 담는 변수를 선언
            # training 할 때와 test 할 때 dropout 하는 게 다르므로 그 부분을 결정해준다.

            self.X = tf.placeholder(tf.float32, [None,784])

            X_img = tf.reshape(self.X, [-1,28,28,1])
            self.Y = tf.placeholder(tf.float32, [None, 10])

            conv1 = tf.layers.conv2d(inputs = X_img, filters = 32, kernel_size = [3,3],
                                            padding = 'SAME', activation = tf.nn.relu)
            # 매개변수 : inputs 데이터 / 필터의 개수 / 커널의 사이즈 / padding / 활성함수
            pool1 = tf.layers.max_pooling2d(inputs = conv1, pool_size= [2,2],
                                            padding = "SAME", strides = 2)
            # 매개변수 : inputs 데이터 / pooling의 사이즈 / padding / strides 크기
            dropout1 = tf.layers.dropout(inputs=pool1, rate = 0.3, training=self.training)   

            # 매개변수 : inputs 데이터 / 드랍아웃할 비율 / 할지말지에 대한 boolean 값
            conv2 = tf.layers.conv2d(inputs = dropout1, filters = 64, kernel_size = [3,3],
                                            padding = 'SAME', activation = tf.nn.relu)

            pool2 = tf.layers.max_pooling2d(inputs = conv2, pool_size = [2,2],
                                            padding = "SAME", strides = 2)  

            dropout2 = tf.layers.dropout(inputs = pool2, rate = 0.3, training= self.training)                                                           

            conv3 = tf.layers.conv2d(inputs = dropout2, filters = 128, kernel_size = [3,3],
                                            padding = "SAME", activation = tf.nn.relu)
            
            pool3 = tf.layers.max_pooling2d(inputs = conv3, pool_size = [2,2],
                                            padding = "SAME", strides = 2)

            dropout3 = tf.layers.dropout(inputs = pool3, rate = 0.3, training = self.training)                                


            flat = tf.reshape(dropout3 , [-1, 128*4*4])
            dense4 = tf.layers.dense(inputs = flat, units = 625, activation = tf.nn.relu)
            dropout4 = tf.layers.dropout(inputs = dense4, rate = 0.5, training = self.training)

            self.logits = tf.layers.dense(inputs = dropout4, units = 10)
        
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits = self.logits, labels = self.Y))
        self.optimizer = tf.train.AdamOptimizer(
            learning_rate = learning_rate).minimize(self.cost)
        
        correct_prediction = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def predict(self, x_test, training=False):
        return self.sess.run(self.logits,
                            feed_dict = {self.X : x_test, self.training : training})

    def get_accuracy(self, x_test, y_test, training = False):
        return self.sess.run(self.accuracy,
                            fedd_dict = {self.X : x_test, self.Y: y_test, self.training : training})
    
    def train(self, x_data , y_data , training = True):
        return self.sess.run([self.cost, self.optimizer], feed_dict= {
            self.X : x_data, self.Y: y_data, self.training : training})
        

sess = tf.Session()
m1 = Model(sess, "m1")

sess.run(tf.global_variables_initializer())

print('learning started!')

for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = int(mnist.train.num_examples / batch_sizes )

    for i in range(total_batch):
        batch_xs , batch_ys = mnist.train.next_batch(batch_sizes)
        c, _ = m1.train(batch_xs, batch_ys)
        avg_cost += c / total_batch

    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))

print('Learning Finished!')

# Test model and check accuracy
print('Accuracy:', m1.get_accuracy(mnist.test.images, mnist.test.labels))