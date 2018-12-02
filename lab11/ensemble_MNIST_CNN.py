import tensorflow as tf 
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)

learning_rate = 0.01
training_epochs = 20
batch_size = 100

class Model:
# class 로 만들어서 했던 이유는 ensemble을 하기 위해서

    def __init__(self, sess, name):
        self.sess = sess
        self.name = name
        self._build_net()

    def _build_net(self):
        with tf.variable_scope(self.name):
        # with를 사용하면 중간에 애러가 나더라도 close해준다.
            self.training = tf.placeholder(tf.bool)

            self.X = tf.placeholder(tf.float32, [None , 784])

            X_img = tf.reshape(self.X, [-1,28,28,1])
            self.Y = tf.placeholder(tf.float32, [None, 10])
            
            # Convolutional Layer #1
            conv1 = tf.layers.conv2d(inputs = X_img, filters = 32, kernel_size = [3,3],
                                    padding = "SAME", activation = tf.nn.relu)

            pool1 = tf.layers.max_pooling2d(inputs = conv1 , pool_size = [2,2],
                                            padding= "SAME", strides = 2)

            dropout1 = tf.layers.dropout(inputs = pool1, rate = 0.3 , training= self.training)

            # Convolutional Layer #2
            conv2 = tf.layers.conv2d(inputs = dropout1, filters = 64, kernel_size = [3,3],
                                    padding = "SAME", activation = tf.nn.relu)

            pool2 = tf.layers.max_pooling2d(inputs = conv2 , pool_size = [2,2],
                                            padding = "SAME", strides = 2)  

            dropout2 = tf.layers.dropout(inputs = pool2 , rate = 0.3, training = self.training)

            # Convolutional Layer #3
            conv3 = tf.layers.conv2d(inputs = dropout2, filters = 128, kernel_size = [3,3],
                                    padding = "SAME", activation = tf.nn.relu)

            pool3 = tf.layers.max_pooling2d(inputs = conv3, pool_size = [2,2],
                                            padding = "SAME", strides = 2)
            
            dropout3 = tf.layers.dropout(inputs = pool3 , rate = 0.3 , training = self.training)

            # Dense Layer with Relu
            flat = tf.reshape(dropout3, [-1, 128*4*4])
            dense4 = tf.layers.dense(inputs = flat, units = 625, activation = tf.nn.relu)
            dropout4 = tf.layers.dropout(inputs = dense4, rate = 0.5, training = self.training )

             # Logits (no activation) Layer: L5 Final FC 625 inputs -> 10 outputs
            self.logits = tf.layers.dense(inputs = dropout4, units = 10)

        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits = self.logits, labels = self.Y))

        self.optimizer = tf.train.AdamOptimizer(
            learning_rate = learning_rate).minimize(self.cost)

        correct_prediction = tf.equal(
            tf.argmax(self.logits, 1), tf.argmax(self.Y, 1))
        
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

   

    def predict(self, x_test, training=False):
        return self.sess.run(self.logits,
                             feed_dict={self.X: x_test, self.training: training})

    def get_accuracy(self, x_test ,y_test, training = False):
        return self.sess.run(self.accuracy,
                             feed_dict={self.X: x_test,
                                        self.Y: y_test, self.training: training})

    def train(self, x_data , y_data , training = True):
        return self.sess.run([self.cost, self.optimizer], feed_dict={
            self.X : x_data, self.Y : y_data, self.training : training})

sess = tf.Session()

models = [] # model들을 담을 list
num_models = 2 # model의 갯수
for m in range(num_models):
    models.append(Model(sess, "model"+ str(m)))

sess.run(tf.global_variables_initializer())

print("learning started")

for epoch in range(training_epochs):
    avg_cost_list = np.zeros(len(models))
    # 각 모델의 cost를 저장할 리스트
    # 0을 원소로 하고 개수가 models의 개수와 동일한 리스트
    total_batch = int(mnist.train.num_examples / batch_size) # 55000 / 100 = 550
    for i in range(total_batch):
        batch_xs , batch_ys = mnist.train.next_batch(batch_size)

        for m_idx, m in enumerate(models):
        # model이 여러 개 이기 때문에 반복문으로 각각 학습시킨다.
            c, _ = m.train(batch_xs, batch_ys)
            avg_cost_list[m_idx] += c / total_batch
            # total batch(550) 만큼 반복을 하게 되는데 각각의 batch_xs와 ys의 해당하는 
            # cost를 550으로 나누고 더해주는 과정을 550번 걸쳐 평균을 구하게 된다.

    print('Epoch:', '%04d' % (epoch + 1), 'cost =', avg_cost_list)

print('Learning Finished!')

test_size = len(mnist.test.labels) # minst.test.labels의 shape가 (10000,10) 이므로 len의 값은 10000
predictions = np.zeros([test_size , 10]) #  (10000,10) 크기의 원소가 0인 2차원 배열
for m_idx, m in enumerate(models): # m_idx는 0부터 순차적으로, m은 models 안에 있는 원소를 순회
    print(m_idx, 'Accuracy:', m.get_accuracy(
        mnist.test.images, mnist.test.labels))
    p = m.predict(mnist.test.images) # mnist.test.images = 10000
    print(p)
    predictions += p # predictions에는 각 모델의 predict들이 반복적으로 더해진다.
"""
predictions의 모습
      0 1 2 3 4 5 6 7 8 9
1
2
3
.
.
.  
10000

여기에 p를    

여기부분 잘모르겟다아아아
"""    

ensemble_correct_prediction = tf.equal(
    tf.argmax(predictions, 1), tf.argmax(mnist.test.labels, 1))
ensemble_accuracy = tf.reduce_mean(
    tf.cast(ensemble_correct_prediction, tf.float32))
print('Ensemble accuracy:', sess.run(ensemble_accuracy))