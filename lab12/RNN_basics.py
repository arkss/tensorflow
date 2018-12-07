import numpy as np
import tensorflow as tf

hidden_size = 2
cell = tf.contrib.rnn.BasicLSTMCell(num_units = hidden_size)

h = [1,0,0,0]
e = [0,1,0,0]
l = [0,0,1,0]
o = [0,0,0,1]

x_data = np.array([[h,e,l,l,o],
                   [e,o,l,l,l]], dtype = np.float32)
print(x_data)
outputs, _states = tf.nn.dynamic_rnn(cell, x_data, dtype = tf.float32)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
print(outputs.eval(session = sess))

