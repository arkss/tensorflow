import tensorflow as tf
import numpy as np



from tensorflow.examples.tutorials.mnist import input_data
# https://www.tensorflow.org/get_started/mnist/beginners

mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)


print("mnist.test.num_examples:",mnist.test.num_examples) # 10000
print("mnist.train.num_examples:",mnist.train.num_examples) # 55000
print("mnist.validation.num_examples:",mnist.validation.num_examples) # 5000

print("np.shape(mnist.train.images):",np.shape(mnist.train.images))  # (55000, 784)
print("np.shape(mnist.train.labels):",np.shape(mnist.train.labels))  # (55000, 10)

print("np.shape(mnist.test.images):", np.shape(mnist.test.images))  # (10000, 784)
print("np.shape(mnist.test.labels):",np.shape(mnist.test.labels))  # (10000, 10)

print("np.shape(mnist.test.labels):",len(mnist.test.labels))  # 10000