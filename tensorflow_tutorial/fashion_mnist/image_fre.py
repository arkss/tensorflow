import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

print('train_images.shape : ', train_images.shape) # (60000,28,28)
print('len(train_labels) : ',len(train_labels)) # 60000
print('train_label : ',train_labels) # [9 0 0 ... 3 0 5]
print('test_images.shape : ', test_images.shape) #(10000, 28, 28)
print('test_labels : ', test_labels) # [9 2 1 ... 8 1 5]


plt.figure()
plt.imshow(train_images[10])
plt.colorbar() # 우측에 색깔 막대
plt.grid(False) # 격자제거
plt.show()

train_images = train_images / 255.0
test_images = test_images / 255.0

print('train_images.shape : ', train_images.shape) # (60000,28,28)
print('len(train_labels) : ',len(train_labels)) # 60000
print('train_label : ',train_labels) # [9 0 0 ... 3 0 5]
print('test_images.shape : ', test_images.shape) #(10000, 28, 28)
print('test_labels : ', test_labels) # [9 2 1 ... 8 1 5]


plt.figure()
plt.imshow(train_images[10])
plt.colorbar() # 우측에 색깔 막대
plt.grid(False) # 격자제거
plt.show()