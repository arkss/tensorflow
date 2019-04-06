import tensorflow as tf
from tensorflow import keras

from keras.preprocessing import image
import os
import matplotlib.pyplot as plt

cat_length = len(os.listdir('/Users/rkdalstjd9/Desktop/tensorflow/dogs/images/Cat'))
dog_length = len(os.listdir('/Users/rkdalstjd9/Desktop/tensorflow/dogs/images/Dog'))
print(cat_length) # 2404
print(dog_length) # 4991

img_data = image.load_img('images/Dog/pomeranian_33.jpg')
plt.imshow(img_data)
plt.show()
