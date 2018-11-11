import tensorflow as tf

a = tf.constant([[3, 10, 1],[4, 5, 6],[0, 8, 7]])

session = tf.Session()
print('a:\n', session.run(a))
print('인덱스의 개수 = ', session.run(tf.rank(a)))
print('tf.argmax(a, 0): 인덱스 ', session.run(tf.argmax(a, 0))) # 열기준
print('tf.argmax(a, 1): 인덱스 ', session.run(tf.argmax(a, 1) )) # 행기준

a2 = tf.constant([[[1, 3, 5],
                   [3, 5, 1]],
                  [[5, 1, 3],
                   [1, 3, 5]],
                  [[3, 5, 1],
                   [5, 1, 3]]])

print('a2:\n', session.run(a2))
print('인덱스의 개수 = ', session.run(tf.rank(a2)))
print('tf.argmax(a2, 0): 인덱스\n ', session.run(tf.argmax(a2, 0)))
print('tf.argmax(a2, 1): 인덱스\n ', session.run(tf.argmax(a2, 1)))
print('tf.argmax(a2, 2): 인덱스\n ', session.run(tf.argmax(a2, 2)))




