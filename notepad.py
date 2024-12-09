
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # GPU 비활성화

''' numpy vs tensorflow'''
import numpy as np
import tensorflow as tf

a = np.array([1,2,3], dtype='int32') ## 1차원, 크기 : 3
b = np.array([[9.0,8.0,7.0],[6.0,5.0,4.0]]) 

a = tf.constant([1,2,3], dtype='int32') 

v1 = np.array([1,2,3,4])
v2 = np.array([5,6,7,8])

print(np.vstack([v1,v2,v1,v2])) # 이어붙이기만 가능
print(np.hstack([v1,v2,v1,v2])) # 이어붙이기만 가능
print()
print(tf.stack([v1,v2,v1,v2], axis = 0))
# print(tf.concat([v1,v2,v1,v2], axis = 1))

print(np.random.uniform(0, 1, (4,2)))
print(tf.random.uniform((4,2), 0, 1))

print(np.identity(5))
print()
print(tf.eye(5))