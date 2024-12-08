
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # GPU 비활성화

''' numpy vs tensorflow'''
import numpy as np
import tensorflow as tf

a = np.array([1,2,3], dtype='int32') ## 1차원, 크기 : 3
b = np.array([[9.0,8.0,7.0],[6.0,5.0,4.0]]) 

a = tf.constant([1,2,3], dtype='int32') 
print(a)
