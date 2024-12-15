
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

# NumPy 배열 생성
np_array = np.array([[1, 2, 3], [4, 5, 6]])
print("NumPy Array:\n", np_array)

# TensorFlow 텐서 생성
tf_tensor = tf.constant([[1, 2, 3], [4, 5, 6]])
print("TensorFlow Tensor:\n", tf_tensor)

# NumPy 크기/차원 확인
print("NumPy Shape:", np_array.shape)
print("NumPy Dimensions:", np_array.ndim)

# TensorFlow 크기/차원 확인
print("TensorFlow Shape:", tf_tensor.shape)
print("TensorFlow Dimensions:", tf_tensor.ndim)

# NumPy 수평/수직 결합
np_v1 = np.array([1, 2, 3])
np_v2 = np.array([4, 5, 6])

print("NumPy - Vertical Stack:\n", np.vstack([np_v1, np_v2]))
print("NumPy - Horizontal Stack:\n", np.hstack([np_v1, np_v2]))

# TensorFlow 스택 (수직, 수평)
tf_v1 = tf.constant([1, 2, 3])
tf_v2 = tf.constant([4, 5, 6])

print("TensorFlow - Stack (Axis=0):\n", tf.stack([tf_v1, tf_v2], axis=0))
print("TensorFlow - Stack (Axis=1):\n", tf.stack([tf_v1, tf_v2], axis=1))

# NumPy 배열 결합
print("NumPy - Concatenate (Axis=0):\n", np.concatenate([np_v1, np_v2], axis=0))
print("NumPy - Concatenate (Axis=1):\n", np.concatenate([np_v1.reshape(1, -1), np_v2.reshape(1, -1)], axis=1))

# TensorFlow 배열 결합
print("TensorFlow - Concatenate (Axis=0):\n", tf.concat([tf_v1, tf_v2], axis=0))
print("TensorFlow - Concatenate (Axis=1):\n", tf.concat([tf_v1[None, :], tf_v2[None, :]], axis=1))

# NumPy 배열 연산
np_array1 = np.array([1, 2, 3])
np_array2 = np.array([4, 5, 6])

print("NumPy Addition:\n", np_array1 + np_array2)
print("NumPy Subtraction:\n", np_array1 - np_array2)
print("NumPy Multiplication:\n", np_array1 * np_array2)
print("NumPy Division:\n", np_array1 / np_array2)

# TensorFlow 텐서 연산
tf_tensor1 = tf.constant([1, 2, 3])
tf_tensor2 = tf.constant([4, 5, 6])

print("TensorFlow Addition:\n", tf_tensor1 + tf_tensor2)
print("TensorFlow Subtraction:\n", tf_tensor1 - tf_tensor2)
print("TensorFlow Multiplication:\n", tf_tensor1 * tf_tensor2)
print("TensorFlow Division:\n", tf_tensor1 / tf_tensor2)

# NumPy 행렬 연산
np_matrix1 = np.array([[1, 2], [3, 4]])
np_matrix2 = np.array([[5, 6], [7, 8]])

print("NumPy Matrix Multiplication:\n", np.dot(np_matrix1, np_matrix2))
print("NumPy Matrix Transpose:\n", np_matrix1.T)

# TensorFlow 행렬 연산
tf_matrix1 = tf.constant([[1, 2], [3, 4]])
tf_matrix2 = tf.constant([[5, 6], [7, 8]])

print("TensorFlow Matrix Multiplication:\n", tf.matmul(tf_matrix1, tf_matrix2))
print("TensorFlow Matrix Transpose:\n", tf.transpose(tf_matrix1))

# NumPy에서 Uniform Distribution 난수 생성
np_random = np.random.uniform(0, 1, (3, 3))  # 0과 1 사이의 균등 분포
print("NumPy Random Uniform:\n", np_random)

# TensorFlow에서 Uniform Distribution 난수 생성
tf_random = tf.random.uniform((3, 3), minval=0, maxval=1)
print("TensorFlow Random Uniform:\n", tf_random)

# NumPy에서 Normal Distribution 난수 생성
np_normal = np.random.normal(loc=0, scale=1, size=(3, 3))  # 평균 0, 표준편차 1
print("NumPy Random Normal:\n", np_normal)

# TensorFlow에서 Normal Distribution 난수 생성
tf_normal = tf.random.normal((3, 3), mean=0, stddev=1)
print("TensorFlow Random Normal:\n", tf_normal)

# NumPy에서 고유값 계산
np_matrix = np.array([[4, -2], [1,  1]])
np_eigenvalues, np_eigenvectors = np.linalg.eig(np_matrix)
print("NumPy Eigenvalues:\n", np_eigenvalues)
print("NumPy Eigenvectors:\n", np_eigenvectors)

# NumPy에서 역행렬 계산
np_inverse = np.linalg.inv(np_matrix)
print("NumPy Inverse Matrix:\n", np_inverse)

# TensorFlow에서 고유값 계산 (선택적)
tf_matrix = tf.constant([[4.0, -2.0], [1.0,  1.0]])
# TensorFlow는 고유값을 직접 제공하지 않지만, `tf.linalg.eig`를 통해 계산 가능
tf_eigenvalues, tf_eigenvectors = tf.linalg.eig(tf_matrix)
print("TensorFlow Eigenvalues:\n", tf_eigenvalues)
print("TensorFlow Eigenvectors:\n", tf_eigenvectors)

# TensorFlow에서 역행렬 계산
tf_inverse = tf.linalg.inv(tf_matrix)
print("TensorFlow Inverse Matrix:\n", tf_inverse)

# NumPy 정수 나누기 (Floor Division) 및 나머지
np_divide = np.floor_divide(np_array1, np_array2)
np_remainder = np.mod(np_array1, np_array2)
print("NumPy Floor Division:\n", np_divide)
print("NumPy Remainder:\n", np_remainder)

# TensorFlow 정수 나누기 (Floor Division) 및 나머지
tf_divide = tf.math.floordiv(tf_tensor1, tf_tensor2)
tf_remainder = tf.math.mod(tf_tensor1, tf_tensor2)
print("TensorFlow Floor Division:\n", tf_divide)
print("TensorFlow Remainder:\n", tf_remainder)
