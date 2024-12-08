
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
#4-1 

def logistic_regression(X, b, b0):
    return 1. / (1. + np.exp(-np.dot(X, b) - b0))

def accuracy(y_pred, y_true):
    correction_prediction = np.equal(np.round(y_pred), y_true.astype(np.int64))
    return np.mean(correction_prediction.astype(np.float32))

def logistic_regression_wo_vectorization(x_test, b, b0):
    pred = list()
    for t in x_test:
        pred.append(logistic_regression(t, b, b0))
    return pred

''' 단항 분리 0과 1만 분리 '''
num_feacture = 784 # 28* 28

learning_rate = 0.0001 ## 넘어가는 사이즈
training_step = 50 ## 오차가 0이 되는 지점을 탐색하면 좋겠지만 불가능하기 떄문에 충분히 큰수로 지정

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train, y_train = map(list, zip(*[(x,y) for x, y in zip(x_train, y_train) if y ==0 or y== 1]))
x_test, y_test = map(list, zip(*[[x,y] for x, y in zip(x_train, y_train) if y ==0 or y== 1]))

x_train, x_test = np.array(x_train, np.float32), np.array(x_test, np.float32)
y_train, y_test = np.array(y_train, np.float32), np.array(y_test, np.float32)

x_train, x_test = x_train.reshape([-1, num_feacture]), x_test.reshape([-1, num_feacture])
x_train, x_test = x_train / 255.0, x_test / 255.0 ## 이미지 정규화(255가 검은색, 0이 흰색)

b= np.random.uniform(-1, 1, num_feacture) ## 784개의 원소값을 가지는 벡터 생성
b0 = np.random.uniform(-1, 1)

# (1)
for step in range(training_step): 
    db = np.zeros(num_feacture, dtype='float32')
    db0 = 0.
    for x, y in zip(x_train, y_train):
        a = logistic_regression(x, b, b0)
        db += (y - a) * x  # 수정된 부분
        db0 += y - a
    b += learning_rate * db
    b0 += learning_rate * db0

pred = logistic_regression_wo_vectorization(x_test, b, b0)
print("Accuracy : ", accuracy(pred, y_test))

# (2) : 벡터 변경
for step in range(training_step):
    a = logistic_regression(x_train, b, b0)
    
    # 벡터화된 업데이트 
    db = np.dot(x_train.T, (y_train - a))  # (num_feature,)
    db0 = np.sum(y_train - a)  # 스칼라
    
    b += learning_rate * db
    b0 += learning_rate * db0

pred = logistic_regression(x_test, b, b0)
print("Accuracy: ", accuracy(pred, y_test))

# (3)
pred = np.vectorize(logistic_regression, signature='(n),(n),()->()')(x_test, b, b0)
print("Accuracy: ", accuracy(pred, y_test))