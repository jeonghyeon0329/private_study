
import os
import tensorflow as tf
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # GPU 비활성화

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

'''
    6만개의 training, 1만개의 testing 
    28pixel * 28pixel로 이미지 구성
    grayscale : 0이면 가장 어두운 값, 255이면 가장 밝은 값
'''
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train, y_train = map(list, zip(*[(x,y) for x, y in zip(x_train, y_train) if y ==0 or y== 1]))
x_test, y_test = map(list, zip(*[[x,y] for x, y in zip(x_test, y_test) if y ==0 or y== 1]))

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

pred = logistic_regression_wo_vectorization(x_test, b, b0) ## 반올림 했을때 1인지 0인지 구분
print("Accuracy : ", accuracy(pred, y_test))

# 6-1

# (2) : 벡터 변경
for step in range(training_step):
    a = logistic_regression(x_train, b, b0)
    
    # # 벡터화된 업데이트 
    # db = np.dot(x_train.T, (y_train - a))  # (num_feature,)
    # db0 = np.sum(y_train - a)  # 스칼라
    
    # 벡터화된 업데이트
    error = y_train - a  # (num_train_samples, )
    db = np.dot(x_train.T, error)  # (num_feature,)
    db0 = np.sum(error)  # 스칼라
    
    
    
    b += learning_rate * db
    b0 += learning_rate * db0

pred = logistic_regression(x_test, b, b0)
print("Accuracy: ", accuracy(pred, y_test))

# (3)
pred = np.vectorize(logistic_regression, signature='(n),(n),()->()')(x_test, b, b0)
print("Accuracy: ", accuracy(pred, y_test))


''' 다항 분리 '''

num_classes = 10 # 0~9까지 숫자
num_feacture = 784 # 28* 28

# Training parameters.
learning_rate = 0.0001 ## 넘어가는 사이즈
training_step = 50 ## 오차가 0이 되는 지점을 탐색하면 좋겠지만 불가능하기 떄문에 충분히 큰수로 지정
batch_size = 256
display_step = 50

# (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# x_train, x_test = np.array(x_train, np.float32), np.array(x_test, np.float32)
# x_train, x_test = x_train.reshape([-1, num_feacture]), x_test.reshape([-1, num_feacture])
# x_train, x_test = x_train /255. , x_test / 255.

# #(1)
# b= np.random.uniform(-1, 1, num_feacture*num_classes).reshape((num_classes, num_feacture))
# b0 = np.random.uniform(-1, 1, num_classes)

# for step in range(training_step):
#     db = np.zeros((num_classes, num_feacture), dtype= 'float32')
#     db0 = np.zeros(num_classes, dtype='float32')
    
#     for x, y in zip(x_train, y_train):
#         yy = tf.one_hot(y, depth=num_classes).numpy()
#         a= logistic_regression_multi(x, b, b0)
#         db += np.matmul(np.expand_dims(yy-a, axis = 1), np.expand_dims(x, axis = 0))
#         db0 += yy -a
    
#     b += learning_rate * db
#     b0 += learning_rate * db0
    
# pred = np.argmax(np.array([logistic_regression_multi(x, b, b0) for x in x_test]), axis=1)
# print("Accuracy: ", np.mean(pred == y_test))

# # TensorFlow 텐서로 변환(tensorflow-io-gcs-filesystem 0.31.0)
# x_train_tensor = tf.convert_to_tensor(x_train)
# y_train_tensor = tf.convert_to_tensor(y_train, dtype=tf.int32)
# x_test_tensor = tf.convert_to_tensor(x_test)
# y_test_tensor = tf.convert_to_tensor(y_test, dtype=tf.int32)

# # 경사 하강법 학습
# for step in range(training_step):
#     # 배치 처리
#     for start in range(0, x_train.shape[0], batch_size):
#         end = min(start + batch_size, x_train.shape[0])
#         x_batch = x_train_tensor[start:end]
#         y_batch = y_train_tensor[start:end]

#         # 원핫 인코딩
#         yy = tf.one_hot(y_batch, depth=num_classes)

#         # 예측값 계산
#         a = logistic_regression_multi(x_batch, b, b0)

#         # 기울기 계산
#         db = tf.matmul(tf.transpose(yy - a), x_batch)
#         db0 = tf.reduce_sum(yy - a, axis=0)

#         # 파라미터 업데이트
#         b += learning_rate * db
#         b0 += learning_rate * db0

#     # 결과 출력 (Optional)
#     if step % display_step == 0:
#         pred = np.argmax(logistic_regression_multi(x_test, b, b0).numpy(), axis=1)
#         accuracy = np.mean(pred == y_test)
#         print(f"Step {step}, Accuracy: {accuracy:.4f}")

# # 최종 정확도 계산
# pred = np.argmax(logistic_regression_multi(x_test, b, b0).numpy(), axis=1)
# accuracy = np.mean(pred == y_test)
# print("Final Accuracy: ", accuracy)
