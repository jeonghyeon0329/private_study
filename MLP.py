
import os
import tensorflow as tf
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # GPU 비활성화

'''
mnist dataset: 
    x_train : 총 60,000개의 28x28 크기의 이미지
    y_train : 총 60,000개의 이미지 레이블
    x_test : 총 10,000개의 이미지
    y_test : 총 10,000개의 이미지 레이블

	수기로 작성된 0-9까지 데이터
	0이면 가장 검은색 255면 가장 밝은 값    
'''
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

##  이미지 데이터를 255로 나누어 0과 1 사이로 변환하는 것은 신경망 모델이 효과적으로 학습할 수 있도록 돕는 중요한 전처리 단계입니다.
x_train, x_test = x_train / 255.0, x_test / 255.0

# model = tf.keras.models.Sequential([
#     tf.keras.layers.Flatten(input_shape = (28,28)), ## 2D 이미지를 1D 벡터로 변환
#     tf.keras.layers.Dense(239, activation='relu'), ## 239개의 노드를 가진 완전 연결(Dense) 레이어
#     tf.keras.layers.Dropout(0.2), ## 20%의 노드를 무작위로 제거하여 과적합(overfitting)을 방지
#     tf.keras.layers.Dense(32, activation='relu'),  ## 32개의 노드를 가진 완전 연결(Dense) 레이어 (RELU 활성화 함수)
#     tf.keras.layers.Dropout(0.2), ## 또 다른 20%의 노드를 무작위로 제거
#     tf.keras.layers.Dense(10) ## 최종 레이어는 10개의 노드를 가진 완전 연결(Dense) 레이어입니다. 이 레이어는 MNIST 데이터셋의 10개의 클래스를 예측하기 위한 출력 레이어입니다. 
# ])

model = tf.keras.models.Sequential([
    tf.keras.layers.Rescaling(1./255, input_shape=(28, 28, 1)),  # 이미지 리스케일링
    tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])

loss_func = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True)
model.compile(optimizer = 'adam', loss = loss_func, metrics = ['accuracy'])
model.fit(x_train, y_train, epochs = 10)
print(model.evaluate(x_test, y_test))
