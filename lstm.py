import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

# 예시 데이터 (문장과 그에 해당하는 레이블)
texts = ['I love programming', 'I hate bugs', 'Python is awesome', 'I hate errors', 'I enjoy learning new things']
labels = [1, 0, 1, 0, 1]  # 1: 긍정, 0: 부정

# 텍스트 전처리
tokenizer = Tokenizer(num_words=10000)  # 사용할 단어의 수 제한
tokenizer.fit_on_texts(texts)

# 문장을 숫자 시퀀스로 변환
sequences = tokenizer.texts_to_sequences(texts)

# 시퀀스의 길이를 맞추기 위해 padding
X = pad_sequences(sequences, padding='post', maxlen=10)

# 레이블 배열
y = np.array(labels)

# LSTM 모델 정의
model = Sequential()
model.add(Embedding(input_dim=10000, output_dim=128, input_length=10))
model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))  # 이진 분류 (긍정/부정)

# 모델 컴파일
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 모델 훈련
model.fit(X, y, epochs=5, batch_size=2)

# 추론 예시
test_text = ["I love learning new languages"]

# 테스트 문장 전처리
test_sequence = tokenizer.texts_to_sequences(test_text)
test_X = pad_sequences(test_sequence, padding='post', maxlen=10)

# 예측
prediction = model.predict(test_X)
print("예측 결과:", prediction)  # 예측 결과 (0 또는 1)
