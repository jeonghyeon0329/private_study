import numpy as np

# 예시 관절 데이터 (타임스탬프와 관절 좌표)
# pose_data는 [(timestamp, keypoints), ...] 형태로 저장되어 있다고 가정
pose_data = [
    (0.0, np.array([x1, y1, x2, y2, ...])),  # 프레임 1
    (0.1, np.array([x1, y1, x2, y2, ...])),  # 프레임 2
    # ... (더 많은 프레임)
    (10.0, np.array([x1, y1, x2, y2, ...]))  # 프레임 n
]

# 운동 구간을 설정 (t초부터 t+10초까지)
t = 0.0  # 시작 시간
duration = 10.0  # 구간 길이 (초)

# 구간별 데이터 필터링
def filter_motion_data(pose_data, start_time, duration):
    filtered_data = []
    end_time = start_time + duration
    
    for timestamp, keypoints in pose_data:
        if start_time <= timestamp < end_time:
            filtered_data.append((timestamp, keypoints))
    
    return filtered_data

# t초부터 t+10초까지의 운동 데이터 필터링
motion_set = filter_motion_data(pose_data, t, duration)

# 운동 데이터 출력
for timestamp, keypoints in motion_set:
    print(f"Timestamp: {timestamp}, Keypoints: {keypoints}")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# 예시 데이터셋 클래스
class MotionDataset(Dataset):
    def __init__(self, motion_data):
        self.motion_data = motion_data
    
    def __len__(self):
        return len(self.motion_data)
    
    def __getitem__(self, idx):
        timestamp, keypoints = self.motion_data[idx]
        # 여기서는 keypoints를 텐서로 변환하고 레이블을 준비해야 합니다.
        return torch.tensor(keypoints, dtype=torch.float32), 1  # 레이블은 예시로 1로 설정

# 모델 정의
class SimpleNN(nn.Module):
    def __init__(self, input_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)  # 회귀 문제의 경우 1개의 출력

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 모델 학습 함수
def train_model(model, dataset, num_epochs=100):
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    criterion = nn.MSELoss()  # 회귀 문제의 경우 MSE
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.view(-1, 1))  # 레이블 차원 맞추기
            loss.backward()
            optimizer.step()
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# 사용 예시
input_size = len(motion_set[0][1])  # keypoints의 길이
dataset = MotionDataset(motion_set)
model = SimpleNN(input_size)

# 모델 학습
train_model(model, dataset)