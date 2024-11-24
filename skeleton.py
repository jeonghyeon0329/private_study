import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# 1. 가상의 운동 데이터 생성 함수 (예: 팔굽혀펴기)
def generate_synthetic_data(num_frames, movement_type='pushup'):
    pose_data = []
    for timestamp in range(num_frames):
        # 간단한 팔굽혀펴기 모션 시뮬레이션
        if movement_type == 'pushup':
            # 팔꿈치 높이가 시간에 따라 변하는 형태
            elbow_y = 100 + 50 * np.sin(np.pi * timestamp / 30)  # 팔꿈치 Y 좌표 변화
            shoulder_y = elbow_y + 50  # 어깨는 팔꿈치보다 높게 설정
            wrist_y = elbow_y - 10  # 손목은 팔꿈치보다 조금 더 낮게 설정
            keypoints = np.array([50, shoulder_y, 80, elbow_y, 110, wrist_y, 50, 150, 80, 200])  # x1, y1, ..., x5, y5
        else:
            keypoints = np.random.rand(10) * 200  # 기본적인 랜덤 데이터
        
        pose_data.append((timestamp * 0.1, keypoints))  # 0.1초 간격으로 타임스탬프 추가

    return pose_data

# 2. 운동 구간 설정 함수
def filter_motion_data(pose_data, start_time, duration):
    filtered_data = []
    end_time = start_time + duration
    for timestamp, keypoints in pose_data:
        if start_time <= timestamp < end_time:
            filtered_data.append((timestamp, keypoints))
    return filtered_data

# # 3. 데이터셋 클래스 정의
# class MotionDataset(Dataset):
#     def __init__(self, motion_data):
#         self.motion_data = motion_data
    
#     def __len__(self):
#         return len(self.motion_data)
    
#     def __getitem__(self, idx):
#         timestamp, keypoints = self.motion_data[idx]
#         return torch.tensor(keypoints, dtype=torch.float32), 1  # 레이블을 1로 설정 (예시)

class MotionDataset(Dataset):
    def __init__(self, motion_data):
        self.motion_data = [(torch.tensor(keypoints, dtype=torch.float32), 1) for _, keypoints in motion_data]
      
    def __len__(self):
        return len(self.motion_data)
      
    def __getitem__(self, idx):
        return self.motion_data[idx]

# 4. 모델 정의
class SimpleNN(nn.Module):
    def __init__(self, input_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)  # 회귀 문제이므로 1개의 출력

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 5. 모델 학습 함수
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

# 6. 가상의 운동 데이터 생성 및 필터링
num_frames = 100  # 100프레임 생성
pose_data = generate_synthetic_data(num_frames, movement_type='pushup')

# 운동 구간 설정 (0초부터 10초까지)
t = 0.0
duration = 10.0

# 구간별 데이터 필터링
motion_set = filter_motion_data(pose_data, t, duration)

# 운동 데이터 출력 (디버깅용)
for timestamp, keypoints in motion_set:
    print(f"Timestamp: {timestamp}, Keypoints: {keypoints}")

# 7. 모델 학습
input_size = len(motion_set[0][1])  # keypoints의 길이 (예시: 10개의 키포인트)
dataset = MotionDataset(motion_set)
model = SimpleNN(input_size)

# 학습 시작
train_model(model, dataset)
