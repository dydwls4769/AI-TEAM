# train.py (가장 심플한 버전)
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import os

# 1. 맥미니 가속(MPS) 설정
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# 2. 이미지 전처리 설정
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
])

# 3. 데이터 로드 (현재 폴더의 dataset 폴더를 읽음)
image_datasets = datasets.ImageFolder('./dataset', data_transforms)
dataloaders = DataLoader(image_datasets, batch_size=16, shuffle=True)

# 4. 모델 설정 (5개 클래스 구분용)
model = models.resnet18(weights='IMAGENET1K_V1')
model.fc = nn.Linear(model.fc.in_features, len(image_datasets.classes))
model = model.to(device)

# 5. 학습 도구
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 6. 딱 5바퀴만 학습
print(f"인식 대상: {image_datasets.classes}")
print("학습을 시작합니다... (맥미니 성능을 믿어보세요!)")

for epoch in range(20):
    model.train()
    for inputs, labels in dataloaders:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f"에포크 {epoch+1}/5 완료")

# 7. 결과물 저장
torch.save(model.state_dict(), 'biology_model.pt')
print("축하합니다! 'biology_model.pt' 파일이 생성되었습니다.")