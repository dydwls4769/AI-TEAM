import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# 1. 모델 구조 재설정 (학습 때와 동일하게)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = models.resnet18()
model.fc = nn.Linear(model.fc.in_features, 5) # 5개 클래스
model.load_state_dict(torch.load('biology_model.pt')) # 저장된 뇌 불러오기
model = model.to(device)
model.eval() # 평가 모드 전환

# 2. 이미지 전처리 (학습 때와 똑같은 필터 적용)
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 3. 테스트할 이미지 불러오기 (사진 파일명을 넣으세요)
img_path = 'test_image.jpg' 
img = Image.open(img_path).convert('RGB')
img_t = preprocess(img).unsqueeze(0).to(device)

# 4. 예측 시작!
classes = ["dandelion", "foxtail", "heron", "magpie", "mallard"] # 폴더 순서와 동일해야 함
with torch.no_grad():
    output = model(img_t)
    _, predicted = torch.max(output, 1)
    print(f"결과: 이 생물은 '{classes[predicted[0]]}' 일 확률이 높습니다!")