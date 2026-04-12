from fastapi import APIRouter, File, UploadFile, Depends, Form
from sqlalchemy.orm import Session
import torch
import torch.nn as nn
from torchvision import models, transforms
import os, uuid
from PIL import Image
from io import BytesIO
from database import get_db
import database_models
from biology_info import biology_data, class_names
from ultralytics import YOLO

router = APIRouter()

# 이미지를 저장할 경로 설정
UPLOAD_DIR = "uploads"

# 폴더가 없으면 생성
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

# --- 1. 모델 로드 부분 (이름을 명확히 구분하세요) ---
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# 생물 분류용 모델 (ResNet18)
classifier_model = models.resnet18()
classifier_model.fc = nn.Linear(classifier_model.fc.in_features, 5) # 현재 5개 학습됨
classifier_model.load_state_dict(torch.load("biology_model.pt", map_location=device))
classifier_model = classifier_model.to(device).eval()

# 사물 검출용 모델 (YOLO)
yolo_model = YOLO('yolov8n.pt') 

# 전처리 설정
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- 2. API 엔드포인트 ---

# [1단계] 사진을 올리면 사물들의 '위치'만 알려주는 API
@router.post("/detect")
async def detect_objects(file: UploadFile = File(...)):
    image_bytes = await file.read()
    img = Image.open(BytesIO(image_bytes))
    
    results = yolo_model(img)
    
    detections = []
    for box in results[0].boxes:
        detections.append({
            "label": results[0].names[int(box.cls)], 
            "confidence": float(box.conf),
            "box": box.xyxy[0].tolist() # [x1, y1, x2, y2]
        })
    
    return {"objects": detections}

# [2단계] 선택된 이미지를 실제로 분석하고 DB에 저장하는 API
@router.post("/predict")
async def predict(
    file: UploadFile = File(...), 
    lat: float = Form(...), 
    lng: float = Form(...), 
    db: Session = Depends(get_db)):
    
    image_bytes = await file.read()
    file_extension = file.filename.split(".")[-1]
    unique_filename = f"{uuid.uuid4()}.{file_extension}"
    file_path = os.path.join("uploads", unique_filename)

    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    
    # 이미지 저장
    save_img = img.copy()
    save_img.thumbnail((800, 800))
    save_img.save(file_path, optimize=True, quality=85)

    # 진짜 AI 분석 (ResNet18 사용)
    input_tensor = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = classifier_model(input_tensor) # classifier_model로 이름 변경됨!
        _, preds = torch.max(outputs, 1)
        label = class_names[preds[0]]

    # DB 정보 가져오기 및 저장
    info = biology_data.get(label, {"name": "알 수 없음", "habitat": "정보 없음", "description": "분석 실패"})
 
    # 1. AI가 예측한 영문 label로 Species 테이블에서 해당 생물의 ID를 조회합니다.
    # biology_data[label]["name"]은 "왜가리" 같은 한글 이름입니다.
    kor_name = info["name"]
    species_record = db.query(database_models.Species).filter(database_models.Species.name == kor_name).first()

    if not species_record:
        # 혹시 seed 데이터에 없는 생물이 예측되었다면 에러를 방지하기 위해 처리
        return {"error": "도감 정보가 존재하지 않는 생물입니다."}

    # 2. BiologyLog 대신 Post 모델을 생성합니다.
    new_post = database_models.Post(
        user_id=1,  # User 테이블의 기본값 (임시)
        species_id=species_record.id,  # 찾은 생물의 ID(PK)를 넣어줍니다.
        image_url=f"/uploads/{unique_filename}", # DB 필드명에 맞춰 image_url 사용
        latitude=lat,
        longitude=lng
    )
    
    db.add(new_post)
    db.commit()
    db.refresh(new_post)

    return {
        "id": new_post.id,
        "prediction": info["name"],
        "description": info["description"],
        "habitat": info["habitat"],
        "image_url": f"/uploads/{unique_filename}"
    }

@router.get("/logs")
def get_logs(db: Session = Depends(get_db)):
    """
    DB에 저장된 모든 생물 탐지 기록을 가져옵니다.
    나중에 리액트 지도에서 이 데이터를 호출해서 핀을 꽂게 됩니다.
    """
    # BiologyLog -> Post로 변경
    logs = db.query(database_models.Post).all()
    return logs

@router.get("/logs/{log_id}")
def get_log_detail(log_id: int, db: Session = Depends(get_db)):
    """
    특정 기록 하나만 자세히 보고 싶을 때 사용합니다.
    """
    # BiologyLog -> Post로 변경
    log = db.query(database_models.Post).filter(database_models.Post.id == log_id).first()
    return log