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

    new_log = database_models.BiologyLog(
        species_name=info["name"],
        image_path=file_path,
        # habitat=info["habitat"],
        # description=info["description"],
        latitude=lat,
        longitude=lng,
        user_uid="anonymous_tester"  # 👈 일단 모두가 같은 아이디로 저장
        # user_uid="guest_user",
        # description과 habitat은 BiologyInfo 테이블로 분리했으므로 
        # Log 테이블 생성 시에는 넣지 않습니다.
    )
    db.add(new_log)
    db.commit()
    db.refresh(new_log)

    return {
        "id": new_log.id,
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
    # 1. DB에서 모든 기록을 쿼리(질의)합니다.
    logs = db.query(database_models.BiologyLog).all()
    
    # 2. 만약 결과가 없다면 빈 리스트 [] 를 반환하고, 있으면 리스트를 반환합니다.
    return logs

@router.get("/logs/{log_id}")
def get_log_detail(log_id: int, db: Session = Depends(get_db)):
    """
    특정 기록 하나만 자세히 보고 싶을 때 사용합니다.
    """
    log = db.query(database_models.BiologyLog).filter(database_models.BiologyLog.id == log_id).first()
    return log