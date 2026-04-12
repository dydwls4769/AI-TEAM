from fastapi import APIRouter, File, UploadFile, Depends, Form
from sqlalchemy.orm import Session
import torch
import torch.nn as nn
from torchvision import models, transforms
import os, uuid
from PIL import Image, ImageOps
from io import BytesIO
from database import get_db
import database_models
from biology_info import biology_data, class_names
from ultralytics import YOLO

router = APIRouter()

# 이미지를 저장할 경로 설정
UPLOAD_DIR = "uploads"
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

# --- 1. 모델 로드 (가속 장치 확인) ---
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# 생물 분류용 모델 (ResNet18)
classifier_model = models.resnet18()
classifier_model.fc = nn.Linear(classifier_model.fc.in_features, 5) 
classifier_model.load_state_dict(torch.load("biology_model.pt", map_location=device))
classifier_model = classifier_model.to(device).eval()

# 사물 검출용 모델 (YOLO) - [기존 유지]
yolo_model = YOLO('yolov8n.pt') 

# 전처리 설정
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- 2. 도움 함수 (친구분의 Smart Predict 로직) ---

def predict_crop(img):
    """단일 이미지 조각 예측"""
    input_tensor = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = classifier_model(input_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        confidence, pred = torch.max(probs, 1)
    return pred.item(), confidence.item()

def smart_predict(img):
    """멀티 크롭 + 가중 투표 (인식률 향상의 핵심)"""
    w, h = img.size
    results = []
    
    # 1. 세로 스트립 (서 있는 새 등에 효과적)
    strips = [(0.40, 8.0), (0.35, 7.5), (0.45, 7.5), (0.50, 7.0)]
    for w_ratio, weight in strips:
        crop_w = int(w * w_ratio)
        left = (w - crop_w) // 2
        cropped = img.crop((left, 0, left + crop_w, h))
        pred, conf = predict_crop(cropped)
        results.append((pred, conf * weight))
    
    # 2. 중앙 크롭 (보조)
    for ratio, weight in [(0.50, 4.0), (0.40, 4.5)]:
        size = int(min(w, h) * ratio)
        left, top = (w - size) // 2, (h - size) // 2
        cropped = img.crop((left, top, left + size, top + size))
        pred, conf = predict_crop(cropped)
        results.append((pred, conf * weight))
    
    # 3. 전체 이미지 (낮은 가중치)
    pred, conf = predict_crop(img)
    results.append((pred, conf * 0.1))
    
    # 클래스별 점수 합산
    scores = {}
    for pred, weighted_score in results:
        scores[pred] = scores.get(pred, 0) + weighted_score
    
    best_idx = max(scores, key=scores.get)
    return class_names[best_idx]

# --- 3. API 엔드포인트 ---

@router.post("/predict")
async def predict(
    file: UploadFile = File(...), 
    lat: float = Form(...), 
    lng: float = Form(...), 
    db: Session = Depends(get_db)):
    
    # 이미지 읽기
    image_bytes = await file.read()
    file_extension = file.filename.split(".")[-1]
    unique_filename = f"{uuid.uuid4()}.{file_extension}"
    file_path = os.path.join(UPLOAD_DIR, unique_filename)

    img = Image.open(BytesIO(image_bytes)).convert("RGB")

    # EXIF 정보를 바탕으로 사진을 올바른 방향으로 자동 회전 시킵니다.
    img = ImageOps.exif_transpose(img)
    
    # 원본 저장용 (thumbnail)
    save_img = img.copy()
    save_img.thumbnail((800, 800))
    save_img.save(file_path, optimize=True, quality=85)

    # AI 분석 (친구분의 smart_predict 적용!)
    label = smart_predict(img)

    # 정보 가져오기
    info = biology_data.get(label, {"name": "알 수 없음", "habitat": "정보 없음", "description": "분석 실패"})
    kor_name = info["name"]

    # DB 저장 (우리가 맞춘 Post 모델 구조 유지)
    species_record = db.query(database_models.Species).filter(database_models.Species.name == kor_name).first()

    if not species_record:
        return {"error": "도감 정보가 존재하지 않는 생물입니다."}

    new_post = database_models.Post(
        user_id=1, 
        species_id=species_record.id,
        image_url=f"/uploads/{unique_filename}",
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

# 기존 /logs 및 /detect 엔드포인트는 그대로 유지하시면 됩니다.
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
            "box": box.xyxy[0].tolist()
        })
    return {"objects": detections}

@router.get("/logs")
def get_logs(db: Session = Depends(get_db)):
    return db.query(database_models.Post).all()