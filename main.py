import torch
import torch.nn as nn
from torchvision import models, transforms
from fastapi.middleware.cors import CORSMiddleware
import os
import uuid
from PIL import Image
from io import BytesIO
from fastapi import FastAPI, File, UploadFile, Depends
from sqlalchemy.orm import Session
from dotenv import load_dotenv
import database_models
from database import engine, get_db
from biology_info import biology_data, class_names
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

load_dotenv()
database_models.Base.metadata.create_all(bind=engine)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

# 모델 로드
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
model = models.resnet18()
model.fc = nn.Linear(model.fc.in_features, 5)
model.load_state_dict(torch.load("biology_model.pt", map_location=device))
model = model.to(device)
model.eval()

# 전처리
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def predict_crop(img):
    """단일 크롭 이미지 예측"""
    input_tensor = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        confidence, pred = torch.max(probs, 1)
    return pred.item(), confidence.item()


def smart_predict(img):
    """
    멀티 크롭 + 가중 투표 예측
    - 세로 스트립: 서 있는 피사체(새) 인식에 효과적
    - 전체 이미지: 배경 영향이 크므로 낮은 가중치
    """
    w, h = img.size
    results = []
    
    # 세로 스트립 전략 (핵심)
    strips = [
        (0.40, 8.0),  # 가로 40% - 최고 가중치
        (0.35, 7.5),
        (0.45, 7.5),
        (0.50, 7.0),
    ]
    
    for w_ratio, weight in strips:
        crop_w = int(w * w_ratio)
        left = (w - crop_w) // 2
        cropped = img.crop((left, 0, left + crop_w, h))
        pred, conf = predict_crop(cropped)
        results.append((pred, conf * weight))
    
    # 중앙 크롭 (보조)
    for ratio, weight in [(0.50, 4.0), (0.40, 4.5)]:
        size = int(min(w, h) * ratio)
        left, top = (w - size) // 2, (h - size) // 2
        cropped = img.crop((left, top, left + size, top + size))
        pred, conf = predict_crop(cropped)
        results.append((pred, conf * weight))
    
    # 전체 이미지 (낮은 가중치)
    pred, conf = predict_crop(img)
    results.append((pred, conf * 0.1))
    
    # 클래스별 점수 합산
    scores = {}
    for pred, weighted_score in results:
        scores[pred] = scores.get(pred, 0) + weighted_score
    
    # 최고 점수 클래스 반환
    best = max(scores, key=scores.get)
    best_conf = max([r[1] for r in results if r[0] == best])
    
    return class_names[best], best_conf


@app.post("/predict")
async def predict(file: UploadFile = File(...), db: Session = Depends(get_db)):
    # 이미지 읽기 및 저장
    image_bytes = await file.read()
    file_extension = file.filename.split(".")[-1]
    unique_filename = f"{uuid.uuid4()}.{file_extension}"
    file_path = os.path.join("uploads", unique_filename)

    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    save_img = img.copy()
    save_img.thumbnail((800, 800))
    save_img.save(file_path, optimize=True, quality=85)

    # AI 분석 (멀티 크롭 + 가중 투표)
    label, confidence = smart_predict(img)

    # 정보 가져오기
    info = biology_data.get(label, {
        "name": "알 수 없음",
        "habitat": "정보 없음",
        "description": "분석 실패"
    })

    # DB 저장
    new_log = database_models.BiologyLog(
        species=info["name"],
        image_path=file_path,
        habitat=info["habitat"],
        description=info["description"]
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


@app.get("/logs")
def get_logs(db: Session = Depends(get_db)):
    return db.query(database_models.BiologyLog).all()


if os.path.exists("dist"):
    app.mount("/assets", StaticFiles(directory="dist/assets"), name="static")

    @app.get("/")
    def serve_index():
        return FileResponse("dist/index.html")
else:
    @app.get("/")
    def read_root():
        return {"message": "dist 폴더를 찾을 수 없습니다. 빌드 후 다시 시도하세요!"}


    # ============================================
# 수정사항 
# - 멀티 크롭 + 가중 투표 방식 추가
# - 원거리 촬영 시 오인식 문제 해결
# ============================================