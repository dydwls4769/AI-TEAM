from fastapi import APIRouter, Depends, UploadFile, File, Form
from sqlalchemy.orm import Session
from database import get_db
import database_models
# import database_models as models
import uuid
import os

router = APIRouter()

# 이미지 저장 경로 설정
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.get("/")
def get_all_posts(db: Session = Depends(get_db)):
    # Post 테이블의 모든 데이터를 가져와서 리턴합니다.
    posts = db.query(database_models.Post).all()
    return posts

@router.post("/upload")
async def create_post(
    user_uid: str = Form(...),
    species_id: int = Form(...),
    latitude: float = Form(...),
    longitude: float = Form(...),
    image: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    # 1. 파일 저장 로직 (실제로는 S3 같은 곳에 올리지만 일단 로컬에 저장)
    file_extension = image.filename.split(".")[-1]
    file_name = f"{uuid.uuid4()}.{file_extension}"
    file_path = os.path.join(UPLOAD_DIR, file_name)
    
    with open(file_path, "wb") as f:
        f.write(await image.read())

    # 2. DB 저장
    new_post = models.Post(
        user_id=1, # 일단 테스트용 유저 ID
        species_id=species_id,
        image_url=f"/static/uploads/{file_name}",
        latitude=latitude,
        longitude=longitude
    )
    db.add(new_post)
    db.commit()
    db.refresh(new_post)
    
    return {"message": "업로드 성공", "post_id": new_post.id}