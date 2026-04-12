from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from database import get_db
import database_models as models

router = APIRouter()

@router.get("/observations")  # 이름을 생물 관찰 데이터에 맞게 변경
def get_observations(db: Session = Depends(get_db)):
    # Post 테이블과 Species 테이블을 조인해서 가져오면 이름과 설명까지 한 번에 보낼 수 있습니다.
    posts = db.query(models.Post).all()
    
    result = []
    for post in posts:
        result.append({
            "id": post.id,
            "lat": post.latitude,
            "lng": post.longitude,
            "species_name": post.species_info.name if post.species_info else "알 수 없음",
            "author": post.author.nickname if post.author else "익명",
            "image_url": post.image_url,
            "created_at": post.created_at.strftime("%Y-%m-%d")
        })
    return result