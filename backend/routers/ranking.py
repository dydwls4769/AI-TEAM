# backend/routers/ranking.py
from fastapi import APIRouter

# 이 부분이 반드시 있어야 합니다! 이름이 'router'여야 main.py와 매칭됩니다.
router = APIRouter()

@router.get("/")
def get_ranking():
    return {"message": "랭킹 목록"}