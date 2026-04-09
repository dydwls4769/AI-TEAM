from fastapi import APIRouter
import pandas as pd

# 리액트 컴포넌트 만들듯 라우터 생성
router = APIRouter()

# 데이터 로드
df = pd.read_csv("EvCharger_data.csv")

@router.get("/chargers") # prefix를 main에서 줄 거라 /api/chargers -> /chargers로 수정
def get_chargers():
    data = df.to_dict(orient="records") 
    return data