from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd

app = FastAPI()

# React와 통신을 위한 CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 데이터 로드 (서버 시작 시 한 번만 로드하거나 함수 내에서 호출)
df = pd.read_csv("EvCharger_data.csv")

@app.get("/api/chargers")
def get_chargers():
    # 주피터에서 하셨던 데이터 가공 로직을 여기에 넣습니다.
    # 예: 필요한 컬럼만 추출하거나 chgerType 매핑 등
    data = df.to_dict(orient="records") 
    return data