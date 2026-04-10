from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from database import engine
import database_models
from routers import map, biology
import uvicorn

app = FastAPI()

# 1. CORS 미들웨어를 최상단에 배치
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. DB 테이블 생성 (app 선언 이후에 실행)
database_models.Base.metadata.create_all(bind=engine)

# 3. 라우터 등록
app.include_router(map.router, prefix="/api", tags=["지도"])
app.include_router(biology.router, prefix="/biology", tags=["생물인식"])

@app.get("/")
def root():
    return {"message": "통합 API 서버 정상 작동 중!"}

if __name__ == "__main__":
    # 여기 0.0.0.0으로 고정해두면 python main.py만 쳐도 항상 외부 접속 허용!
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)