from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqladmin import Admin, ModelView
from database import engine
# Base를 추가로 임포트합니다.
from database_models import User, Species, Post, ImageLog, Base 
# ranking 라우터가 누락되어 있다면 추가하세요.
from routers import map, biology, blog, ranking 
import uvicorn
from fastapi.staticfiles import StaticFiles
import os

app = FastAPI()

# 관리자 페이지 설정
admin = Admin(app, engine)

# 관리자 페이지에서 보고 싶은 테이블들을 등록
class UserAdmin(ModelView, model=User):
    column_list = [User.id, User.user_uid, User.nickname]

class SpeciesAdmin(ModelView, model=Species):
    column_list = [Species.id, Species.name, Species.category, Species.habitat]

class PostAdmin(ModelView, model=Post):
    column_list = [Post.id, Post.species_id, Post.created_at]

# 관리자 패널에 추가
admin.add_view(UserAdmin)
admin.add_view(SpeciesAdmin)
admin.add_view(PostAdmin)

# 1. CORS 미들웨어 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

# 2. DB 테이블 생성 (database_models. 삭제 후 Base만 사용)
Base.metadata.create_all(bind=engine)

# 3. 라우터 등록
app.include_router(blog.router, prefix="/blog", tags=["블로그"])
app.include_router(map.router, prefix="/api", tags=["지도"])
app.include_router(biology.router, prefix="/biology", tags=["생물인식"])
app.include_router(ranking.router, prefix="/ranking", tags=["랭킹"])

@app.get("/")
def root():
    return {"message": "통합 API 서버 정상 작동 중!"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)