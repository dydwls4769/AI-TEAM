from sqlalchemy import Column, Integer, String, DateTime, Float, ForeignKey, Text
from sqlalchemy.orm import relationship
from datetime import datetime
from database import Base

# 1. 사용자 테이블 (추후 로그인 기능 대비)
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    user_uid = Column(String, unique=True, index=True) # 기기 고유값 또는 ID
    nickname = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.now)

    posts = relationship("Post", back_populates="author")

# 2. 생물 종 백과사전 (도감 데이터)
class Species(Base):
    __tablename__ = "species"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True) # 생물 이름 (중복 방지)
    category = Column(String)                      # 식물, 동물, 곤충 등
    habitat = Column(String)                       # 서식지 정보
    description = Column(Text)                     # 상세 설명
    
    posts = relationship("Post", back_populates="species_info")

# 3. 게시글 (촬영 기록 및 위치 정보)
class Post(Base):
    __tablename__ = "posts"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    species_id = Column(Integer, ForeignKey("species.id"))
    
    image_url = Column(String)      # 원본 이미지 경로
    latitude = Column(Float)        # 위도
    longitude = Column(Float)       # 경도
    created_at = Column(DateTime, default=datetime.now)

    # 관계 설정
    author = relationship("User", back_populates="posts")
    species_info = relationship("Species", back_populates="posts")
    image_logs = relationship("ImageLog", back_populates="parent_post")

# 4. 이미지 변형 이력 (AI 처리 결과물)
class ImageLog(Base):
    __tablename__ = "image_logs"
    id = Column(Integer, primary_key=True, index=True)
    post_id = Column(Integer, ForeignKey("posts.id"))
    
    original_url = Column(String)   # 변형 전 이미지
    processed_url = Column(String)  # 변형 후(화질개선/색상변경 등) 이미지
    filter_type = Column(String)    # 적용된 모델/필터 이름 (예: 'SR-GAN', 'Grayscale')
    
    created_at = Column(DateTime, default=datetime.now)

    parent_post = relationship("Post", back_populates="image_logs")