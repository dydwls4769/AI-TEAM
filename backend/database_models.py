from sqlalchemy import Column, Integer, String, DateTime, Float, ForeignKey, Text
from sqlalchemy.orm import relationship
from datetime import datetime
from database import Base

# 1. 사용자 테이블 (기존 유지)
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    user_uid = Column(String, unique=True, index=True) 
    nickname = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.now)

    posts = relationship("Post", back_populates="author")

# 2. 생물 종 백과사전 (기존 유지)
class Species(Base):
    __tablename__ = "species"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True) 
    category = Column(String)                      
    habitat = Column(String)                       
    description = Column(Text)                     
    
    posts = relationship("Post", back_populates="species_info")

# 3. 게시글 (수정: 좋아요 필드 추가 및 댓글 관계 연결)
class Post(Base):
    __tablename__ = "posts"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    species_id = Column(Integer, ForeignKey("species.id"))
    
    image_url = Column(String)      
    latitude = Column(Float)        
    longitude = Column(Float)       
    
    # 🌟 추가된 필드: 좋아요 수를 저장합니다.
    likes_count = Column(Integer, default=0) 
    
    created_at = Column(DateTime, default=datetime.now)

    # 관계 설정 (기존 유지 + 댓글 추가)
    author = relationship("User", back_populates="posts")
    species_info = relationship("Species", back_populates="posts")
    image_logs = relationship("ImageLog", back_populates="parent_post")
    
    # 🌟 추가된 관계: 이 포스트에 달린 댓글들을 가져올 수 있게 합니다.
    comments = relationship("Comment", back_populates="parent_post", cascade="all, delete-orphan")

# 4. 이미지 변형 이력 (기존 유지)
class ImageLog(Base):
    __tablename__ = "image_logs"
    id = Column(Integer, primary_key=True, index=True)
    post_id = Column(Integer, ForeignKey("posts.id"))
    
    original_url = Column(String)   
    processed_url = Column(String)  
    filter_type = Column(String)    
    
    created_at = Column(DateTime, default=datetime.now)

    parent_post = relationship("Post", back_populates="image_logs")

# 🌟 5. 댓글 테이블 (새로 추가)
class Comment(Base):
    __tablename__ = "comments"
    id = Column(Integer, primary_key=True, index=True)
    post_id = Column(Integer, ForeignKey("posts.id")) # 어느 게시글의 댓글인지
    author = Column(String, default="익명 탐사대원")      # 댓글 작성자 닉네임
    content = Column(Text, nullable=False)            # 댓글 내용
    created_at = Column(DateTime, default=datetime.now)

    # 게시글과의 관계 설정
    parent_post = relationship("Post", back_populates="comments")