from sqlalchemy import Column, Integer, String, DateTime, Float, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime
from database import Base

# 1. 생물 백과사전 테이블 (중복 방지용)
class BiologyInfo(Base):
    __tablename__ = "biology_info"
    
    # 생물 이름 자체를 기본키(PK)로 써서 중복 등록을 막습니다.
    species = Column(String, primary_key=True) 
    description = Column(String)  # 상세 설명 (한 번만 저장)
    habitat = Column(String)      # 서식지 (한 번만 저장)
    
    # Log 테이블과의 연결 고리
    logs = relationship("BiologyLog", back_populates="info")

# 2. 사용자의 촬영 기록 테이블
class BiologyLog(Base):
    __tablename__ = "biology_logs"

    id = Column(Integer, primary_key=True, index=True)
    
    # 어떤 생물인지 이름만 기록 (BiologyInfo의 species를 참조)
    species_name = Column(String, ForeignKey("biology_info.species"))
    
    # 촬영 정보
    latitude = Column(Float, nullable=True)
    longitude = Column(Float, nullable=True)
    image_path = Column(String)
    
    # 사용자 구분을 위한 고유 ID (로그인 없이 기기별 생성값)
    user_uid = Column(String, index=True) 
    
    created_at = Column(DateTime, default=datetime.now)

    # Info 테이블과의 연결
    info = relationship("BiologyInfo", back_populates="logs")