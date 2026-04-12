# seed.py
from database import SessionLocal, engine
import database_models as models # 파일명이 database_models.py라고 하셨으므로
from biology_info import biology_data

def seed_species():
    # DB 세션 생성
    db = SessionLocal()
    
    print("🌱 기초 데이터 이식을 시작합니다...")
    
    try:
        for key, info in biology_data.items():
            # 중복 체크: 이미 같은 이름의 생물이 있는지 확인
            existing = db.query(models.Species).filter(models.Species.name == info["name"]).first()
            
            if not existing:
                new_species = models.Species(
                    name=info["name"],
                    category=info["category"],
                    habitat=info["habitat"],
                    description=info["description"]
                )
                db.add(new_species)
                print(f"✅ 추가됨: {info['name']}")
            else:
                print(f"⚠️ 이미 존재함 (건너뜀): {info['name']}")
        
        db.commit()
        print("✨ 모든 데이터 이식이 완료되었습니다!")
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    # 테이블이 아직 생성되지 않았다면 생성 (안전장치)
    models.Base.metadata.create_all(bind=engine)
    seed_species()