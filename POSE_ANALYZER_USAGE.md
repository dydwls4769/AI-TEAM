# pose_analyzer.py — 단일 파일 사용 가이드

본인 FastAPI 프로젝트에 **이 파일 1개만** 끼워 넣고 import해서 쓰시면 됩니다.

---

## 1. 파일 배치

본인 프로젝트의 적절한 위치에 두세요. 예시:
```
my_fastapi_project/
├── app/
│   ├── routers/
│   │   └── pose.py             ← 본인이 작성하는 라우터
│   ├── services/
│   │   └── pose_analyzer.py    ← 여기에 두기
│   └── ...
└── models/                      ← 모델 가중치 파일들
    ├── best_big_bounding.pt
    ├── benchpress/benchpress.pkl
    ├── squat/squat.pkl
    └── deadlift/deadlift.pkl
```

## 2. 모델 가중치 배치

**기본 경로:** `pose_analyzer.py` 와 같은 디렉토리의 `models/`

**다른 경로 쓰고 싶으면:** 환경변수로 덮어쓰기
```bash
export POSE_MODELS_DIR=/path/to/your/models
```
또는 `pose_analyzer.py` 상단의 `MODELS_DIR` 상수를 직접 수정.

가중치 파일은 원본 저장소에서 받으세요:
- https://github.com/PSLeon24/AI_Exercise_Pose_Feedback (`models/` 폴더)

## 3. 의존성

```
pip install torch torchvision opencv-python mediapipe \
            ultralytics scikit-learn==1.3.0 numpy pandas pillow
```

> ⚠️ `scikit-learn==1.3.0` 으로 핀하세요. 학습 시 사용한 버전과 달라지면 pickle 로딩 시 경고/오류 가능.

## 4. 사용법 — 가장 단순한 형태

```python
from pose_analyzer import analyze_video, build_response_payload

raw = analyze_video("my_squat.mp4", exercise="squat")
result = build_response_payload(raw, exercise="squat")

print(result["score"], result["rep_count"], result["feedbacks"])
```

## 5. FastAPI 라우터 통합 예시

```python
# app/routers/pose.py
from fastapi import APIRouter, File, Form, UploadFile, HTTPException
from app.services.pose_analyzer import analyze_video, build_response_payload, EXERCISE_MODEL_PATHS
import os, tempfile

router = APIRouter()

@router.post("/analyze")
async def analyze(
    video: UploadFile = File(...),
    exercise: str = Form(...),
):
    if exercise not in EXERCISE_MODEL_PATHS:
        raise HTTPException(400, f"unknown exercise: {exercise}")

    suffix = os.path.splitext(video.filename or "")[1] or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await video.read())
        tmp_path = tmp.name

    try:
        raw = analyze_video(tmp_path, exercise=exercise)
        return build_response_payload(raw, exercise=exercise)
    except FileNotFoundError as e:
        raise HTTPException(503, str(e))
    finally:
        os.unlink(tmp_path)
```

본인 라우터/스키마/응답 스타일은 자유롭게 짜시면 됩니다. 위는 참고용 예시입니다.

## 6. 노출되는 함수 / 상수

| 이름 | 용도 |
|---|---|
| `analyze_video(video_path, exercise, ...)` | 영상 분석 메인 함수 |
| `compute_score_from_events(event_groups, ...)` | 점수 계산 |
| `build_response_payload(raw, exercise, ...)` | 결과 → 응답 dict 변환 (편의 함수) |
| `warmup()` | 서버 시작 시 호출하면 첫 요청 빨라짐 |
| `load_yolo_model()`, `load_classifier(ex)` | 수동 로딩 (필요 시) |
| `EXERCISE_MODEL_PATHS` | 지원 종목 키 목록 (`benchpress`, `squat`, `deadlift`) |
| `EXERCISE_LABELS` | 종목 한글 라벨 |
| `CAMERA_GUIDE` | 종목별 촬영 각도 가이드 |
| `FEEDBACK_MESSAGES` | 오류 코드 → 사용자 메시지 |
| `ERROR_KEYS` | 오류 코드 목록 |

## 7. 응답 dict 스키마

`build_response_payload` 가 반환하는 dict:
```json
{
  "exercise": "squat",
  "exercise_label": "스쿼트",
  "score": 75.0,
  "rep_count": 8,
  "analyzed_frames": 420,
  "total_frames": 1200,
  "fps": 30.0,
  "feedbacks": [
    {
      "code": "caved_in_knees",
      "message": "무릎이 움푹 들어간 자세입니다...",
      "occurrences": 3,
      "total_duration_sec": 4.2,
      "max_duration_sec": 2.1,
      "counted": true
    }
  ],
  "class_distribution": {"down_correct": 120, "up_correct": 95},
  "settings": {"penalty_per_type": 25, "min_duration_sec": 1.0}
}
```

응답 형식이 마음에 안 들면 `build_response_payload` 안 쓰시고, `analyze_video` 결과 dict를 직접 가공하시면 됩니다.

## 8. CLI 단독 테스트

웹 통합 전에 단독으로 동작 확인 가능:
```bash
python pose_analyzer.py 스쿼트2.mp4 squat
```
JSON 결과가 stdout 으로 출력됩니다.

## 9. 알아두면 좋은 점

- **모델 로딩은 1번만**: `_yolo_model`, `_classifiers` 가 모듈 전역에 캐시됩니다. FastAPI `lifespan` 에서 `warmup()` 호출하면 첫 요청이 느리지 않습니다.
- **처음 실행 시 30~60초** 걸립니다 — PyTorch hub 캐시 다운로드 + YOLOv5 로딩 시간.
- **분석 시간**: CPU 기준 1분 영상이 30초~2분 걸립니다. GPU 있으면 자동으로 CUDA 사용.
- **한글 경로 (Windows)**: site-packages 경로에 한글이 있으면 mediapipe 로딩 실패. 모듈 import 시 자동 우회 코드 동작.

## 10. 점수 / rep 카운트 로직 (참고)

원본 Streamlit 코드와 100% 동일한 알고리즘입니다:

**rep 카운트:**
```python
if stage == "down":
    current_stage = "down"
elif stage == "up" and current_stage == "down":
    current_stage = "up"
    rep_count += 1
```

**점수 산식:**
```
score = max(0, 100 − (유의미_오류_유형_수 × penalty_per_type))
유의미: 같은 오류가 연속으로 min_duration_sec(기본 1초) 이상 지속된 경우
```
