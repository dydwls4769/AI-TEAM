# AI Exercise Pose Feedback — FastAPI Backend

YOLOv5 + MediaPipe + Random Forest 파이프라인을 FastAPI 위에 올려둔 영상 자세 분석 서버입니다.
원본 Streamlit 코드의 핵심 로직(`analyze_video`, `compute_score_from_events`)을 그대로 포팅했습니다.

> 원본 Streamlit 앱(`Streamlit_Upload.py`)은 결과 비교/시각 확인용으로 `../legacy/` 에 보관되어 있습니다.
> 라인별 매핑은 루트의 `MIGRATION_LOG.md`, 전체 설계 가이드는 `../docs/MIGRATION_FastAPI_React.md` 참조.

---

## 빠른 시작

### 1. Python 환경
- Python 3.10 권장 (3.9~3.11 동작 확인)
- Windows / Mac / Linux 모두 지원

### 2. 가상환경 + 의존성 설치
```bash
cd backend
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. 모델 파일 배치 (필수)
원본 저장소(<https://github.com/PSLeon24/AI_Exercise_Pose_Feedback>) 또는 팀에서 공유한 zip 에서 가중치를 받아 다음 경로에 둡니다:

```
backend/
└── models/
    ├── best_big_bounding.pt          ← YOLOv5 사람 검출 가중치
    ├── benchpress/
    │   └── benchpress.pkl
    ├── squat/
    │   └── squat.pkl
    └── deadlift/
        └── deadlift.pkl
```

> ⚠️ 모델 파일은 100MB 가까이 되어 git에 올리지 않습니다 (`.gitignore` 처리됨).

### 4. 서버 실행
```bash
uvicorn app.main:app --reload --port 8000
```

처음 한 번은 YOLOv5 캐시 다운로드 + 모델 로딩으로 30~60초 걸립니다. 콘솔에 `Warmup complete` 로그가 뜨면 준비 완료.

### 5. 동작 확인
- Swagger UI: <http://localhost:8000/docs>
- 헬스체크: <http://localhost:8000/api/health>

---

## API 명세

### `GET /api/health`
서버 / 모델 로딩 상태.

```json
{ "status": "ok", "yolo_loaded": true, "classifiers_loaded": ["benchpress","squat","deadlift"], "device": "cpu", "detail": null }
```

### `GET /api/exercises`
지원 운동 종목 + 권장 촬영 가이드.

```json
[
  { "key": "benchpress", "label": "벤치프레스", "camera_guide": "정면에서 촬영하세요..." },
  { "key": "squat",      "label": "스쿼트",     "camera_guide": "측면에서 촬영하세요..." },
  { "key": "deadlift",   "label": "데드리프트", "camera_guide": "측면에서 촬영하세요..." }
]
```

### `POST /api/analyze`
영상 업로드 → 자세 분석 → 점수 + 피드백 반환.

**Request (multipart/form-data)**
| 필드 | 타입 | 기본값 | 설명 |
|---|---|---|---|
| `video` | file | — | mp4 / mov / avi (최대 200MB) |
| `exercise` | string | — | `benchpress` \| `squat` \| `deadlift` |
| `frame_skip` | int | 3 | n프레임마다 1번 분석 (1~10, 클수록 빠름) |
| `yolo_conf` | float | 0.5 | YOLO 사람 검출 신뢰도 (0.1~0.9) |
| `penalty_per_type` | float | 25 | 오류 유형당 감점 (5~50) |
| `min_duration_sec` | float | 1.0 | 유의미 오류 최소 지속 시간 초 (0.2~3.0) |

**Response 200**
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
  "class_distribution": { "down_correct": 120, "up_correct": 95, "down_caved_in_knees": 30 },
  "settings": { "frame_skip": 3, "yolo_conf": 0.5, "penalty_per_type": 25, "min_duration_sec": 1.0 }
}
```

**Response 에러**
| 코드 | 의미 |
|---|---|
| 400 | 잘못된 exercise / 확장자 |
| 413 | 200MB 초과 |
| 500 | 분석 중 예외 |
| 503 | 모델 파일 없음 (models/ 배치 확인) |

---

## 프론트엔드(React) 연동 가이드

CORS는 `http://localhost:3000`, `http://localhost:5173` 이 기본 허용입니다. 다른 포트/도메인을 쓴다면 `app/config.py` 의 `CORS_ORIGINS` 에 추가.

```js
// React 예시 (axios)
const form = new FormData();
form.append("video", file);
form.append("exercise", "squat");
form.append("frame_skip", 3);

const { data } = await axios.post("http://localhost:8000/api/analyze", form);
console.log(data.score, data.rep_count, data.feedbacks);
```

분석은 영상 길이/스펙에 따라 30초~수 분 걸릴 수 있습니다. axios 요청에 충분한 타임아웃(예: `timeout: 600000`)을 두세요.

---

## 점수 산식

```
score = max(0, 100 − (유의미 오류 유형 수 × penalty_per_type))
```

- "유의미"의 기준: 같은 오류 유형의 **연속 지속 시간이 `min_duration_sec` 이상**이어야 1회로 카운트
- 짧게 튕긴 오작동(예: 0.3초만 잠깐 잘못 잡힘)은 자동 필터링됨

## rep 카운트 규칙
원본 알고리즘 그대로:
- 분류 결과에 `down` 라벨이 들어오면 `current_stage = "down"`
- 그 후 `up` 라벨이 들어오면 `current_stage = "up"` + `rep_count += 1`

---

## 트러블슈팅

| 증상 | 원인 / 해결 |
|---|---|
| `FileNotFoundError: ...best_big_bounding.pt` | `backend/models/` 에 모델 파일 미배치. 위 "3. 모델 파일 배치" 참조 |
| YOLO 로딩 시 `weights_only` 관련 에러 | 코드에서 `torch.load` 패치 적용함. `requirements.txt` 의 torch 버전 확인 |
| MediaPipe `.binarypb` 경로 오류 (Windows) | 경로에 한글이 있을 때 발생. 코드에서 자동 우회 (ASCII junction 생성) |
| sklearn 버전 경고 | `scikit-learn==1.3.0` 으로 핀. 다른 버전 쓰면 pickle 호환 경고 가능 |
| CUDA 미사용 | `torch.cuda.is_available()` 결과에 따라 자동 분기. GPU 쓰려면 PyTorch CUDA 빌드 설치 필요 |

---

## 디렉토리 구조

```
backend/
├── app/
│   ├── __init__.py
│   ├── main.py          ← FastAPI 엔트리, 라우터, lifespan(warmup)
│   ├── config.py        ← 모델 경로, 운동 라벨, CORS, 피드백 메시지
│   ├── inference.py     ← YOLO/MediaPipe/sklearn 로딩, analyze_video, 점수 계산
│   └── schemas.py       ← Pydantic 응답 모델
├── models/              ← (직접 배치, gitignore 처리)
├── requirements.txt
├── .gitignore
└── README.md            ← 이 파일
```

---

## 다음 작업 제안 (선택)

- 분석을 **백그라운드 작업** 으로 전환 (`/api/analyze` → 즉시 job_id 반환, `/api/jobs/{id}` 폴링)
- Redis + RQ 또는 Celery 도입
- 결과 캐싱 (같은 영상 재요청 방지)
- Docker 이미지화 (`Dockerfile` 추가)
- 인증/세션 추가
