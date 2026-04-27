# Streamlit → FastAPI 마이그레이션 로그

작성: 2026-04-27
대상 원본: `legacy/Streamlit_Upload.py` (영상 업로드 + 점수 분석 모드)
결과물: `backend/` (FastAPI 서버)

> 원본은 GitHub 저장소의 `legacy/` 폴더에 보관되어 있습니다. 본 문서의 라인 번호는 그 파일을 기준으로 합니다.

---

## 1. 백업 현황 (롤백 자료)

| 백업 | 위치 |
|---|---|
| 로컬 폴더 사본 | `바탕 화면/AI_Exercise_Pose_Feedback-BACKUP_2026-04-27/` (venv·캐시 제외) |
| GitHub 백업 브랜치 | `https://github.com/dydwls4769/AI-TEAM/tree/before-fastapi-migration` |
| 원본 저장소 | `https://github.com/PSLeon24/AI_Exercise_Pose_Feedback` (변경 없음) |

원본 폴더(`AI_Exercise_Pose_Feedback-main/`) 자체는 한 글자도 수정하지 않았습니다. 본 작업물은 **새 폴더(`AI_Exercise_FullStack/`)** 에서만 진행됨.

---

## 2. 코드 매핑 — Streamlit_Upload.py → backend/

| 원본 위치 | 원본 함수/상수 | 이동 위치 | 비고 |
|---|---|---|---|
| 8~38행 | `_setup_ascii_mediapipe_path()` | `app/main.py` 상단 | 한글 경로 우회. 그대로 |
| 48~53행 | `YOLO_WEIGHTS_PATH`, `EXERCISE_MODEL_PATHS` | `app/config.py` | `Path` 객체로 변환, 키를 한글→영문(`벤치프레스`→`benchpress`)으로 정규화 |
| 55~59행 | `CAMERA_GUIDE` | `app/config.py` | 키 영문화 |
| 61~70행 | `FEEDBACK_MESSAGES`, `ERROR_KEYS` | `app/config.py` | 그대로 |
| 73~96행 | `load_yolo_model()` (Streamlit cache) | `app/inference.py::load_yolo_model()` | `@st.cache_resource` 제거, 모듈 전역 변수로 1회 로딩 |
| 99~103행 | `load_exercise_model()` | `app/inference.py::load_classifier()` | 함수명 변경, dict 캐시 |
| 106~111행 | `extract_landmark_row` | `app/inference.py::_extract_landmark_row` | 그대로 |
| 114~127행 | `classify_posture` | `app/inference.py::_classify_posture` | `is_correct` 반환은 미사용이라 제거 |
| 130~140행 | `compute_score_from_events` | `app/inference.py::compute_score_from_events` | 그대로 |
| 143~285행 | `analyze_video()` | `app/inference.py::analyze_video()` | 그대로 (Streamlit 의존 없음) |
| 288~358행 | `render_results()` (Streamlit UI) | **삭제** → JSON 응답으로 대체 | `app/main.py` 의 `/api/analyze` 응답 빌드에서 처리 |
| 361~467행 | `main()` (Streamlit UI) | **삭제** | FastAPI 라우터로 대체 |

---

## 3. 신규 추가물

| 파일 | 역할 |
|---|---|
| `app/main.py` | FastAPI 엔트리. `/api/health`, `/api/exercises`, `/api/analyze` 라우터. `lifespan` 에서 모델 워밍업 |
| `app/schemas.py` | Pydantic 응답 모델 (AnalyzeResponse, FeedbackItem, ExerciseInfo, HealthResponse) |
| `app/config.py` | 경로/상수/CORS 설정 모음 |
| `requirements.txt` | 백엔드 전용 최소 의존성 |
| `.gitignore` | venv, models/ 등 제외 |
| `README.md` | 실행 방법, API 명세, 트러블슈팅 |

---

## 4. 의도적 변경 사항

1. **운동 종목 키를 한글→영문화**: API 키는 영문이 안전 (`benchpress`, `squat`, `deadlift`). UI용 한글 라벨은 `EXERCISE_LABELS` 와 응답의 `exercise_label` 로 별도 제공.
2. **모델 경로 분리**: `MODELS_DIR = backend/models/` 로 통일. 원본 저장소 디렉토리에서 분리해서 백엔드 단독 배포 가능.
3. **로깅 도입**: 원본의 `print` / `st.error` 를 `logging` 으로 교체.
4. **에러 핸들링**: 모델 파일 미존재 시 503, 파일 크기 초과 시 413, 잘못된 입력 시 400 으로 명확히 분기.
5. **응답 스키마 정형화**: `feedbacks` 를 객체 리스트로 만들고 `counted`(점수 반영 여부), `total_duration_sec`, `max_duration_sec` 를 명시적으로 노출.

---

## 5. 점수 / rep 카운트 로직 — 원본과 동일

### rep 카운트
원본 `Streamlit_Upload.py:230~234` 와 100% 동일:
```python
if stage == "down":
    current_stage = "down"
elif stage == "up" and current_stage == "down":
    current_stage = "up"
    rep_count += 1
```
이 알고리즘 자체는 원본 저장소(`Streamlit.py:286~296`) 의 카운터 로직을 따른 것.

### 점수 산식
원본 `Streamlit_Upload.py:130~140` 와 100% 동일:
```
score = max(0, 100 − (유의미 오류 유형 수 × penalty_per_type))
유의미 = max(연속 지속시간) >= min_duration_sec
```

---

## 6. 검증 체크리스트

- [ ] `cd backend && python -m venv .venv && .venv\Scripts\activate && pip install -r requirements.txt`
- [ ] `models/` 에 4개 가중치 파일 배치 (`best_big_bounding.pt`, `benchpress.pkl`, `squat.pkl`, `deadlift.pkl`)
- [ ] `uvicorn app.main:app --reload --port 8000` 실행 → 콘솔에 `Warmup complete` 로그
- [ ] `http://localhost:8000/docs` 접속 가능
- [ ] `GET /api/health` → `{"status":"ok",...}`
- [ ] `GET /api/exercises` → 종목 3개 응답
- [ ] Swagger UI `/api/analyze` 에서 mp4 영상 업로드 → 점수/피드백 응답 확인

---

## 7. 환경 / 의존성 메모

- Python 3.10 권장
- `scikit-learn==1.3.0` 으로 핀 (학습 시 사용한 sklearn 버전과 호환 위해)
- `mediapipe>=0.10` (Windows 한글 경로에서 `.binarypb` 로딩 실패 → ASCII junction 우회 코드 포함)
- `torch.load` 의 `weights_only=False` 패치는 `inference.py::load_yolo_model()` 안에서 처리

---

## 8. 향후 작업 (스코프 외)

원본 가이드 문서 `docs/MIGRATION_FastAPI_React.md` 의 7~13번 항목 참조:
- WebSocket 실시간 모드 (`Streamlit.py` 측 로직)
- 세션 상태 머신 (`ExerciseSession` 클래스)
- 백그라운드 작업 큐 (RQ/Celery)
- 인증, Docker 화 등
