# Streamlit → FastAPI + React 마이그레이션 가이드

> 대상 독자: 백엔드 담당자
> 원본: `Streamlit.py`, `Streamlit_Upload.py` (Streamlit 단일 프로세스)
> 목표: FastAPI(추론·상태 관리) + React(카메라·UI) 분리 구조

---

## 1. 현재 구조 요약

`Streamlit.py` 한 파일에 다음이 모두 들어 있음:

| 책임 | 사용 라이브러리 |
|---|---|
| 카메라 캡처 | `cv2.VideoCapture(0)` (서버 측 카메라) |
| UI 렌더링 | `streamlit` (사이드바, 셀렉트박스, 슬라이더, 이미지 위젯) |
| 사람 검출 | `YOLOv5` (`./models/best_big_bounding.pt`, torch.hub) |
| 관절점 추정 | `mediapipe.solutions.pose` |
| 자세 분류 | `sklearn` 모델 3종 pickle (`benchpress.pkl`, `squat.pkl`, `deadlift.pkl`) |
| 카운터/상태 머신 | 전역 변수 `counter`, `current_stage`, `posture_status[]` |
| 음성 피드백 | `pygame.mixer` + `./resources/sounds/*.mp3` |
| 디바이스 | `model.to("mps")` (M1 전용 — 서버에서는 cuda/cpu로 바꿔야 함) |

문제는 카메라 캡처와 UI가 **서버 프로세스에 묶여 있다**는 점이고, 이게 React로 옮길 때 가장 크게 깨지는 부분임.

---

## 2. 목표 아키텍처

```
[브라우저(React)]                              [서버(FastAPI)]
  ┌─────────────┐                               ┌────────────────────────┐
  │ getUserMedia │ ── WebSocket(JPEG/binary) ──▶│ YOLOv5 + MediaPipe     │
  │ <video>      │                               │ + sklearn 분류기        │
  │ Canvas overlay│◀── WebSocket(JSON) ─────────│ + 카운터/상태 머신 (세션)│
  │ <audio>(피드백)│                              └────────────────────────┘
  └─────────────┘
        │
        └── HTTP REST: 운동 종목 변경, 카운터 리셋, 사운드 파일 다운로드
```

핵심 설계 원칙:
- **카메라는 무조건 브라우저에서 잡는다.** 서버 카메라(`cv2.VideoCapture`)는 폐기.
- **추론은 서버에서 한다.** YOLOv5(.pt)와 sklearn(.pkl)을 브라우저로 옮기기는 무리.
- **사운드 재생은 브라우저에서 한다.** `pygame.mixer`는 제거. 서버는 사운드 파일 URL만 응답에 실어 보냄.
- **상태(카운터, 자세 이력)는 세션별로 서버에 둔다.** React는 stateless가 되어야 종목 전환·새로고침이 깔끔함.

---

## 3. 영상 전송 방식 — 3가지 옵션 비교

| 방식 | 지연 | 구현 난이도 | 추천 |
|---|---|---|---|
| **A. WebSocket + JPEG 프레임** | 80~200ms | 낮음 | **MVP 추천** |
| B. WebRTC (`aiortc`) | 30~80ms | 높음 | 안정화 후 고려 |
| C. HTTP 폴링(매 프레임 POST) | 200ms+ | 매우 낮음 | 비추 (오버헤드 큼) |

원본 `requirements.txt`에 `aiortc==1.5.0`, `av==10.0.0`이 이미 들어 있으니 추후 WebRTC로 갈 수도 있지만, 1차 구현은 **WebSocket + JPEG**으로 가는 것이 빠름. 프레임 레이트는 10~15fps면 충분(추론이 그보다 빠르지 않음).

---

## 4. 백엔드 (FastAPI) 디렉토리 구조 제안

```
backend/
├── app/
│   ├── main.py                  # FastAPI 엔트리포인트
│   ├── config.py                # 모델 경로, fps, 디바이스
│   ├── ws/
│   │   └── pose_stream.py       # WebSocket 핸들러
│   ├── inference/
│   │   ├── yolo.py              # YOLOv5 로딩/추론 래퍼
│   │   ├── pose.py              # MediaPipe Pose 래퍼
│   │   ├── classifier.py        # sklearn pkl 로딩
│   │   └── angles.py            # calculateAngle 등 유틸
│   ├── session/
│   │   └── exercise_state.py    # 카운터, posture_status 상태 머신
│   ├── feedback/
│   │   └── messages.py          # FEEDBACK_MESSAGES, 사운드 매핑
│   └── api/
│       ├── exercises.py         # GET /exercises (종목 목록·가이드)
│       └── sessions.py          # POST /sessions (세션 시작), DELETE (리셋)
├── models/                      # 기존 ./models 그대로 옮김
├── resources/sounds/            # 기존 mp3 그대로 (StaticFiles로 서빙)
└── requirements.txt
```

---

## 5. 핵심 API 명세

### 5.1 REST

| Method | Path | 설명 | 응답 |
|---|---|---|---|
| GET | `/api/exercises` | 종목 목록 + 카메라 가이드 | `[{key, label, guide}]` |
| POST | `/api/sessions` | 새 운동 세션 생성 | `{session_id, exercise}` |
| GET | `/api/sessions/{id}` | 현재 카운터·상태 조회 | `{counter, current_stage, last_feedback}` |
| DELETE | `/api/sessions/{id}` | 카운터 리셋/세션 종료 | `204` |
| GET | `/static/sounds/*.mp3` | 사운드 파일 (FastAPI `StaticFiles`로 서빙) | 바이너리 |

### 5.2 WebSocket — `/ws/pose/{session_id}`

**클라이언트 → 서버 (binary 프레임)**
- 메시지 타입 1: `bytes` — JPEG 인코딩된 프레임 (브라우저 `<canvas>.toBlob('image/jpeg', 0.6)`)
- 메시지 타입 2: `text/json` — 제어 명령 (`{"type": "change_exercise", "exercise": "squat"}`)

**서버 → 클라이언트 (JSON)**
```json
{
  "frame_id": 123,
  "landmarks": [{"x": 0.5, "y": 0.3, "z": 0.0, "visibility": 0.9}, ...],
  "bbox": [x1, y1, x2, y2],
  "angles": {
    "left_knee": 145.3, "right_knee": 142.1, "neck": 88.0, ...
  },
  "exercise_class": "down_correct",
  "stage": "down",
  "counter": 7,
  "feedback": {
    "code": "caved_in_knees",
    "message": "무릎이 움푹 들어간 자세입니다...",
    "sound_url": "/static/sounds/caved_in_knees_feedback_1.mp3"
  } // 피드백 없으면 null
}
```

랜드마크는 정규화 좌표(0~1)로 보내고, **그리기는 React에서 `<canvas>` overlay로 처리**한다. 서버에서 그려서 다시 인코딩해 보내면 대역폭/지연 모두 커짐.

---

## 6. Streamlit 코드 → 백엔드 모듈 매핑

원본 `Streamlit.py`의 어떤 코드가 어디로 가는지:

| 원본 라인 | 원본 코드 | 이동 위치 |
|---|---|---|
| 19~23 | YOLOv5 로딩, `model.to("mps")` | `inference/yolo.py` (디바이스는 env로) |
| 34~47 | `calculateAngle()` | `inference/angles.py` |
| 51~55 | `detect_objects()` | `inference/yolo.py` |
| 73~88 | `pickle.load` 종목별 분류기 | `inference/classifier.py` (시작 시 3개 모두 로딩 후 캐시) |
| 91~97 | `cv2.VideoCapture(0)`, MediaPipe Pose 초기화 | 카메라는 **삭제**, Pose는 `inference/pose.py`로 (세션당 1개 인스턴스) |
| 115~141 | 메인 루프, YOLO→ROI→Pose | `ws/pose_stream.py`의 메시지 핸들러 |
| 143~273 | 랜드마크 추출 + 각도 계산 | 그대로 옮기되 결과를 dict로 반환 |
| 276~403 | 카운터·자세 머신, 피드백 분기 | `session/exercise_state.py` 클래스로 캡슐화 |
| `pygame.mixer.music.load/play` 전체 | — | **전부 삭제.** 응답 JSON에 `sound_url`만 실어 보냄. 재생은 React `<audio>` |
| `st.error/st.info` 전체 | — | 응답 JSON의 `feedback.message`로 |

---

## 7. 세션 상태 머신 (가장 중요)

원본의 카운터 로직(280~296행)은 전역 변수에 의존하기 때문에 그대로 옮기면 동시 접속자가 섞임. 세션 단위로 격리해야 함:

```python
# app/session/exercise_state.py
from collections import deque
from time import time

class ExerciseSession:
    def __init__(self, exercise: str):
        self.exercise = exercise
        self.counter = 0
        self.current_stage = ""
        self.posture_status: list[str] = []
        self.previous_alert_time = 0.0
        self.alert_cooldown = 3.0  # 원본의 3초 쿨다운

    def update(self, exercise_class: str) -> dict | None:
        """원본 286~403행의 카운터·피드백 로직.
        반환값은 클라이언트에 보낼 feedback dict 또는 None."""
        feedback = None
        if "down" in exercise_class:
            self.current_stage = "down"
            self.posture_status.append(exercise_class)
        elif self.current_stage == "down" and "up" in exercise_class:
            self.current_stage = "up"
            self.counter += 1
            self.posture_status.append(exercise_class)
            feedback = self._resolve_feedback()
        return feedback

    def _resolve_feedback(self) -> dict | None:
        # most_frequent + 쿨다운 + FEEDBACK_MESSAGES 매핑
        ...
```

세션 저장소는 처음에는 in-memory dict면 충분(`sessions: dict[str, ExerciseSession] = {}`). 멀티 워커로 갈 때만 Redis 고려.

---

## 8. 디바이스 / 모델 로딩 주의사항

원본 22행의 `model.to("mps")`는 **M1 맥 전용**. 서버 환경에 맞게 바꿔야 함:

```python
# app/inference/yolo.py
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model = torch.hub.load("ultralytics/yolov5", "custom", path=YOLO_WEIGHTS_PATH)
model.to(DEVICE).eval()
```

추가로 `Streamlit_Upload.py`의 76~79행에 `torch.load`의 `weights_only=False` 패치가 있음 — 최신 PyTorch에서 .pt 로딩이 깨지는 이슈 회피용이니 백엔드에도 동일하게 적용해야 함.

`mediapipe.Pose` 인스턴스는 **세션당 1개**를 만들고 재사용해야 함(생성 비용이 큼). 매 프레임 새로 만들면 안 됨.

---

## 9. requirements.txt (백엔드용 정리본)

원본 `requirements.txt`는 Streamlit·gTTS·pygame·dev 도구가 잔뜩 섞여 있음. 백엔드에 필요한 최소 셋:

```
fastapi>=0.110
uvicorn[standard]>=0.27
python-multipart  # 파일 업로드 쓸 경우
websockets

# 추론
torch>=2.0
torchvision
opencv-python-headless  # 서버는 headless 버전
mediapipe>=0.10
ultralytics  # 또는 yolov5 직접 (torch.hub 쓰는 현재 방식 유지 가능)
scikit-learn  # pickle 호환 위해 학습 시 버전과 맞춰야 함
numpy
pandas
pillow
```

**주의:** sklearn pickle은 학습 시 사용한 버전과 로딩 시 버전이 다르면 경고/에러가 남. 원본 학습 환경의 sklearn 버전을 확인해서 핀(pin)으로 고정하는 것을 권장.

---

## 10. 프론트엔드(React) 측에서 백엔드에 기대하는 것 — 인터페이스 합의용

백엔드 담당자가 결정만 해주면 프론트는 다음을 구현:

1. `getUserMedia`로 카메라 스트림 → `<video>` 표시
2. `requestAnimationFrame` 또는 `setInterval(100ms)`로 `<canvas>`에 프레임 캡처 → `toBlob('image/jpeg', 0.6)` → WebSocket으로 전송
3. WebSocket 수신 → 랜드마크/bbox를 overlay `<canvas>`에 그리기, 카운터 표시, `feedback.sound_url`이 오면 `<audio>` 재생

따라서 백엔드가 확정해줘야 할 것:
- WebSocket 엔드포인트 URL과 메시지 포맷 (위 5.2)
- JPEG 품질·해상도 권장값 (예: 640×480, quality 0.6)
- 사운드 파일 정적 서빙 경로 (CORS 헤더 포함)
- CORS 설정 (`fastapi.middleware.cors.CORSMiddleware`로 React dev 서버 origin 허용)

---

## 11. 단계별 작업 제안

| 단계 | 작업 | 검증 방법 |
|---|---|---|
| 1 | FastAPI 스켈레톤 + `/api/exercises` REST | curl로 200 응답 확인 |
| 2 | YOLO/MediaPipe/분류기 로딩만 단독 스크립트로 분리 (Streamlit 의존 제거) | 샘플 이미지 1장 넣고 추론 결과 dict 출력 |
| 3 | WebSocket 핸들러 + `ExerciseSession` 클래스 | Postman/wscat으로 수동 프레임 전송 후 응답 JSON 확인 |
| 4 | React 쪽과 인터페이스 맞추기 (CORS, 메시지 스키마) | 브라우저에서 카메라→서버→오버레이까지 E2E |
| 5 | 사운드 파일 StaticFiles 서빙 + 피드백 쿨다운 동작 확인 | 일부러 잘못된 자세로 3초 쿨다운 검증 |
| 6 | 디바이스 분기, 모델 로딩 캐시, 동시 세션 테스트 | 2개 탭 동시 접속, 카운터 격리 확인 |

---

## 12. 마이그레이션 시 빠뜨리기 쉬운 것

- 원본 117행 `cv2.flip(frame, 1)` (좌우반전) — 거울 모드. **프론트에서 `<video>`에 `transform: scaleX(-1)`로 처리**하고, 서버로 보낼 때는 원본을 보내거나 합의된 한 방향으로 통일할 것.
- 원본 207~210행의 `neck_angle` 계산식에 연산자 우선순위 버그처럼 보이는 부분이 있음(`+ ... / 2`가 평균이 아니라 우측 항만 절반). 옮길 때 의도 확인 필요.
- 원본 405행 `except Exception: pass`로 모든 에러를 삼키고 있음. 마이그레이션할 때는 **반드시 로깅으로 바꿀 것** — 안 그러면 추론 실패가 조용히 묻혀서 디버깅 불가.
- `Streamlit_Upload.py` 8~36행의 한글 경로 우회 코드는 서버 배포 시 경로를 ASCII로 두면 불필요. 백엔드 컨테이너/배포 경로는 ASCII로 강제.
- `posture_status`가 무한히 쌓이지 않도록 길이 제한(`deque(maxlen=30)` 권장).

---

## 13. 최소 동작 예시 (참고)

```python
# app/main.py
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import cv2, numpy as np, uuid

from app.inference.yolo import detect_person
from app.inference.pose import estimate_pose
from app.inference.classifier import classify
from app.session.exercise_state import ExerciseSession

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173"], allow_methods=["*"], allow_headers=["*"])
app.mount("/static/sounds", StaticFiles(directory="resources/sounds"), name="sounds")

sessions: dict[str, ExerciseSession] = {}

@app.post("/api/sessions")
def create_session(exercise: str):
    sid = str(uuid.uuid4())
    sessions[sid] = ExerciseSession(exercise)
    return {"session_id": sid, "exercise": exercise}

@app.websocket("/ws/pose/{sid}")
async def pose_stream(ws: WebSocket, sid: str):
    await ws.accept()
    sess = sessions[sid]
    try:
        while True:
            data = await ws.receive_bytes()
            frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            bbox = detect_person(frame)
            if bbox is None:
                await ws.send_json({"landmarks": None}); continue
            landmarks, angles = estimate_pose(frame, bbox)
            ex_class = classify(sess.exercise, landmarks)
            feedback = sess.update(ex_class)
            await ws.send_json({
                "bbox": bbox, "landmarks": landmarks, "angles": angles,
                "stage": sess.current_stage, "counter": sess.counter,
                "feedback": feedback,
            })
    except Exception as e:
        await ws.close()
```

---

## 14. 정리 — 백엔드 담당자에게 결정 요청드릴 사항

1. 영상 전송 방식: **WebSocket+JPEG (MVP)** vs WebRTC (정식) — 추천: 1차는 전자
2. 세션 저장소: in-memory dict vs Redis — 추천: 1차는 in-memory
3. 추론 디바이스: CUDA GPU 서버가 있는지, 없으면 CPU 가능 fps 측정 필요
4. sklearn 버전 핀: 학습 시 버전 확인 후 requirements에 고정
5. CORS 허용 origin (React dev 서버 포트 / 배포 도메인)

이 다섯 개만 정해지면 위 디렉토리 구조대로 1주 내 MVP 가능합니다.
