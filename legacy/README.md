# Legacy — Streamlit 원본 자료

이 폴더는 **참고용**입니다. 실제 백엔드 작업은 `../backend/` 에서 진행하세요.

## 들어있는 것
| 파일 | 설명 |
|---|---|
| `Streamlit_Upload.py` | FastAPI 백엔드(`../backend/`)가 포팅한 **원본 Streamlit 앱** |
| `README_UPLOAD.md` | Streamlit 앱 사용법 (로컬에서 빠르게 시각 결과 확인하고 싶을 때) |

## 언제 보면 되나?
1. **결과 비교** — FastAPI 응답이 Streamlit 결과와 일치하는지 확인할 때
2. **빠른 시각 확인** — `streamlit run Streamlit_Upload.py` 로 띄우면 차트/UI까지 즉시 확인 가능
3. **로직 의문** — `analyze_video()` 나 `compute_score_from_events()` 동작이 헷갈릴 때 원본 한 번 더 보기

## 주의
- 이 폴더의 코드는 **유지보수 대상이 아닙니다**. 새 기능은 `../backend/` 에 추가하세요.
- 모델 파일(`*.pt`, `*.pkl`) 은 원본 저장소(`PSLeon24/AI_Exercise_Pose_Feedback`)의 `models/` 폴더에서 그대로 사용합니다.

## 아직 포팅 안 된 것
- 원본 저장소의 `Streamlit.py` (실시간 웹캠 모드, `cv2.VideoCapture(0)` 사용) 는 아직 FastAPI로 옮기지 않았습니다.
- 이 작업이 필요하면 `../docs/MIGRATION_FastAPI_React.md` 의 WebSocket 섹션을 참고하세요.
