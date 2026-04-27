"""
pose_analyzer.py — AI 운동 자세 분석 단일 파일 모듈

YOLOv5(사람 검출) + MediaPipe(관절 추정) + Random Forest(자세 분류)
파이프라인을 한 파일에 모은 순수 파이썬 모듈입니다.

FastAPI / Flask / Django 등 어떤 웹 프레임워크에서도 사용할 수 있고,
CLI 로 단독 실행해 테스트도 가능합니다.

──────────────────────────────────────────────────────────────────────────
사용법
──────────────────────────────────────────────────────────────────────────

  from pose_analyzer import analyze_video, compute_score_from_events, FEEDBACK_MESSAGES

  result = analyze_video(
      video_path="my_squat.mp4",
      exercise="squat",                # benchpress | squat | deadlift
      frame_skip=3,
      yolo_conf=0.5,
  )
  score, significant_keys, _ = compute_score_from_events(
      result["event_groups"], penalty_per_type=25, min_duration_sec=1.0
  )
  print(score, result["rep_count"])

──────────────────────────────────────────────────────────────────────────
필수 모델 파일 배치
──────────────────────────────────────────────────────────────────────────

  <MODELS_DIR>/best_big_bounding.pt
  <MODELS_DIR>/benchpress/benchpress.pkl
  <MODELS_DIR>/squat/squat.pkl
  <MODELS_DIR>/deadlift/deadlift.pkl

기본 MODELS_DIR 는 본 파일과 같은 디렉토리의 "models/" 입니다.
환경변수 POSE_MODELS_DIR 로 덮어쓸 수 있습니다.

──────────────────────────────────────────────────────────────────────────
의존성
──────────────────────────────────────────────────────────────────────────

  pip install torch torchvision opencv-python mediapipe \
              ultralytics scikit-learn==1.3.0 numpy pandas pillow
"""

from __future__ import annotations

import logging
import os
import pickle
import sys
from collections import Counter
from pathlib import Path
from typing import Callable, Optional


# ─────────────────────────────────────────────────────────────────────────
# Windows 한글 경로 우회 (mediapipe import 전에 실행되어야 함)
# ─────────────────────────────────────────────────────────────────────────
def _setup_ascii_mediapipe_path():
    if sys.platform != "win32":
        return
    site_pkg = None
    for candidate in sys.path:
        if candidate.lower().endswith("site-packages") and os.path.isdir(candidate):
            try:
                candidate.encode("ascii")
            except UnicodeEncodeError:
                site_pkg = candidate
                break
    if site_pkg is None:
        return
    link = r"C:\mp_ascii_path"
    if not os.path.exists(link):
        import subprocess
        subprocess.run(
            ["cmd", "/c", "mklink", "/J", link, site_pkg],
            check=False,
            capture_output=True,
        )
    if os.path.exists(link) and link not in sys.path:
        sys.path.insert(0, link)


_setup_ascii_mediapipe_path()


import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import torch


# ─────────────────────────────────────────────────────────────────────────
# 경로 / 상수 — 환경에 맞게 수정하거나 환경변수로 덮어쓰세요
# ─────────────────────────────────────────────────────────────────────────
_DEFAULT_MODELS_DIR = Path(__file__).resolve().parent / "models"
MODELS_DIR = Path(os.environ.get("POSE_MODELS_DIR", _DEFAULT_MODELS_DIR))

YOLO_WEIGHTS_PATH = MODELS_DIR / "best_big_bounding.pt"

EXERCISE_MODEL_PATHS = {
    "benchpress": MODELS_DIR / "benchpress" / "benchpress.pkl",
    "squat":      MODELS_DIR / "squat"      / "squat.pkl",
    "deadlift":   MODELS_DIR / "deadlift"   / "deadlift.pkl",
}

EXERCISE_LABELS = {
    "benchpress": "벤치프레스",
    "squat":      "스쿼트",
    "deadlift":   "데드리프트",
}

CAMERA_GUIDE = {
    "benchpress": "정면에서 촬영하세요. 카메라를 발 아래쪽에 두고 바가 정면으로 보이도록 배치하면 그립 너비와 허리 아치를 판정하기 좋습니다.",
    "squat":      "측면에서 촬영하세요. 무릎·허리 라인이 한눈에 보이게 옆에서 찍어야 무릎 안쪽 꺾임과 척추 각도를 판정할 수 있습니다.",
    "deadlift":   "측면에서 촬영하세요. 바·허리·무릎이 한 라인에 들어오도록 옆면을 잡으면 척추 중립 여부를 판정하기 좋습니다.",
}

FEEDBACK_MESSAGES = {
    "excessive_arch": "허리가 과도한 아치 자세입니다. 허리를 너무 아치 모양으로 만들지 말고 가슴을 피려고 노력하세요. 골반을 조금 더 들어올리고 복부를 긴장시켜 허리를 평평하게 유지하세요.",
    "arms_spread":    "바를 너무 넓게 잡은 자세입니다. 어깨 너비보다 약간만 넓게 잡는 것이 좋습니다.",
    "arms_narrow":    "바를 너무 좁게 잡은 자세입니다. 어깨 너비보다 조금 넓게 잡는 것이 좋습니다.",
    "spine_neutral":  "척추가 중립이 아닌 자세입니다. 척추가 과도하게 굽지 않도록 가슴을 들어올리고 어깨를 뒤로 넣으세요.",
    "caved_in_knees": "무릎이 움푹 들어간 자세입니다. 엉덩이를 뒤로 빼서 무릎과 발끝을 일직선으로 유지하세요.",
    "feet_spread":    "발을 너무 넓게 벌린 자세입니다. 발을 어깨 너비 정도로만 벌리도록 좁히세요.",
}

ERROR_KEYS = list(FEEDBACK_MESSAGES.keys())

DEFAULT_FRAME_SKIP = 3
DEFAULT_YOLO_CONF = 0.5
DEFAULT_PENALTY_PER_TYPE = 25
DEFAULT_MIN_DURATION_SEC = 1.0


# ─────────────────────────────────────────────────────────────────────────
# 모델 로딩 — 모듈 전역에 1회 캐시 (요청마다 다시 안 로드)
# ─────────────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

_yolo_model = None
_classifiers: dict[str, object] = {}
_device: str = "cpu"


def get_device() -> str:
    return _device


def yolo_is_loaded() -> bool:
    return _yolo_model is not None


def loaded_classifier_keys() -> list[str]:
    return list(_classifiers.keys())


def load_yolo_model():
    """YOLOv5 가중치 로딩 (PyTorch hub 캐시 사용). 한 번만 실행됨."""
    global _yolo_model, _device
    if _yolo_model is not None:
        return _yolo_model

    if not Path(YOLO_WEIGHTS_PATH).exists():
        raise FileNotFoundError(
            f"YOLO 가중치를 찾을 수 없습니다: {YOLO_WEIGHTS_PATH}\n"
            f"원본 저장소(PSLeon24/AI_Exercise_Pose_Feedback)의 models/best_big_bounding.pt 를 "
            f"위 경로에 두거나 환경변수 POSE_MODELS_DIR 를 적절히 설정하세요."
        )

    # 최신 PyTorch에서 .pt 로딩 시 weights_only=True 가 기본이라 깨짐 → 임시 패치
    original_load = torch.load

    def _unsafe_load(*args, **kwargs):
        kwargs["weights_only"] = False
        return original_load(*args, **kwargs)

    torch.load = _unsafe_load
    try:
        model = torch.hub.load(
            "ultralytics/yolov5:v7.0",
            "custom",
            path=str(YOLO_WEIGHTS_PATH),
            force_reload=False,
            trust_repo=True,
        )
    finally:
        torch.load = original_load

    _device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(_device)
    model.eval()
    model.conf = 0.5
    _yolo_model = model
    logger.info("YOLOv5 loaded on device=%s", _device)
    return _yolo_model


def load_classifier(exercise: str):
    """운동 종목별 sklearn 분류기 로딩."""
    if exercise in _classifiers:
        return _classifiers[exercise]
    if exercise not in EXERCISE_MODEL_PATHS:
        raise ValueError(f"알 수 없는 운동 종목: {exercise}")
    path = EXERCISE_MODEL_PATHS[exercise]
    if not Path(path).exists():
        raise FileNotFoundError(
            f"분류기 파일을 찾을 수 없습니다: {path}\n"
            f"원본 저장소의 models/{exercise}/{exercise}.pkl 를 위 경로에 두세요."
        )
    with open(path, "rb") as f:
        clf = pickle.load(f)
    _classifiers[exercise] = clf
    logger.info("Classifier '%s' loaded", exercise)
    return clf


def warmup() -> None:
    """서버 시작 시 호출하면 첫 요청이 빨라집니다 (선택)."""
    load_yolo_model()
    for ex in EXERCISE_MODEL_PATHS:
        try:
            load_classifier(ex)
        except FileNotFoundError as e:
            logger.warning(str(e))


# ─────────────────────────────────────────────────────────────────────────
# 헬퍼
# ─────────────────────────────────────────────────────────────────────────
def _extract_landmark_row(pose_landmarks):
    return [
        coord
        for lm in pose_landmarks.landmark
        for coord in [lm.x, lm.y, lm.z, lm.visibility]
    ]


def _classify_posture(class_name: str):
    name = str(class_name).lower()
    stage = None
    if "down" in name:
        stage = "down"
    elif "up" in name:
        stage = "up"
    error_key = None
    for key in ERROR_KEYS:
        if key in name:
            error_key = key
            break
    return stage, error_key


# ─────────────────────────────────────────────────────────────────────────
# 점수 계산
# ─────────────────────────────────────────────────────────────────────────
def compute_score_from_events(
    event_groups: dict,
    penalty_per_type: float = DEFAULT_PENALTY_PER_TYPE,
    min_duration_sec: float = DEFAULT_MIN_DURATION_SEC,
) -> tuple[float, list[str], list[str]]:
    """
    점수 = max(0, 100 − (유의미_오류_유형_수 × penalty_per_type))
    유의미: 같은 오류가 연속으로 min_duration_sec 이상 지속될 때만 1회로 카운트
    """
    significant: list[str] = []
    filtered: list[str] = []
    for key, evs in event_groups.items():
        max_dur = max((ev["duration_sec"] for ev in evs), default=0.0)
        if max_dur >= min_duration_sec:
            significant.append(key)
        else:
            filtered.append(key)
    score = max(0.0, 100.0 - len(significant) * penalty_per_type)
    return score, significant, filtered


# ─────────────────────────────────────────────────────────────────────────
# 메인 분석 함수
# ─────────────────────────────────────────────────────────────────────────
def analyze_video(
    video_path: str,
    exercise: str,
    frame_skip: int = DEFAULT_FRAME_SKIP,
    yolo_conf: float = DEFAULT_YOLO_CONF,
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> dict:
    """
    영상 1개를 분석해 rep 카운트와 자세 오류 이벤트를 반환합니다.

    Args:
        video_path: 분석할 영상 파일 경로
        exercise:   "benchpress" | "squat" | "deadlift"
        frame_skip: n프레임마다 1번 분석 (1~10, 클수록 빠름)
        yolo_conf:  YOLO 사람 검출 신뢰도 임계값 (0.1~0.9)
        progress_cb: 진행률 콜백 fn(current, total) — 선택

    Returns dict:
        total_frames, analyzed_frames, rep_count, fps,
        error_counter:  {error_key: count, ...}
        class_counter:  {class_name: count, ...}
        event_groups:   {error_key: [{duration_sec, start_sec}, ...]}
        correct_prob_sum: 정상 자세일 확률 합 (참고용)
    """
    yolo = load_yolo_model()
    yolo.conf = yolo_conf
    classifier = load_classifier(exercise)
    classifier_classes = [str(c) for c in classifier.classes_]
    correct_class_mask = np.array(
        ["correct" in c.lower() for c in classifier_classes], dtype=bool
    )

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.7,
        model_complexity=1,
    )

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    analyzed = 0
    correct_prob_sum = 0.0
    error_counter: Counter = Counter()
    class_counter: Counter = Counter()
    rep_count = 0
    current_stage = ""

    events: list[dict] = []
    current_event: Optional[dict] = None

    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_skip != 0:
                frame_idx += 1
                if progress_cb:
                    progress_cb(frame_idx, total_frames)
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            yolo_results = yolo(frame_rgb)
            preds = yolo_results.pred[0]

            crop = None
            if preds is not None and len(preds) > 0:
                best = preds[preds[:, 4].argmax()]
                x1, y1, x2, y2 = [int(v.item()) for v in best[:4]]
                h, w = frame_rgb.shape[:2]
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w, x2)
                y2 = min(h, y2)
                if x2 > x1 and y2 > y1:
                    crop = frame_rgb[y1:y2, x1:x2]

            if crop is None or crop.size == 0:
                frame_idx += 1
                if progress_cb:
                    progress_cb(frame_idx, total_frames)
                continue

            pose_result = pose.process(crop)
            if pose_result.pose_landmarks is None:
                frame_idx += 1
                if progress_cb:
                    progress_cb(frame_idx, total_frames)
                continue

            row = _extract_landmark_row(pose_result.pose_landmarks)
            X = pd.DataFrame([row])
            try:
                probs = classifier.predict_proba(X)[0]
                pred_class = classifier_classes[int(np.argmax(probs))]
            except Exception:
                logger.exception("classifier predict failed")
                frame_idx += 1
                if progress_cb:
                    progress_cb(frame_idx, total_frames)
                continue

            class_counter[str(pred_class)] += 1
            stage, error_key = _classify_posture(pred_class)

            # rep 카운트: down → up 전환 시 +1
            if stage == "down":
                current_stage = "down"
            elif stage == "up" and current_stage == "down":
                current_stage = "up"
                rep_count += 1

            # 오류 이벤트 그룹화 (연속된 동일 오류는 하나의 이벤트로)
            if error_key is not None:
                if current_event is not None and current_event["key"] == error_key:
                    current_event["end_frame"] = frame_idx
                else:
                    if current_event is not None:
                        events.append(current_event)
                    current_event = {
                        "key": error_key,
                        "start_frame": frame_idx,
                        "end_frame": frame_idx,
                    }
            else:
                if current_event is not None:
                    events.append(current_event)
                    current_event = None

            if stage in ("up", "down"):
                analyzed += 1
                correct_prob_sum += float(probs[correct_class_mask].sum())
                if error_key is not None:
                    error_counter[error_key] += 1

            frame_idx += 1
            if progress_cb:
                progress_cb(frame_idx, total_frames)
    finally:
        if current_event is not None:
            events.append(current_event)
        cap.release()
        pose.close()

    # 이벤트 → {error_key: [{duration_sec, start_sec}, ...]}
    event_groups: dict[str, list[dict]] = {}
    for ev in events:
        dur_frames = ev["end_frame"] - ev["start_frame"] + 1
        dur_sec = dur_frames / fps if fps > 0 else 0.0
        start_sec = ev["start_frame"] / fps if fps > 0 else 0.0
        event_groups.setdefault(ev["key"], []).append(
            {"duration_sec": dur_sec, "start_sec": start_sec}
        )

    return {
        "total_frames": total_frames,
        "analyzed_frames": analyzed,
        "rep_count": rep_count,
        "error_counter": dict(error_counter),
        "class_counter": dict(class_counter),
        "event_groups": event_groups,
        "fps": fps,
        "correct_prob_sum": correct_prob_sum,
    }


# ─────────────────────────────────────────────────────────────────────────
# 결과 → 클라이언트 응답용 dict 빌더 (선택, 편의 함수)
# ─────────────────────────────────────────────────────────────────────────
def build_response_payload(
    raw_result: dict,
    exercise: str,
    penalty_per_type: float = DEFAULT_PENALTY_PER_TYPE,
    min_duration_sec: float = DEFAULT_MIN_DURATION_SEC,
) -> dict:
    """analyze_video 결과를 점수 + 피드백 dict 로 변환합니다."""
    score, significant, _ = compute_score_from_events(
        raw_result["event_groups"], penalty_per_type, min_duration_sec
    )

    feedbacks = []
    for key, evs in raw_result["event_groups"].items():
        total_sec = sum(ev["duration_sec"] for ev in evs)
        max_sec = max((ev["duration_sec"] for ev in evs), default=0.0)
        feedbacks.append({
            "code": key,
            "message": FEEDBACK_MESSAGES.get(key, key),
            "occurrences": len(evs),
            "total_duration_sec": round(total_sec, 2),
            "max_duration_sec": round(max_sec, 2),
            "counted": key in significant,
        })
    feedbacks.sort(key=lambda f: f["total_duration_sec"], reverse=True)

    return {
        "exercise": exercise,
        "exercise_label": EXERCISE_LABELS.get(exercise, exercise),
        "score": round(score, 1),
        "rep_count": raw_result["rep_count"],
        "analyzed_frames": raw_result["analyzed_frames"],
        "total_frames": raw_result["total_frames"],
        "fps": round(raw_result["fps"], 2),
        "feedbacks": feedbacks,
        "class_distribution": raw_result["class_counter"],
        "settings": {
            "penalty_per_type": penalty_per_type,
            "min_duration_sec": min_duration_sec,
        },
    }


# ─────────────────────────────────────────────────────────────────────────
# CLI 테스트: python pose_analyzer.py <video_path> <exercise>
# ─────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse, json

    parser = argparse.ArgumentParser(description="자세 분석 CLI 테스트")
    parser.add_argument("video", help="영상 파일 경로")
    parser.add_argument("exercise", choices=list(EXERCISE_MODEL_PATHS.keys()))
    parser.add_argument("--frame-skip", type=int, default=DEFAULT_FRAME_SKIP)
    parser.add_argument("--yolo-conf", type=float, default=DEFAULT_YOLO_CONF)
    parser.add_argument("--penalty", type=float, default=DEFAULT_PENALTY_PER_TYPE)
    parser.add_argument("--min-duration", type=float, default=DEFAULT_MIN_DURATION_SEC)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    raw = analyze_video(
        video_path=args.video,
        exercise=args.exercise,
        frame_skip=args.frame_skip,
        yolo_conf=args.yolo_conf,
    )
    payload = build_response_payload(
        raw, args.exercise, penalty_per_type=args.penalty, min_duration_sec=args.min_duration
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
