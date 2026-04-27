import logging
import pickle
from collections import Counter
from pathlib import Path
from typing import Callable, Optional

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import torch

from .config import (
    ERROR_KEYS,
    EXERCISE_MODEL_PATHS,
    YOLO_WEIGHTS_PATH,
)

logger = logging.getLogger(__name__)

_yolo_model = None
_classifiers: dict[str, object] = {}
_device: str = "cpu"


def get_device() -> str:
    return _device


def load_yolo_model():
    global _yolo_model, _device
    if _yolo_model is not None:
        return _yolo_model

    if not Path(YOLO_WEIGHTS_PATH).exists():
        raise FileNotFoundError(
            f"YOLO weights not found at {YOLO_WEIGHTS_PATH}. "
            "원본 저장소(PSLeon24/AI_Exercise_Pose_Feedback)의 models/best_big_bounding.pt 를 backend/models/ 에 두세요."
        )

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
    if exercise in _classifiers:
        return _classifiers[exercise]
    if exercise not in EXERCISE_MODEL_PATHS:
        raise ValueError(f"Unknown exercise key: {exercise}")
    path = EXERCISE_MODEL_PATHS[exercise]
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Classifier not found at {path}. 원본 저장소의 models/{exercise}/{exercise}.pkl 를 backend/models/{exercise}/ 에 두세요."
        )
    with open(path, "rb") as f:
        clf = pickle.load(f)
    _classifiers[exercise] = clf
    logger.info("Classifier '%s' loaded", exercise)
    return clf


def warmup():
    load_yolo_model()
    for ex in EXERCISE_MODEL_PATHS:
        try:
            load_classifier(ex)
        except FileNotFoundError as e:
            logger.warning(str(e))


def loaded_classifier_keys() -> list[str]:
    return list(_classifiers.keys())


def yolo_is_loaded() -> bool:
    return _yolo_model is not None


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


def compute_score_from_events(
    event_groups: dict, penalty_per_type: float, min_duration_sec: float
) -> tuple[float, list[str], list[str]]:
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


def analyze_video(
    video_path: str,
    exercise: str,
    frame_skip: int,
    yolo_conf: float,
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> dict:
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

            if stage == "down":
                current_stage = "down"
            elif stage == "up" and current_stage == "down":
                current_stage = "up"
                rep_count += 1

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
