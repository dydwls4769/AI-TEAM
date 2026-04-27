import logging
import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .config import (
    ALLOWED_VIDEO_EXTENSIONS,
    CAMERA_GUIDE,
    CORS_ORIGINS,
    DEFAULT_FRAME_SKIP,
    DEFAULT_MIN_DURATION_SEC,
    DEFAULT_PENALTY_PER_TYPE,
    DEFAULT_YOLO_CONF,
    EXERCISE_LABELS,
    EXERCISE_MODEL_PATHS,
    FEEDBACK_MESSAGES,
    MAX_UPLOAD_BYTES,
)
from .inference import (
    analyze_video,
    compute_score_from_events,
    get_device,
    loaded_classifier_keys,
    warmup,
    yolo_is_loaded,
)
from .schemas import AnalyzeResponse, ExerciseInfo, FeedbackItem, HealthResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Warming up models...")
    try:
        warmup()
        logger.info("Warmup complete. device=%s, classifiers=%s", get_device(), loaded_classifier_keys())
    except Exception:
        logger.exception("Warmup failed (server still starting; analyze requests may fail)")
    yield


app = FastAPI(
    title="AI Exercise Pose Feedback API",
    version="1.0.0",
    description="YOLOv5 + MediaPipe + Random Forest 기반 운동 영상 자세 분석 API",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health", response_model=HealthResponse)
def health():
    classifiers = loaded_classifier_keys()
    if yolo_is_loaded() and len(classifiers) == len(EXERCISE_MODEL_PATHS):
        status = "ok"
        detail = None
    elif yolo_is_loaded():
        status = "degraded"
        detail = f"분류기 일부만 로딩됨: {classifiers}"
    else:
        status = "error"
        detail = "YOLO 모델이 로딩되지 않았습니다. backend/models/best_big_bounding.pt 를 확인하세요."
    return HealthResponse(
        status=status,
        yolo_loaded=yolo_is_loaded(),
        classifiers_loaded=classifiers,
        device=get_device(),
        detail=detail,
    )


@app.get("/api/exercises", response_model=list[ExerciseInfo])
def list_exercises():
    return [
        ExerciseInfo(key=k, label=EXERCISE_LABELS[k], camera_guide=CAMERA_GUIDE[k])
        for k in EXERCISE_MODEL_PATHS
    ]


@app.post("/api/analyze", response_model=AnalyzeResponse)
async def analyze(
    video: UploadFile = File(..., description="운동 영상 파일 (mp4/mov/avi)"),
    exercise: str = Form(..., description="benchpress | squat | deadlift"),
    frame_skip: int = Form(DEFAULT_FRAME_SKIP, ge=1, le=10),
    yolo_conf: float = Form(DEFAULT_YOLO_CONF, ge=0.1, le=0.9),
    penalty_per_type: float = Form(DEFAULT_PENALTY_PER_TYPE, ge=5, le=50),
    min_duration_sec: float = Form(DEFAULT_MIN_DURATION_SEC, ge=0.2, le=3.0),
):
    if exercise not in EXERCISE_MODEL_PATHS:
        raise HTTPException(status_code=400, detail=f"unknown exercise: {exercise}")

    suffix = Path(video.filename or "").suffix.lower() or ".mp4"
    if suffix not in ALLOWED_VIDEO_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"허용되지 않는 확장자: {suffix}")

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            written = 0
            while True:
                chunk = await video.read(1024 * 1024)
                if not chunk:
                    break
                written += len(chunk)
                if written > MAX_UPLOAD_BYTES:
                    raise HTTPException(status_code=413, detail="파일이 너무 큽니다 (최대 200MB)")
                tmp.write(chunk)
            tmp_path = tmp.name

        result = analyze_video(
            video_path=tmp_path,
            exercise=exercise,
            frame_skip=frame_skip,
            yolo_conf=yolo_conf,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("analyze failed")
        raise HTTPException(status_code=500, detail=f"분석 중 오류: {e}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    score, significant, _ = compute_score_from_events(
        result["event_groups"], penalty_per_type, min_duration_sec
    )

    feedbacks: list[FeedbackItem] = []
    for key, evs in result["event_groups"].items():
        total_sec = sum(ev["duration_sec"] for ev in evs)
        max_sec = max((ev["duration_sec"] for ev in evs), default=0.0)
        feedbacks.append(
            FeedbackItem(
                code=key,
                message=FEEDBACK_MESSAGES.get(key, key),
                occurrences=len(evs),
                total_duration_sec=round(total_sec, 2),
                max_duration_sec=round(max_sec, 2),
                counted=key in significant,
            )
        )
    feedbacks.sort(key=lambda f: f.total_duration_sec, reverse=True)

    return AnalyzeResponse(
        exercise=exercise,
        exercise_label=EXERCISE_LABELS[exercise],
        score=round(score, 1),
        rep_count=result["rep_count"],
        analyzed_frames=result["analyzed_frames"],
        total_frames=result["total_frames"],
        fps=round(result["fps"], 2),
        feedbacks=feedbacks,
        class_distribution=result["class_counter"],
        settings={
            "frame_skip": frame_skip,
            "yolo_conf": yolo_conf,
            "penalty_per_type": penalty_per_type,
            "min_duration_sec": min_duration_sec,
        },
    )
