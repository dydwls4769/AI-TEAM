from typing import Optional
from pydantic import BaseModel


class ExerciseInfo(BaseModel):
    key: str
    label: str
    camera_guide: str


class FeedbackItem(BaseModel):
    code: str
    message: str
    occurrences: int
    total_duration_sec: float
    max_duration_sec: float
    counted: bool


class AnalyzeResponse(BaseModel):
    exercise: str
    exercise_label: str
    score: float
    rep_count: int
    analyzed_frames: int
    total_frames: int
    fps: float
    feedbacks: list[FeedbackItem]
    class_distribution: dict[str, int]
    settings: dict


class HealthResponse(BaseModel):
    status: str
    yolo_loaded: bool
    classifiers_loaded: list[str]
    device: str
    detail: Optional[str] = None
