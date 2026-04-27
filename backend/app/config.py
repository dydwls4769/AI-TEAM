from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

MODELS_DIR = BASE_DIR / "models"
YOLO_WEIGHTS_PATH = MODELS_DIR / "best_big_bounding.pt"

EXERCISE_MODEL_PATHS = {
    "benchpress": MODELS_DIR / "benchpress" / "benchpress.pkl",
    "squat": MODELS_DIR / "squat" / "squat.pkl",
    "deadlift": MODELS_DIR / "deadlift" / "deadlift.pkl",
}

EXERCISE_LABELS = {
    "benchpress": "벤치프레스",
    "squat": "스쿼트",
    "deadlift": "데드리프트",
}

CAMERA_GUIDE = {
    "benchpress": "정면에서 촬영하세요. 카메라를 발 아래쪽에 두고 바가 정면으로 보이도록 배치하면 그립 너비와 허리 아치를 판정하기 좋습니다.",
    "squat": "측면에서 촬영하세요. 무릎·허리 라인이 한눈에 보이게 옆에서 찍어야 무릎 안쪽 꺾임과 척추 각도를 판정할 수 있습니다.",
    "deadlift": "측면에서 촬영하세요. 바·허리·무릎이 한 라인에 들어오도록 옆면을 잡으면 척추 중립 여부를 판정하기 좋습니다.",
}

FEEDBACK_MESSAGES = {
    "excessive_arch": "허리가 과도한 아치 자세입니다. 허리를 너무 아치 모양으로 만들지 말고 가슴을 피려고 노력하세요. 골반을 조금 더 들어올리고 복부를 긴장시켜 허리를 평평하게 유지하세요.",
    "arms_spread": "바를 너무 넓게 잡은 자세입니다. 어깨 너비보다 약간만 넓게 잡는 것이 좋습니다.",
    "arms_narrow": "바를 너무 좁게 잡은 자세입니다. 어깨 너비보다 조금 넓게 잡는 것이 좋습니다.",
    "spine_neutral": "척추가 중립이 아닌 자세입니다. 척추가 과도하게 굽지 않도록 가슴을 들어올리고 어깨를 뒤로 넣으세요.",
    "caved_in_knees": "무릎이 움푹 들어간 자세입니다. 엉덩이를 뒤로 빼서 무릎과 발끝을 일직선으로 유지하세요.",
    "feet_spread": "발을 너무 넓게 벌린 자세입니다. 발을 어깨 너비 정도로만 벌리도록 좁히세요.",
}

ERROR_KEYS = list(FEEDBACK_MESSAGES.keys())

DEFAULT_FRAME_SKIP = 3
DEFAULT_YOLO_CONF = 0.5
DEFAULT_PENALTY_PER_TYPE = 25
DEFAULT_MIN_DURATION_SEC = 1.0

ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi"}
MAX_UPLOAD_BYTES = 200 * 1024 * 1024

CORS_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:5173",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5173",
]
