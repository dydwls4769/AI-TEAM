import os
import sys
import subprocess
import tempfile
import base64
from collections import Counter


def _setup_ascii_mediapipe_path():
    # MediaPipe on Windows fails to open .binarypb files when the site-packages
    # path contains non-ASCII characters (e.g. Korean "바탕 화면"). Junction the
    # venv site-packages to an ASCII location and import mediapipe from there.
    if sys.platform != "win32":
        return
    site_pkg = None
    for candidate in sys.path:
        if candidate.lower().endswith("site-packages") and os.path.isdir(candidate):
            if "venv" in candidate.lower():
                site_pkg = candidate
                break
    if site_pkg is None:
        return
    try:
        site_pkg.encode("ascii")
        return
    except UnicodeEncodeError:
        pass
    link = r"C:\mp_ascii_path"
    if not os.path.exists(link):
        subprocess.run(
            ["cmd", "/c", "mklink", "/J", link, site_pkg],
            check=False,
            capture_output=True,
        )
    if os.path.exists(link):
        sys.path.insert(0, link)


_setup_ascii_mediapipe_path()

import cv2
import numpy as np
import pandas as pd
import pickle
import streamlit as st
import mediapipe as mp
import torch
import plotly.graph_objects as go
from PIL import Image, ImageDraw, ImageFont
import imageio.v2 as imageio

YOLO_WEIGHTS_PATH = "./models/best_big_bounding.pt"

EXERCISE_PIPELINES = {
    "벤치프레스": {
        "type": "rf_yolo",
        "model_path": "./models/benchpress/benchpress.pkl",
    },
    "스쿼트": {
        "type": "rf_yolo",
        "model_path": "./models/squat/squat.pkl",
    },
    "데드리프트": {
        "type": "rf_yolo",
        "model_path": "./models/deadlift/deadlift.pkl",
    },
    "바벨 컬": {
        "type": "rule_based",
        "state_key": "bicep_curl_side",
    },
    "플랭크": {
        "type": "rule_based",
        "state_key": "plank",
        "static": True,
    },
    "런지": {
        "type": "rule_based",
        "state_key": "lunge",
    },
    "오버헤드 프레스": {
        "type": "rule_based",
        "state_key": "ohp",
    },
    "바벨 로우": {
        "type": "rule_based",
        "state_key": "barbell_row",
    },
    "인클라인 벤치프레스": {
        "type": "rule_based",
        "state_key": "incline_bench",
    },
    "랫풀다운": {
        "type": "rule_based",
        "state_key": "lat_pulldown",
    },
    "사이드 레터럴 레이즈": {
        "type": "rule_based",
        "state_key": "side_lateral_raise",
    },
    "머신플라이": {
        "type": "rule_based",
        "state_key": "machine_fly",
    },
    "스모 데드리프트": {
        "type": "rule_based",
        "state_key": "sumo_deadlift",
    },
    "프론트 레이즈": {
        "type": "rule_based",
        "state_key": "front_raise",
    },
    "클로즈 그립 벤치프레스": {
        "type": "rule_based",
        "state_key": "close_grip_bench",
    },
    "라잉 트라이셉스 익스텐션": {
        "type": "rule_based",
        "state_key": "skull_crusher",
    },
    "크런치": {
        "type": "rule_based",
        "state_key": "crunch",
    },
    "레그 레이즈": {
        "type": "rule_based",
        "state_key": "leg_raise",
    },
}

EXERCISE_CATEGORIES = {
    "가슴": ["벤치프레스", "인클라인 벤치프레스", "머신플라이"],
    "등": ["바벨 로우", "데드리프트", "랫풀다운"],
    "하체": ["스쿼트", "런지", "스모 데드리프트"],
    "어깨": ["오버헤드 프레스", "사이드 레터럴 레이즈", "프론트 레이즈"],
    "복근": ["플랭크", "크런치", "레그 레이즈"],
    "팔": ["바벨 컬", "클로즈 그립 벤치프레스", "라잉 트라이셉스 익스텐션"],
}

CAMERA_GUIDE = {
    "벤치프레스": "정면에서 촬영하세요. 카메라를 발 아래쪽에 두고 바가 정면으로 보이도록 배치하면 그립 너비와 허리 아치를 판정하기 좋습니다.",
    "스쿼트": "측면에서 촬영하세요. 무릎·허리 라인이 한눈에 보이게 옆에서 찍어야 무릎 안쪽 꺾임과 척추 각도를 판정할 수 있습니다.",
    "데드리프트": "측면에서 촬영하세요. 바·허리·무릎이 한 라인에 들어오도록 옆면을 잡으면 척추 중립 여부를 판정하기 좋습니다.",
    "바벨 컬": "측면에서 촬영하세요. 한쪽 팔(어깨·팔꿈치·손목)과 골반이 한 화면에 들어와야 상체 반동과 팔꿈치 흔들림을 판정할 수 있습니다.",
    "플랭크": "측면에서 촬영하세요. 머리부터 발끝까지 일직선이 한 화면에 들어와야 엉덩이 높낮이를 판정할 수 있습니다.",
    "런지": "측면에서 촬영하세요. 앞다리 무릎 각도와 발끝 위치가 보여야 무릎이 발끝을 넘는지를 판정할 수 있습니다.",
    "오버헤드 프레스": "정면에서 촬영하세요. 양 어깨·팔꿈치·손목이 한 화면에 들어와야 좌우 균형과 락아웃 정도를 판정할 수 있습니다.",
    "바벨 로우": "측면에서 촬영하세요. 어깨·골반·무릎·팔꿈치가 한 라인에 보여야 척추 중립과 힙 힌지 각도를 판정할 수 있습니다.",
    "인클라인 벤치프레스": "정면에서 촬영하세요. 인클라인 각도(30~45°) 기준 발 아래쪽에서 바가 정면으로 보이게 배치하면 그립 너비와 좌우 밸런스를 판정할 수 있습니다.",
    "랫풀다운": "정면에서 촬영하세요. 머리부터 골반까지 상체와 양팔이 한 화면에 들어와야 그립 너비, 상체 기울기, 좌우 균형을 판정할 수 있습니다.",
    "사이드 레터럴 레이즈": "정면에서 촬영하세요. 양팔이 측면으로 어깨 라인까지 올라가는 모습이 화면에 들어와야 가동범위와 좌우 비대칭을 판정할 수 있습니다.",
    "머신플라이": "정면에서 촬영하세요. 양 어깨와 양 손목이 한 화면에 들어와야 가동범위, 좌우 비대칭, 상체 흔들림을 판정할 수 있습니다.",
    "스모 데드리프트": "측면에서 촬영하세요. 어깨·골반·무릎이 한 라인에 보여야 척추 중립, 락아웃, 바 경로를 판정할 수 있습니다.",
    "프론트 레이즈": "측면에서 촬영하세요. 어깨·팔꿈치·손목과 골반이 한 화면에 들어와야 정점 높이, 상체 젖힘, 반동을 판정할 수 있습니다.",
    "클로즈 그립 벤치프레스": "정면에서 촬영하세요. 양 어깨·팔꿈치·손목이 한 화면에 들어와야 그립 너비, 팔꿈치 벌어짐, 좌우 균형을 판정할 수 있습니다.",
    "라잉 트라이셉스 익스텐션": "측면에서 촬영하세요. 누운 자세에서 한쪽 어깨·팔꿈치·손목이 보여야 상완 흔들림과 가동범위를 판정할 수 있습니다.",
    "크런치": "측면에서 촬영하세요. 어깨·골반·무릎이 한 화면에 들어와야 상체 들어올림 정도와 허리 부착 여부를 판정할 수 있습니다.",
    "레그 레이즈": "측면에서 촬영하세요. 골반·무릎·발목이 한 화면에 들어와야 다리 각도, 허리 부착, 무릎 굽힘 정도를 판정할 수 있습니다.",
}

GUIDE_IMAGES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resources", "guide_images")
GUIDE_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp")


def _find_guide_image(exercise_name: str):
    for ext in GUIDE_IMAGE_EXTS:
        candidate = os.path.join(GUIDE_IMAGES_DIR, f"{exercise_name}{ext}")
        if os.path.exists(candidate):
            return candidate
    return None


FEEDBACK_MESSAGES = {
    "excessive_arch": "허리가 과도한 아치 자세입니다. 허리를 너무 아치 모양으로 만들지 말고 가슴을 피려고 노력하세요. 골반을 조금 더 들어올리고 복부를 긴장시켜 허리를 평평하게 유지하세요.",
    "arms_spread": "바를 너무 넓게 잡은 자세입니다. 어깨 너비보다 약간만 넓게 잡는 것이 좋습니다.",
    "arms_narrow": "바를 너무 좁게 잡은 자세입니다. 어깨 너비보다 조금 넓게 잡는 것이 좋습니다.",
    "spine_neutral": "척추가 중립이 아닌 자세입니다. 척추가 과도하게 굽지 않도록 가슴을 들어올리고 어깨를 뒤로 넣으세요.",
    "caved_in_knees": "무릎이 움푹 들어간 자세입니다. 엉덩이를 뒤로 빼서 무릎과 발끝을 일직선으로 유지하세요.",
    "feet_spread": "발을 너무 넓게 벌린 자세입니다. 발을 어깨 너비 정도로만 벌리도록 좁히세요.",
    "lean_back": "상체를 너무 뒤로 젖혔습니다. 반동을 쓰지 말고 코어를 잡은 채로 상체를 세워 컬을 수행하세요.",
    "loose_upper_arm": "팔꿈치가 흔들립니다. 상완(어깨~팔꿈치)을 몸통 옆에 고정한 상태로 들어올리세요.",
    "weak_peak_contraction": "끝까지 짜내지 못했습니다. 정점에서 이두를 한 박자 더 강하게 수축시키세요.",
    "plank_high_back": "엉덩이가 너무 높습니다. 골반을 낮춰 머리부터 발끝까지 일직선을 만드세요.",
    "plank_low_back": "허리가 처졌습니다. 코어로 골반을 살짝 들어올려 일직선을 유지하세요.",
    "knee_over_toe": "무릎이 발끝보다 앞으로 나왔습니다. 엉덩이를 뒤로 빼고 정강이를 수직에 가깝게 두세요.",
    "ohp_uneven_press": "양쪽 손목 높이 차이가 큽니다. 좌우 균형을 맞춰 동시에 밀어 올리세요.",
    "ohp_incomplete_lockout": "정점에서 팔을 끝까지 펴지 못했습니다. 머리 위에서 팔꿈치를 완전히 잠그세요.",
    "row_round_back": "등이 굽었습니다. 가슴을 들고 어깨를 뒤로 빼서 척추 중립을 유지하세요.",
    "row_too_upright": "상체를 너무 세웠습니다. 상체를 약 45° 정도 앞으로 숙여 힙 힌지 자세를 유지하세요.",
    "row_body_english": "반동을 너무 사용했습니다. 상체 각도를 고정한 채 팔로만 당기세요.",
    "incline_partial_rom": "바를 가슴까지 충분히 내리지 못했습니다. 가동범위를 끝까지 활용하세요.",
    "incline_uneven_press": "좌우 손목 높이 차이가 큽니다. 양손이 동시에 정점에 닿도록 밀어 올리세요.",
    "lat_excessive_lean": "상체를 너무 뒤로 젖혀서 등 운동이 아닌 로우 동작이 됐습니다. 상체 기울기를 10~15° 정도로만 유지하세요.",
    "lat_partial_rom": "바를 가슴까지 충분히 당기지 못했습니다. 광배근에 자극이 잘 가도록 가슴 윗부분까지 끌어내리세요.",
    "lat_uneven_pull": "양쪽 손목 높이 차이가 큽니다. 좌우 균형을 맞춰 동시에 당기세요.",
    "lat_body_english": "반동을 너무 사용했습니다. 상체 각도를 고정한 채 등 근육으로만 당기세요.",
    "slr_too_high": "팔을 어깨선 위로 너무 들어올렸습니다. 어깨 임핀지먼트 위험이 있으니 수평선 정도까지만 들어올리세요.",
    "slr_partial_rom": "팔을 수평까지 충분히 들어올리지 못했습니다. 어깨 측면이 자극되도록 수평까지 들어올리세요.",
    "slr_uneven": "양팔 높이 차이가 큽니다. 좌우 동시에 같은 높이로 들어올리세요.",
    "slr_body_english": "반동을 너무 사용했습니다. 상체를 고정한 채 어깨 측면 근육으로만 들어올리세요.",
    "fly_too_open": "팔을 너무 뒤로 펼쳤습니다. 어깨 후방 부상 위험이 있으니 어깨 라인 정도까지만 펼치세요.",
    "fly_partial_rom": "양손이 가슴 앞 가운데까지 충분히 모이지 않았습니다. 정점에서 가슴을 짜내듯 양손을 모으세요.",
    "fly_uneven": "좌우 손목 높이 차이가 큽니다. 양손이 동시에 같은 높이로 움직이도록 균형을 맞추세요.",
    "fly_body_english": "상체를 너무 흔들었습니다. 등을 시트에 붙이고 가슴 근육만으로 모으세요.",
    "sumo_round_back": "등이 굽었습니다. 가슴을 들고 시선을 정면에 두어 척추 중립을 유지하세요.",
    "sumo_partial_lockout": "정점에서 완전히 일어서지 못했습니다. 골반을 앞으로 밀어 상체를 수직으로 세워 락아웃하세요.",
    "sumo_bar_drift": "바가 몸에서 떨어졌습니다. 정강이·허벅지에 바를 붙인 채 직선 경로로 들어올리세요.",
    "sumo_hips_first": "엉덩이가 어깨보다 먼저 올라갔습니다(굿모닝 형태). 가슴을 든 채 어깨와 엉덩이가 같은 비율로 함께 올라오게 하세요.",
    "fr_too_high": "팔을 어깨선 위로 너무 들어올렸습니다. 어깨 임핀지먼트 위험이 있으니 어깨 라인 정도까지만 들어올리세요.",
    "fr_partial_rom": "팔을 어깨 높이까지 충분히 들어올리지 못했습니다. 정점에서 손목이 어깨 높이에 오도록 들어올리세요.",
    "fr_lean_back": "상체를 너무 뒤로 젖혔습니다. 코어를 잡고 상체를 수직으로 유지한 채 팔만 들어올리세요.",
    "fr_body_english": "반동을 너무 사용했습니다. 상체를 고정한 채 어깨 전면 근육으로만 들어올리세요.",
    "cgbp_grip_too_wide": "클로즈그립인데 너무 넓게 잡았습니다. 어깨 너비 또는 약간 좁게 잡아 삼두 자극을 살리세요.",
    "cgbp_partial_rom": "바를 가슴까지 충분히 내리지 못했습니다. 가동범위를 끝까지 활용하세요.",
    "cgbp_uneven_press": "좌우 손목 높이 차이가 큽니다. 양손이 동시에 정점에 닿도록 균형을 맞추세요.",
    "cgbp_elbow_flare": "팔꿈치가 옆으로 너무 벌어졌습니다. 팔꿈치를 몸통에 가깝게 붙여 삼두에 집중하세요.",
    "sc_upper_arm_drift": "상완이 흔들렸습니다. 어깨·팔꿈치 위치를 고정한 채 팔꿈치만 굽혔다 펴세요.",
    "sc_partial_rom": "바를 이마 근처까지 충분히 내리지 못했습니다. 팔꿈치를 더 깊이 굽혀 삼두를 늘리세요.",
    "sc_incomplete_lockout": "정점에서 팔을 끝까지 펴지 못했습니다. 팔꿈치를 완전히 잠가 삼두를 짜내세요.",
    "crunch_too_high": "상체를 너무 높이 들어올렸습니다. 어깨가 바닥에서 살짝 떨어질 정도로만 윗배에 집중해 들어올리세요.",
    "crunch_partial_rom": "상체를 충분히 들어올리지 못했습니다. 윗배가 수축되도록 어깨를 바닥에서 분명히 떼세요.",
    "crunch_lower_back_lift": "허리가 바닥에서 들렸습니다. 허리를 바닥에 붙인 채 윗배 힘만으로 상체를 들어올리세요.",
    "legraise_lower_back_lift": "허리가 바닥에서 들렸습니다. 디스크 부상 위험이 있으니 허리를 바닥에 단단히 붙이고 다리만 움직이세요.",
    "legraise_partial_rom": "다리를 충분히 높이 들어올리지 못했습니다. 다리가 수직에 가깝게 올 때까지 들어올리세요.",
    "legraise_partial_descent": "다리를 끝까지 내리지 않았습니다. 매 rep마다 다리를 거의 바닥 높이까지 내려야 가동범위가 살아납니다.",
    "legraise_excessive_knee_flex": "무릎을 너무 굽혔습니다. 무릎을 거의 펴거나 살짝만 굽힌 채로 다리를 들어올리세요.",
}

ERROR_KEYS = list(FEEDBACK_MESSAGES.keys())

ERROR_CATEGORY_MAP = {
    "excessive_arch": "Posture",
    "spine_neutral": "Posture",
    "arms_spread": "Movement Quality",
    "arms_narrow": "Movement Quality",
    "caved_in_knees": "Stability",
    "feet_spread": "Stability",
    "lean_back": "Posture",
    "loose_upper_arm": "Stability",
    "weak_peak_contraction": "ROM",
    "plank_high_back": "Posture",
    "plank_low_back": "Posture",
    "knee_over_toe": "Stability",
    "ohp_uneven_press": "Stability",
    "ohp_incomplete_lockout": "ROM",
    "row_round_back": "Posture",
    "row_too_upright": "Posture",
    "row_body_english": "Movement Quality",
    "incline_partial_rom": "ROM",
    "incline_uneven_press": "Stability",
    "lat_excessive_lean": "Posture",
    "lat_partial_rom": "ROM",
    "lat_uneven_pull": "Stability",
    "lat_body_english": "Movement Quality",
    "slr_too_high": "ROM",
    "slr_partial_rom": "ROM",
    "slr_uneven": "Stability",
    "slr_body_english": "Movement Quality",
    "fly_too_open": "ROM",
    "fly_partial_rom": "ROM",
    "fly_uneven": "Stability",
    "fly_body_english": "Movement Quality",
    "sumo_round_back": "Posture",
    "sumo_partial_lockout": "ROM",
    "sumo_bar_drift": "Movement Quality",
    "sumo_hips_first": "Movement Quality",
    "fr_too_high": "ROM",
    "fr_partial_rom": "ROM",
    "fr_lean_back": "Posture",
    "fr_body_english": "Movement Quality",
    "cgbp_grip_too_wide": "Movement Quality",
    "cgbp_partial_rom": "ROM",
    "cgbp_uneven_press": "Stability",
    "cgbp_elbow_flare": "Stability",
    "sc_upper_arm_drift": "Stability",
    "sc_partial_rom": "ROM",
    "sc_incomplete_lockout": "ROM",
    "crunch_too_high": "ROM",
    "crunch_partial_rom": "ROM",
    "crunch_lower_back_lift": "Posture",
    "legraise_lower_back_lift": "Posture",
    "legraise_partial_rom": "ROM",
    "legraise_partial_descent": "ROM",
    "legraise_excessive_knee_flex": "Movement Quality",
}

CATEGORY_ORDER = ["Stability", "ROM", "Movement Quality", "Posture", "Core"]

CATEGORY_LABELS = {
    "Stability": "Stability(안정성)",
    "ROM": "Range of Motion(가동범위)",
    "Movement Quality": "Movement Quality(동작 품질)",
    "Posture": "Posture(자세)",
    "Core": "Bracing & Core(코어 긴장)",
}

CATEGORY_PRAISE = {
    "Stability": "균형감 좋습니다.",
    "ROM": "깊이 충분히 내려갑니다. 좋아요.",
    "Movement Quality": "동작 컨트롤이 매끄럽습니다.",
    "Posture": "척추 중립이 잘 유지됩니다.",
    "Core": "복압 거의 완벽합니다.",
}

OVERLAY_MESSAGES = {
    "excessive_arch": "허리 아치 과도",
    "arms_spread": "그립 너무 넓음",
    "arms_narrow": "그립 너무 좁음",
    "spine_neutral": "척추 비중립",
    "caved_in_knees": "무릎 안쪽 꺾임",
    "feet_spread": "보폭이 너무 넓습니다",
    "lean_back": "상체 뒤로 젖힘",
    "loose_upper_arm": "팔꿈치 흔들림",
    "weak_peak_contraction": "수축 부족",
    "plank_high_back": "엉덩이 너무 높음",
    "plank_low_back": "허리 처짐",
    "knee_over_toe": "무릎이 발끝 초과",
    "ohp_uneven_press": "좌우 비대칭",
    "ohp_incomplete_lockout": "락아웃 부족",
    "row_round_back": "등이 굽음",
    "row_too_upright": "상체 너무 세움",
    "row_body_english": "반동 사용",
    "incline_partial_rom": "가동범위 부족",
    "incline_uneven_press": "좌우 비대칭",
    "lat_excessive_lean": "상체 너무 젖힘",
    "lat_partial_rom": "가동범위 부족",
    "lat_uneven_pull": "좌우 비대칭",
    "lat_body_english": "반동 사용",
    "slr_too_high": "어깨선 초과",
    "slr_partial_rom": "수평 미도달",
    "slr_uneven": "좌우 비대칭",
    "slr_body_english": "반동 사용",
    "fly_too_open": "과도한 후방 신전",
    "fly_partial_rom": "수축 부족",
    "fly_uneven": "좌우 비대칭",
    "fly_body_english": "상체 흔들림",
    "sumo_round_back": "등 굽음",
    "sumo_partial_lockout": "락아웃 부족",
    "sumo_bar_drift": "바 경로 이탈",
    "sumo_hips_first": "엉덩이 먼저 올라감",
    "fr_too_high": "어깨선 초과",
    "fr_partial_rom": "수평 미도달",
    "fr_lean_back": "상체 뒤로 젖힘",
    "fr_body_english": "반동 사용",
    "cgbp_grip_too_wide": "그립 너무 넓음",
    "cgbp_partial_rom": "가동범위 부족",
    "cgbp_uneven_press": "좌우 비대칭",
    "cgbp_elbow_flare": "팔꿈치 벌어짐",
    "sc_upper_arm_drift": "상완 흔들림",
    "sc_partial_rom": "내림 부족",
    "sc_incomplete_lockout": "락아웃 부족",
    "crunch_too_high": "윗몸일으키기 형태",
    "crunch_partial_rom": "수축 부족",
    "crunch_lower_back_lift": "허리 들림",
    "legraise_lower_back_lift": "허리 들림",
    "legraise_partial_rom": "다리 미도달",
    "legraise_partial_descent": "끝까지 안 내림",
    "legraise_excessive_knee_flex": "무릎 굽힘 과도",
}

ERROR_BODY_PARTS = {
    "feet_spread": [27, 28],
    "caved_in_knees": [25, 26],
    "arms_spread": [15, 16],
    "arms_narrow": [15, 16],
    "excessive_arch": [23, 24],
    "spine_neutral": [11, 12, 23, 24],
    "lean_back": [11, 12, 23, 24],
    "loose_upper_arm": [13, 14],
    "weak_peak_contraction": [13, 14, 15, 16],
    "plank_high_back": [11, 12, 23, 24],
    "plank_low_back": [11, 12, 23, 24],
    "knee_over_toe": [25, 26, 27, 28],
    "ohp_uneven_press": [15, 16],
    "ohp_incomplete_lockout": [13, 14, 15, 16],
    "row_round_back": [11, 12, 23, 24],
    "row_too_upright": [11, 12, 23, 24],
    "row_body_english": [11, 12, 23, 24],
    "incline_partial_rom": [13, 14],
    "incline_uneven_press": [15, 16],
    "lat_excessive_lean": [11, 12, 23, 24],
    "lat_partial_rom": [15, 16],
    "lat_uneven_pull": [15, 16],
    "lat_body_english": [11, 12, 23, 24],
    "slr_too_high": [13, 14],
    "slr_partial_rom": [13, 14],
    "slr_uneven": [13, 14],
    "slr_body_english": [11, 12, 23, 24],
    "fly_too_open": [15, 16],
    "fly_partial_rom": [15, 16],
    "fly_uneven": [15, 16],
    "fly_body_english": [11, 12, 23, 24],
    "sumo_round_back": [0, 11, 12, 23, 24],
    "sumo_partial_lockout": [11, 12, 23, 24],
    "sumo_bar_drift": [15, 16, 11, 12],
    "sumo_hips_first": [11, 12, 23, 24],
    "fr_too_high": [13, 14, 15, 16],
    "fr_partial_rom": [13, 14, 15, 16],
    "fr_lean_back": [11, 12, 23, 24],
    "fr_body_english": [11, 12, 23, 24],
    "cgbp_grip_too_wide": [15, 16],
    "cgbp_partial_rom": [13, 14],
    "cgbp_uneven_press": [15, 16],
    "cgbp_elbow_flare": [13, 14],
    "sc_upper_arm_drift": [11, 12, 13, 14],
    "sc_partial_rom": [15, 16],
    "sc_incomplete_lockout": [13, 14, 15, 16],
    "crunch_too_high": [11, 12, 23, 24],
    "crunch_partial_rom": [11, 12],
    "crunch_lower_back_lift": [23, 24],
    "legraise_lower_back_lift": [23, 24],
    "legraise_partial_rom": [25, 26, 27, 28],
    "legraise_partial_descent": [25, 26, 27, 28],
    "legraise_excessive_knee_flex": [25, 26],
}

CATEGORY_HINTS = {
    "Stability": "발 가운데로 무게중심을 두고 좌우 흔들림을 줄여보세요.",
    "ROM": "rep 사이 깊이가 일정하지 않습니다. 매번 같은 깊이까지 컨트롤하면서 내려가세요.",
    "Movement Quality": "하강·상승 템포를 일정하게(예: 3초 내려가고 1초 올라오기) 유지하세요.",
    "Posture": "가슴을 천장 쪽으로 들고 시선을 정면 한 점에 고정해 척추 중립을 유지하세요.",
    "Core": "복압을 더 강하게 잡고 호흡 타이밍을 의식적으로 맞춰보세요.",
}


@st.cache_resource(show_spinner="YOLOv5 모델을 로딩하는 중입니다...")
def load_yolo_model():
    original_load = torch.load

    def _unsafe_load(*args, **kwargs):
        kwargs["weights_only"] = False
        return original_load(*args, **kwargs)

    torch.load = _unsafe_load
    try:
        model = torch.hub.load(
            "ultralytics/yolov5:v7.0",
            "custom",
            path=YOLO_WEIGHTS_PATH,
            force_reload=False,
            trust_repo=True,
        )
    finally:
        torch.load = original_load

    model.to("cpu")
    model.eval()
    model.conf = 0.5
    return model


@st.cache_resource(show_spinner=False)
def _load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_exercise_bundle(exercise_name):
    """파이프라인 타입에 맞는 모델·스케일러 묶음을 반환.

    rf_yolo: {"type", "model"}
    knn_full / lr_full: {"type", "model", "scaler"}
    svc_lr_full: {"type", "stage_model", "err_model", "scaler"}
    rule_based: {"type"} (모델 없음)
    """
    cfg = EXERCISE_PIPELINES[exercise_name]
    ptype = cfg["type"]
    if ptype == "rf_yolo":
        return {"type": ptype, "model": _load_pickle(cfg["model_path"])}
    if ptype in ("knn_full", "lr_full"):
        return {
            "type": ptype,
            "model": _load_pickle(cfg["model_path"]),
            "scaler": _load_pickle(cfg["scaler_path"]),
        }
    if ptype == "svc_lr_full":
        return {
            "type": ptype,
            "stage_model": _load_pickle(cfg["stage_model_path"]),
            "err_model": _load_pickle(cfg["err_model_path"]),
            "scaler": _load_pickle(cfg["scaler_path"]),
        }
    if ptype == "rule_based":
        return {"type": ptype}
    raise ValueError(f"Unknown pipeline type: {ptype}")


def extract_landmark_row(pose_landmarks):
    return [
        coord
        for lm in pose_landmarks.landmark
        for coord in [lm.x, lm.y, lm.z, lm.visibility]
    ]


def classify_posture(class_name):
    name = str(class_name).lower()
    is_correct = "correct" in name
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
    return is_correct, stage, error_key


def _load_korean_font(size):
    candidates = [
        "C:/Windows/Fonts/malgun.ttf",
        "C:/Windows/Fonts/malgunbd.ttf",
        "C:/Windows/Fonts/gulim.ttc",
        "C:/Windows/Fonts/batang.ttc",
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return ImageFont.load_default()


def _landmark_to_pixel(landmarks, bbox, idx):
    x1, y1, x2, y2 = bbox
    if idx >= len(landmarks):
        return None
    lx, ly, _, vis = landmarks[idx]
    if vis < 0.4:
        return None
    px = int(x1 + lx * (x2 - x1))
    py = int(y1 + ly * (y2 - y1))
    return px, py


def annotate_video_with_errors(
    input_path,
    output_path,
    event_groups,
    significant_keys,
    frame_skip,
    landmarks_by_frame=None,
    progress_cb=None,
):
    intervals = []
    pad_frames = max(int(frame_skip) - 1, 0)
    for key, evs_list in event_groups.items():
        if not evs_list:
            continue
        first_ev = min(evs_list, key=lambda ev: int(ev["start_frame"]))
        s = int(first_ev["start_frame"])
        e = int(first_ev["end_frame"]) + pad_frames
        intervals.append((s, e, key, key in significant_keys))
    intervals.sort()

    landmarks_by_frame = landmarks_by_frame or {}
    sorted_lm_frames = sorted(landmarks_by_frame.keys())

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        return False

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1

    min_visible_frames = max(1, int(round(0.5 * fps)))
    intervals = [
        (s, max(e, s + min_visible_frames - 1), k, sig)
        for (s, e, k, sig) in intervals
    ]
    intervals.sort()

    playback_speed = 0.6
    output_fps = max(1.0, fps * playback_speed)

    try:
        writer = imageio.get_writer(
            output_path,
            format="FFMPEG",
            mode="I",
            fps=output_fps,
            codec="libx264",
            pixelformat="yuv420p",
            macro_block_size=2,
            quality=7,
        )
    except Exception:
        cap.release()
        return False

    font_size = max(22, height // 22)
    font = _load_korean_font(font_size)
    line_height = font_size + 8
    pad = 12
    circle_radius = max(24, min(width, height) // 18)

    import bisect

    def find_nearest_landmark_frame(idx):
        if not sorted_lm_frames:
            return None
        pos = bisect.bisect_right(sorted_lm_frames, idx)
        if pos == 0:
            return sorted_lm_frames[0]
        if pos >= len(sorted_lm_frames):
            return sorted_lm_frames[-1]
        before = sorted_lm_frames[pos - 1]
        after = sorted_lm_frames[pos]
        return before if (idx - before) <= (after - idx) else after

    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            active = []
            for s, e, k, sig in intervals:
                if s <= frame_idx <= e and not any(k == ak for ak, _ in active):
                    active.append((k, sig))
                    if len(active) >= 3:
                        break

            if active:
                img = Image.fromarray(rgb)
                draw = ImageDraw.Draw(img, "RGBA")

                lm_frame = find_nearest_landmark_frame(frame_idx)
                lm_data = landmarks_by_frame.get(lm_frame) if lm_frame is not None else None
                landmarks = lm_data["landmarks"] if lm_data else None
                bbox = lm_data["crop_bbox"] if lm_data else None

                for key, is_sig in active:
                    color_main = (255, 60, 60, 255) if is_sig else (255, 165, 60, 255)
                    color_dim = (255, 60, 60, 90) if is_sig else (255, 165, 60, 90)

                    body_idx_list = ERROR_BODY_PARTS.get(key, [])
                    points = []
                    if landmarks is not None and bbox is not None:
                        for li in body_idx_list:
                            pt = _landmark_to_pixel(landmarks, bbox, li)
                            if pt is not None:
                                points.append(pt)

                    if not points:
                        continue

                    if key == "spine_neutral" and len(points) >= 4:
                        sx = (points[0][0] + points[1][0]) // 2
                        sy = (points[0][1] + points[1][1]) // 2
                        hx = (points[2][0] + points[3][0]) // 2
                        hy = (points[2][1] + points[3][1]) // 2
                        draw.line(
                            [(sx, sy), (hx, hy)],
                            fill=color_main,
                            width=max(4, height // 140),
                        )
                        for px, py in [(sx, sy), (hx, hy)]:
                            draw.ellipse(
                                [
                                    px - circle_radius // 2,
                                    py - circle_radius // 2,
                                    px + circle_radius // 2,
                                    py + circle_radius // 2,
                                ],
                                outline=color_main,
                                width=4,
                            )
                    else:
                        for px, py in points:
                            draw.ellipse(
                                [
                                    px - circle_radius,
                                    py - circle_radius,
                                    px + circle_radius,
                                    py + circle_radius,
                                ],
                                outline=color_main,
                                width=4,
                                fill=color_dim,
                            )

                lines = [OVERLAY_MESSAGES.get(k, k) for k, _ in active]
                widths = []
                for line in lines:
                    bb = draw.textbbox((0, 0), line, font=font)
                    widths.append(bb[2] - bb[0])
                box_w = max(widths) + pad * 2
                box_h = line_height * len(lines) + pad

                x0 = max((width - box_w) // 2, 8)
                y0 = 16
                draw.rectangle(
                    [x0, y0, x0 + box_w, y0 + box_h],
                    fill=(0, 0, 0, 200),
                )
                ty = y0 + pad // 2
                for line in lines:
                    draw.text(
                        (x0 + pad + 1, ty + 1),
                        line,
                        font=font,
                        fill=(0, 0, 0, 255),
                    )
                    draw.text(
                        (x0 + pad, ty),
                        line,
                        font=font,
                        fill=(255, 220, 80, 255),
                    )
                    ty += line_height

                rgb = np.array(img)

            writer.append_data(rgb)
            frame_idx += 1
            if progress_cb and total_frames > 0:
                progress_cb(frame_idx, total_frames)
    finally:
        cap.release()
        try:
            writer.close()
        except Exception:
            pass

    return os.path.exists(output_path) and os.path.getsize(output_path) > 0


def compute_score_from_events(event_groups, penalty_per_type, min_duration_sec, result=None):
    # 플랭크는 정적 운동이라 정자세 프레임 비율로 점수 산정
    if result is not None and result.get("static"):
        analyzed = max(int(result.get("analyzed_frames", 0)), 1)
        wrong_frames = sum(result.get("error_counter", {}).values())
        correct_frames = max(analyzed - wrong_frames, 0)
        score = (correct_frames / analyzed) * 100.0
        significant = list(event_groups.keys())
        return score, significant, []

    significant = []
    filtered = []
    for key, evs in event_groups.items():
        max_dur = max((ev["duration_sec"] for ev in evs), default=0.0)
        if max_dur >= min_duration_sec:
            significant.append(key)
        else:
            filtered.append(key)
    score = max(0.0, 100.0 - len(significant) * penalty_per_type)
    return score, significant, filtered


def compute_category_scores(event_groups, significant, result, penalty_per_type):
    scores = {c: 100.0 for c in CATEGORY_ORDER}
    for key in significant:
        cat = ERROR_CATEGORY_MAP.get(key)
        if cat in scores:
            scores[cat] -= penalty_per_type

    total_frames = max(int(result.get("total_frames", 1)), 1)
    analyzed = int(result.get("analyzed_frames", 0))
    rep_count = int(result.get("rep_count", 0))
    coverage = min(analyzed / total_frames, 1.0)

    rom_base = 60.0 + 40.0 * coverage
    if rep_count == 0:
        rom_base -= 30.0
    elif rep_count < 3:
        rom_base -= 10.0
    scores["ROM"] = rom_base

    correct_prob_sum = float(result.get("correct_prob_sum", 0.0))
    if analyzed > 0:
        avg_correct = correct_prob_sum / analyzed
        core_base = 50.0 + 50.0 * avg_correct
    else:
        core_base = 50.0
    scores["Core"] = core_base

    return {k: max(0.0, min(100.0, v)) for k, v in scores.items()}


def estimate_top_percent(total_score):
    if total_score >= 95:
        return 5
    if total_score >= 90:
        return 10
    if total_score >= 85:
        return 15
    if total_score >= 80:
        return 22
    if total_score >= 75:
        return 28
    if total_score >= 70:
        return 35
    if total_score >= 60:
        return 45
    if total_score >= 50:
        return 55
    return 65


def build_overall_review(exercise_name, total_score, rep_count, category_scores, significant, event_groups):
    top_pct = estimate_top_percent(total_score)
    best_cat = max(category_scores, key=category_scores.get)
    worst_cat = min(category_scores, key=category_scores.get)

    rep_phrase = (
        f"{rep_count} rep을 끝까지 마무리한 점"
        if rep_count > 0
        else "끝까지 자세를 무너뜨리지 않으려는 의지"
    )
    s1 = (
        f"{rep_phrase}, 그리고 {CATEGORY_LABELS[best_cat]}는 거의 흠잡을 데 없이 "
        f"{category_scores[best_cat]:.0f}점 나왔어요."
    )
    s2 = "호흡과 컨트롤 감각은 이미 상위권입니다."

    if significant:
        sorted_keys = sorted(
            significant,
            key=lambda k: sum(ev["duration_sec"] for ev in event_groups.get(k, [])),
            reverse=True,
        )
        primary = sorted_keys[0]
        primary_evs = event_groups.get(primary, [])
        primary_total = sum(ev["duration_sec"] for ev in primary_evs)
        s3 = (
            f"다만 하강/유지 구간에서 '{primary}' 패턴이 {len(primary_evs)}회, "
            f"총 {primary_total:.1f}초간 잡혔습니다."
        )
        s4 = (
            f"무게 욕심보다 {CATEGORY_LABELS[worst_cat]} 한 가지만 잡으면 "
            f"점수가 빠르게 올라갑니다."
        )
    else:
        s3 = "유의미한 자세 오류는 잡히지 않았습니다."
        s4 = "지금 폼을 유지하면서 점진적으로 무게를 올려도 좋습니다."

    return " ".join([s1, s2, s3, s4]), top_pct


RADAR_AXIS_LABELS = {
    "Stability": "Stability",
    "ROM": "Range of Motion",
    "Movement Quality": "Movement Quality",
    "Posture": "Posture",
    "Core": "Bracing and Core",
}


def render_radar_chart(scores, total_score):
    axis_keys = CATEGORY_ORDER
    labels = [RADAR_AXIS_LABELS[k] for k in axis_keys]
    values = [float(scores.get(k, 0.0)) for k in axis_keys]

    theta = labels + [labels[0]]
    r = values + [values[0]]

    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=r,
            theta=theta,
            fill="toself",
            line=dict(color="#1f77b4", width=2),
            fillcolor="rgba(31, 119, 180, 0.35)",
            marker=dict(size=8, color="#1f77b4"),
            hovertemplate="%{theta}: %{r:.0f}<extra></extra>",
            name="점수",
        )
    )

    fig.update_layout(
        polar=dict(
            bgcolor="#fafafa",
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickvals=[20, 40, 60, 80, 100],
                tickfont=dict(size=10, color="#888"),
                gridcolor="#dcdcdc",
                angle=90,
                tickangle=90,
            ),
            angularaxis=dict(
                tickfont=dict(size=12, color="#222"),
                gridcolor="#dcdcdc",
                linecolor="#bbbbbb",
            ),
        ),
        showlegend=False,
        margin=dict(l=40, r=40, t=30, b=30),
        height=420,
        autosize=True,
        paper_bgcolor="white",
    )

    st.plotly_chart(fig, use_container_width=True)
    st.markdown(
        f"""
        <div style="text-align:center; margin-top:-12px;">
            <span style="font-size:46px; font-weight:800; color:#1f77b4;">{total_score:.0f}</span>
            <span style="font-size:24px; font-weight:600; color:#666;"> / 100</span>
            <div style="font-size:13px; color:#888; margin-top:4px;">총점</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_gymscore_feedback(result, exercise_name, score, significant, event_groups, penalty_per_type):
    cat_scores = compute_category_scores(event_groups, significant, result, penalty_per_type)
    overall, top_pct = build_overall_review(
        exercise_name,
        score,
        result.get("rep_count", 0),
        cat_scores,
        significant,
        event_groups,
    )

    st.markdown("---")
    st.markdown(f"### Top {top_pct}% 리프터")

    render_radar_chart(cat_scores, score)

    st.markdown("#### 1. 총평")
    st.write(overall)

    st.markdown("#### 2. 카테고리별 한 줄 피드백")
    cat_to_errors = {}
    for key in event_groups.keys():
        cat = ERROR_CATEGORY_MAP.get(key)
        if cat is not None:
            cat_to_errors.setdefault(cat, []).append(key)

    for cat in CATEGORY_ORDER:
        sc = cat_scores[cat]
        label = CATEGORY_LABELS[cat]
        if sc >= 90:
            line = f"None. {CATEGORY_PRAISE[cat]}"
        else:
            err_keys = cat_to_errors.get(cat, [])
            if err_keys:
                err_keys_sorted = sorted(
                    err_keys,
                    key=lambda k: sum(ev["duration_sec"] for ev in event_groups.get(k, [])),
                    reverse=True,
                )
                line = FEEDBACK_MESSAGES.get(err_keys_sorted[0], err_keys_sorted[0])
            else:
                line = CATEGORY_HINTS[cat]
        st.markdown(f"- **{label} ({sc:.0f})**: {line}")

    st.markdown("#### 3. 상위 % 추정")
    next_tier_msg = ""
    if top_pct > 10:
        next_tier_msg = " 약점 카테고리만 교정하면 상위권 진입 가능합니다."
    st.success(f"**{score:.0f}점 / 100 — Top {top_pct}% of lifters.**{next_tier_msg}")


MP_LM_INDEX = {
    "NOSE": 0,
    "LEFT_EYE_INNER": 1, "LEFT_EYE": 2, "LEFT_EYE_OUTER": 3,
    "RIGHT_EYE_INNER": 4, "RIGHT_EYE": 5, "RIGHT_EYE_OUTER": 6,
    "LEFT_EAR": 7, "RIGHT_EAR": 8,
    "MOUTH_LEFT": 9, "MOUTH_RIGHT": 10,
    "LEFT_SHOULDER": 11, "RIGHT_SHOULDER": 12,
    "LEFT_ELBOW": 13, "RIGHT_ELBOW": 14,
    "LEFT_WRIST": 15, "RIGHT_WRIST": 16,
    "LEFT_PINKY": 17, "RIGHT_PINKY": 18,
    "LEFT_INDEX": 19, "RIGHT_INDEX": 20,
    "LEFT_THUMB": 21, "RIGHT_THUMB": 22,
    "LEFT_HIP": 23, "RIGHT_HIP": 24,
    "LEFT_KNEE": 25, "RIGHT_KNEE": 26,
    "LEFT_ANKLE": 27, "RIGHT_ANKLE": 28,
    "LEFT_HEEL": 29, "RIGHT_HEEL": 30,
    "LEFT_FOOT_INDEX": 31, "RIGHT_FOOT_INDEX": 32,
}


def _calc_angle(p1, p2, p3):
    a = np.array(p1, dtype=float)
    b = np.array(p2, dtype=float)
    c = np.array(p3, dtype=float)
    rad = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    deg = abs(rad * 180.0 / np.pi)
    return deg if deg <= 180 else 360 - deg


def _build_feature_columns(important_lms):
    cols = []
    for lm in important_lms:
        cols += [f"{lm.lower()}_x", f"{lm.lower()}_y", f"{lm.lower()}_z", f"{lm.lower()}_v"]
    return cols


def _extract_subset_row(landmarks_full, important_lms):
    row = []
    for name in important_lms:
        lm = landmarks_full[MP_LM_INDEX[name]]
        row += [lm[0], lm[1], lm[2], lm[3]]
    return row


class BicepArmState:
    def __init__(self, side, cfg):
        self.side = side
        self.stage_down_th = cfg["stage_down_angle"]
        self.stage_up_th = cfg["stage_up_angle"]
        self.peak_th = cfg["peak_contraction_angle"]
        self.loose_arm_th = cfg["loose_upper_arm_angle"]
        self.vis_th = cfg["visibility_threshold"]
        self.counter = 0
        self.stage = "down"
        self.peak_min_angle = 1000.0
        self.loose_active = False

    def update(self, landmarks_full):
        if self.side == "LEFT":
            sh, el, wr = MP_LM_INDEX["LEFT_SHOULDER"], MP_LM_INDEX["LEFT_ELBOW"], MP_LM_INDEX["LEFT_WRIST"]
        else:
            sh, el, wr = MP_LM_INDEX["RIGHT_SHOULDER"], MP_LM_INDEX["RIGHT_ELBOW"], MP_LM_INDEX["RIGHT_WRIST"]
        sh_lm, el_lm, wr_lm = landmarks_full[sh], landmarks_full[el], landmarks_full[wr]

        if min(sh_lm[3], el_lm[3], wr_lm[3]) < self.vis_th:
            return None

        sh_xy = (sh_lm[0], sh_lm[1])
        el_xy = (el_lm[0], el_lm[1])
        wr_xy = (wr_lm[0], wr_lm[1])

        curl_angle = _calc_angle(sh_xy, el_xy, wr_xy)
        upper_arm_angle = _calc_angle(el_xy, sh_xy, (sh_xy[0], 1.0))

        rule_error = None
        if curl_angle > self.stage_down_th:
            if self.stage == "up":
                if self.peak_min_angle != 1000.0 and self.peak_min_angle >= self.peak_th:
                    rule_error = "weak_peak_contraction"
                self.peak_min_angle = 1000.0
            self.stage = "down"
        elif curl_angle < self.stage_up_th and self.stage == "down":
            self.stage = "up"
            self.counter += 1

        if self.stage == "up" and curl_angle < self.peak_min_angle:
            self.peak_min_angle = curl_angle

        if upper_arm_angle > self.loose_arm_th:
            if not self.loose_active:
                self.loose_active = True
                if rule_error is None:
                    rule_error = "loose_upper_arm"
        else:
            self.loose_active = False

        return rule_error


def _torso_angle_from_vertical(shoulder, hip):
    # 골반→어깨 벡터가 수직선과 이루는 각도 (degrees). 누워 있으면 ~90°, 직립이면 ~0°.
    dx = shoulder[0] - hip[0]
    dy = shoulder[1] - hip[1]
    return abs(np.degrees(np.arctan2(dx, -dy)))


class OHPState:
    """오버헤드 프레스 (정면) 룰 기반."""
    DOWN_ELBOW_ANGLE = 100
    UP_ELBOW_ANGLE = 160
    INCOMPLETE_LOCKOUT_ANGLE = 155
    UNEVEN_RATIO = 0.15
    VIS_TH = 0.5

    def __init__(self):
        self.counter = 0
        self.stage = "down"
        self.rep_peak_elbow = 0.0
        self.uneven_active = False

    def update(self, landmarks):
        l_sh, r_sh = landmarks[11], landmarks[12]
        l_el, r_el = landmarks[13], landmarks[14]
        l_wr, r_wr = landmarks[15], landmarks[16]
        if min(l_sh[3], r_sh[3], l_el[3], r_el[3], l_wr[3], r_wr[3]) < self.VIS_TH:
            return []

        l_angle = _calc_angle((l_sh[0], l_sh[1]), (l_el[0], l_el[1]), (l_wr[0], l_wr[1]))
        r_angle = _calc_angle((r_sh[0], r_sh[1]), (r_el[0], r_el[1]), (r_wr[0], r_wr[1]))
        avg_elbow = (l_angle + r_angle) / 2.0

        errors = []

        if self.stage == "down" and avg_elbow > self.UP_ELBOW_ANGLE:
            self.stage = "up"
            self.counter += 1
            self.rep_peak_elbow = avg_elbow
        elif self.stage == "up" and avg_elbow < self.DOWN_ELBOW_ANGLE:
            if self.rep_peak_elbow < self.INCOMPLETE_LOCKOUT_ANGLE:
                errors.append("ohp_incomplete_lockout")
            self.stage = "down"
            self.rep_peak_elbow = 0.0
        elif self.stage == "up":
            if avg_elbow > self.rep_peak_elbow:
                self.rep_peak_elbow = avg_elbow

        if self.stage == "up":
            shoulder_width = abs(l_sh[0] - r_sh[0])
            if shoulder_width > 1e-3:
                wrist_y_diff = abs(l_wr[1] - r_wr[1])
                if (wrist_y_diff / shoulder_width) > self.UNEVEN_RATIO:
                    if not self.uneven_active:
                        self.uneven_active = True
                        errors.append("ohp_uneven_press")
                else:
                    self.uneven_active = False

        return errors


class BarbellRowState:
    """바벨 로우 (측면) 룰 기반. 보이는 쪽 팔 사용."""
    DOWN_ELBOW_ANGLE = 150
    UP_ELBOW_ANGLE = 90
    ROUND_BACK_ANGLE = 160
    UPRIGHT_ANGLE = 30
    BODY_ENGLISH_DELTA = 15.0
    VIS_TH = 0.5

    def __init__(self):
        self.counter = 0
        self.stage = "down"
        self.rep_torso_min = 1000.0
        self.rep_torso_max = -1000.0
        self.round_active = False
        self.upright_active = False

    def update(self, landmarks):
        l_sh, r_sh = landmarks[11], landmarks[12]
        l_el, r_el = landmarks[13], landmarks[14]
        l_wr, r_wr = landmarks[15], landmarks[16]
        l_hip, r_hip = landmarks[23], landmarks[24]
        l_knee, r_knee = landmarks[25], landmarks[26]

        l_vis = min(l_sh[3], l_el[3], l_wr[3], l_hip[3], l_knee[3])
        r_vis = min(r_sh[3], r_el[3], r_wr[3], r_hip[3], r_knee[3])
        if max(l_vis, r_vis) < self.VIS_TH:
            return []
        if l_vis >= r_vis:
            sh, el, wr, hip, knee = l_sh, l_el, l_wr, l_hip, l_knee
        else:
            sh, el, wr, hip, knee = r_sh, r_el, r_wr, r_hip, r_knee

        elbow_angle = _calc_angle((sh[0], sh[1]), (el[0], el[1]), (wr[0], wr[1]))
        spine_angle = _calc_angle((sh[0], sh[1]), (hip[0], hip[1]), (knee[0], knee[1]))
        torso_v = _torso_angle_from_vertical((sh[0], sh[1]), (hip[0], hip[1]))

        errors = []

        if self.stage == "down" and elbow_angle < self.UP_ELBOW_ANGLE:
            self.stage = "up"
            self.counter += 1
        elif self.stage == "up" and elbow_angle > self.DOWN_ELBOW_ANGLE:
            if self.rep_torso_max - self.rep_torso_min > self.BODY_ENGLISH_DELTA:
                errors.append("row_body_english")
            self.stage = "down"
            self.rep_torso_min = 1000.0
            self.rep_torso_max = -1000.0

        if torso_v < self.rep_torso_min:
            self.rep_torso_min = torso_v
        if torso_v > self.rep_torso_max:
            self.rep_torso_max = torso_v

        if spine_angle < self.ROUND_BACK_ANGLE:
            if not self.round_active:
                self.round_active = True
                errors.append("row_round_back")
        else:
            self.round_active = False

        if torso_v < self.UPRIGHT_ANGLE:
            if not self.upright_active:
                self.upright_active = True
                errors.append("row_too_upright")
        else:
            self.upright_active = False

        return errors


class InclineBenchState:
    """인클라인 벤치프레스 (정면) 룰 기반. arms_spread/arms_narrow는 기존 키 재사용."""
    DOWN_ELBOW_ANGLE = 90
    UP_ELBOW_ANGLE = 160
    PARTIAL_ROM_ANGLE = 100
    ARMS_SPREAD_RATIO = 1.5
    ARMS_NARROW_RATIO = 1.0
    UNEVEN_RATIO = 0.15
    VIS_TH = 0.5

    def __init__(self):
        self.counter = 0
        self.stage = "up"
        self.rep_min_elbow = 1000.0
        self.spread_active = False
        self.narrow_active = False
        self.uneven_active = False

    def update(self, landmarks):
        l_sh, r_sh = landmarks[11], landmarks[12]
        l_el, r_el = landmarks[13], landmarks[14]
        l_wr, r_wr = landmarks[15], landmarks[16]
        if min(l_sh[3], r_sh[3], l_el[3], r_el[3], l_wr[3], r_wr[3]) < self.VIS_TH:
            return []

        l_angle = _calc_angle((l_sh[0], l_sh[1]), (l_el[0], l_el[1]), (l_wr[0], l_wr[1]))
        r_angle = _calc_angle((r_sh[0], r_sh[1]), (r_el[0], r_el[1]), (r_wr[0], r_wr[1]))
        avg_elbow = (l_angle + r_angle) / 2.0

        errors = []

        if self.stage == "up" and avg_elbow < self.DOWN_ELBOW_ANGLE:
            self.stage = "down"
            self.rep_min_elbow = avg_elbow
        elif self.stage == "down" and avg_elbow > self.UP_ELBOW_ANGLE:
            if self.rep_min_elbow > self.PARTIAL_ROM_ANGLE:
                errors.append("incline_partial_rom")
            self.stage = "up"
            self.counter += 1
            self.rep_min_elbow = 1000.0
        elif self.stage == "down":
            if avg_elbow < self.rep_min_elbow:
                self.rep_min_elbow = avg_elbow

        shoulder_width = abs(l_sh[0] - r_sh[0])
        wrist_width = abs(l_wr[0] - r_wr[0])
        if shoulder_width > 1e-3:
            ratio = wrist_width / shoulder_width
            if ratio > self.ARMS_SPREAD_RATIO:
                if not self.spread_active:
                    self.spread_active = True
                    errors.append("arms_spread")
                self.narrow_active = False
            elif ratio < self.ARMS_NARROW_RATIO:
                if not self.narrow_active:
                    self.narrow_active = True
                    errors.append("arms_narrow")
                self.spread_active = False
            else:
                self.spread_active = False
                self.narrow_active = False

        if self.stage == "up" and shoulder_width > 1e-3:
            wrist_y_diff = abs(l_wr[1] - r_wr[1])
            if (wrist_y_diff / shoulder_width) > self.UNEVEN_RATIO:
                if not self.uneven_active:
                    self.uneven_active = True
                    errors.append("incline_uneven_press")
            else:
                self.uneven_active = False

        return errors


class LatPulldownState:
    """랫풀다운 (정면) 룰 기반."""
    DOWN_ELBOW_ANGLE = 100
    UP_ELBOW_ANGLE = 160
    EXCESSIVE_LEAN_ANGLE = 25
    PARTIAL_ROM_ANGLE = 110
    UNEVEN_THRESHOLD = 0.15
    BODY_ENGLISH_DELTA = 15.0
    ARMS_SPREAD_RATIO = 1.6
    ARMS_NARROW_RATIO = 1.0
    VIS_TH = 0.5

    def __init__(self):
        self.counter = 0
        self.stage = "up"
        self.rep_min_elbow = 1000.0
        self.rep_torso_min = 1000.0
        self.rep_torso_max = -1000.0
        self.lean_active = False
        self.uneven_active = False
        self.spread_active = False
        self.narrow_active = False

    def update(self, landmarks):
        l_sh, r_sh = landmarks[11], landmarks[12]
        l_el, r_el = landmarks[13], landmarks[14]
        l_wr, r_wr = landmarks[15], landmarks[16]
        l_hip, r_hip = landmarks[23], landmarks[24]
        if min(l_sh[3], r_sh[3], l_el[3], r_el[3], l_wr[3], r_wr[3], l_hip[3], r_hip[3]) < self.VIS_TH:
            return []

        l_angle = _calc_angle((l_sh[0], l_sh[1]), (l_el[0], l_el[1]), (l_wr[0], l_wr[1]))
        r_angle = _calc_angle((r_sh[0], r_sh[1]), (r_el[0], r_el[1]), (r_wr[0], r_wr[1]))
        avg_elbow = (l_angle + r_angle) / 2.0

        sh_mid = ((l_sh[0] + r_sh[0]) / 2.0, (l_sh[1] + r_sh[1]) / 2.0)
        hip_mid = ((l_hip[0] + r_hip[0]) / 2.0, (l_hip[1] + r_hip[1]) / 2.0)
        torso_v = _torso_angle_from_vertical(sh_mid, hip_mid)

        errors = []

        if self.stage == "up" and avg_elbow < self.DOWN_ELBOW_ANGLE:
            self.stage = "down"
            self.counter += 1
            self.rep_min_elbow = avg_elbow
            self.rep_torso_min = torso_v
            self.rep_torso_max = torso_v
        elif self.stage == "down" and avg_elbow > self.UP_ELBOW_ANGLE:
            if self.rep_min_elbow > self.PARTIAL_ROM_ANGLE:
                errors.append("lat_partial_rom")
            if self.rep_torso_max - self.rep_torso_min > self.BODY_ENGLISH_DELTA:
                errors.append("lat_body_english")
            self.stage = "up"
            self.rep_min_elbow = 1000.0
        elif self.stage == "down":
            if avg_elbow < self.rep_min_elbow:
                self.rep_min_elbow = avg_elbow
            if torso_v < self.rep_torso_min:
                self.rep_torso_min = torso_v
            if torso_v > self.rep_torso_max:
                self.rep_torso_max = torso_v

        if self.stage == "down":
            if torso_v > self.EXCESSIVE_LEAN_ANGLE:
                if not self.lean_active:
                    self.lean_active = True
                    errors.append("lat_excessive_lean")
            else:
                self.lean_active = False

        if self.stage == "down":
            shoulder_width_y = abs(l_sh[0] - r_sh[0])
            if shoulder_width_y > 1e-3:
                wrist_y_diff = abs(l_wr[1] - r_wr[1])
                if (wrist_y_diff / shoulder_width_y) > self.UNEVEN_THRESHOLD:
                    if not self.uneven_active:
                        self.uneven_active = True
                        errors.append("lat_uneven_pull")
                else:
                    self.uneven_active = False

        shoulder_width = abs(l_sh[0] - r_sh[0])
        wrist_width = abs(l_wr[0] - r_wr[0])
        if shoulder_width > 1e-3:
            ratio = wrist_width / shoulder_width
            if ratio > self.ARMS_SPREAD_RATIO:
                if not self.spread_active:
                    self.spread_active = True
                    errors.append("arms_spread")
                self.narrow_active = False
            elif ratio < self.ARMS_NARROW_RATIO:
                if not self.narrow_active:
                    self.narrow_active = True
                    errors.append("arms_narrow")
                self.spread_active = False
            else:
                self.spread_active = False
                self.narrow_active = False

        return errors


class SideLateralRaiseState:
    """사이드 레터럴 레이즈 (정면) 룰 기반."""
    DOWN_ARM_ANGLE = 30
    UP_ARM_ANGLE = 75
    TOO_HIGH_ANGLE = 100
    PARTIAL_ROM_ANGLE = 70
    UNEVEN_DELTA = 15.0
    BODY_ENGLISH_DELTA = 10.0
    VIS_TH = 0.5

    def __init__(self):
        self.counter = 0
        self.stage = "down"
        self.rep_max_angle = 0.0
        self.rep_torso_min = 1000.0
        self.rep_torso_max = -1000.0
        self.too_high_active = False
        self.uneven_active = False

    def update(self, landmarks):
        l_sh, r_sh = landmarks[11], landmarks[12]
        l_el, r_el = landmarks[13], landmarks[14]
        l_hip, r_hip = landmarks[23], landmarks[24]
        if min(l_sh[3], r_sh[3], l_el[3], r_el[3], l_hip[3], r_hip[3]) < self.VIS_TH:
            return []

        # 어깨에서 수직 아래로 가상의 점을 두고 어깨-그 점-팔꿈치 각도가 곧 측면 외전 각도
        l_below = (l_sh[0], l_sh[1] + 0.1)
        r_below = (r_sh[0], r_sh[1] + 0.1)
        l_arm_angle = _calc_angle(l_below, (l_sh[0], l_sh[1]), (l_el[0], l_el[1]))
        r_arm_angle = _calc_angle(r_below, (r_sh[0], r_sh[1]), (r_el[0], r_el[1]))
        avg_arm_angle = (l_arm_angle + r_arm_angle) / 2.0

        sh_mid = ((l_sh[0] + r_sh[0]) / 2.0, (l_sh[1] + r_sh[1]) / 2.0)
        hip_mid = ((l_hip[0] + r_hip[0]) / 2.0, (l_hip[1] + r_hip[1]) / 2.0)
        torso_v = _torso_angle_from_vertical(sh_mid, hip_mid)

        errors = []

        if self.stage == "down" and avg_arm_angle > self.UP_ARM_ANGLE:
            self.stage = "up"
            self.counter += 1
            self.rep_max_angle = avg_arm_angle
            self.rep_torso_min = torso_v
            self.rep_torso_max = torso_v
        elif self.stage == "up" and avg_arm_angle < self.DOWN_ARM_ANGLE:
            if self.rep_max_angle < self.PARTIAL_ROM_ANGLE:
                errors.append("slr_partial_rom")
            if self.rep_torso_max - self.rep_torso_min > self.BODY_ENGLISH_DELTA:
                errors.append("slr_body_english")
            self.stage = "down"
            self.rep_max_angle = 0.0
        elif self.stage == "up":
            if avg_arm_angle > self.rep_max_angle:
                self.rep_max_angle = avg_arm_angle
            if torso_v < self.rep_torso_min:
                self.rep_torso_min = torso_v
            if torso_v > self.rep_torso_max:
                self.rep_torso_max = torso_v

        if self.stage == "up":
            if avg_arm_angle > self.TOO_HIGH_ANGLE:
                if not self.too_high_active:
                    self.too_high_active = True
                    errors.append("slr_too_high")
            else:
                self.too_high_active = False

            if abs(l_arm_angle - r_arm_angle) > self.UNEVEN_DELTA:
                if not self.uneven_active:
                    self.uneven_active = True
                    errors.append("slr_uneven")
            else:
                self.uneven_active = False

        return errors


class MachineFlyState:
    """머신플라이 (정면) 룰 기반. 양 손목 사이 거리 / 어깨너비 비율로 사이클 판정."""
    OPEN_RATIO = 1.5
    CLOSE_RATIO = 0.7
    TOO_OPEN_RATIO = 1.8
    PARTIAL_ROM_RATIO = 0.5
    UNEVEN_RATIO = 0.15
    BODY_ENGLISH_DELTA = 10.0
    VIS_TH = 0.5

    def __init__(self):
        self.counter = 0
        self.stage = "open"
        self.rep_min_ratio = 1000.0
        self.rep_torso_min = 1000.0
        self.rep_torso_max = -1000.0
        self.too_open_active = False
        self.uneven_active = False

    def update(self, landmarks):
        l_sh, r_sh = landmarks[11], landmarks[12]
        l_wr, r_wr = landmarks[15], landmarks[16]
        l_hip, r_hip = landmarks[23], landmarks[24]
        if min(l_sh[3], r_sh[3], l_wr[3], r_wr[3], l_hip[3], r_hip[3]) < self.VIS_TH:
            return []

        shoulder_width = abs(l_sh[0] - r_sh[0])
        if shoulder_width < 1e-3:
            return []
        wrist_distance = abs(l_wr[0] - r_wr[0])
        ratio = wrist_distance / shoulder_width

        sh_mid = ((l_sh[0] + r_sh[0]) / 2.0, (l_sh[1] + r_sh[1]) / 2.0)
        hip_mid = ((l_hip[0] + r_hip[0]) / 2.0, (l_hip[1] + r_hip[1]) / 2.0)
        torso_v = _torso_angle_from_vertical(sh_mid, hip_mid)

        errors = []

        if self.stage == "open" and ratio < self.CLOSE_RATIO:
            self.stage = "close"
            self.counter += 1
            self.rep_min_ratio = ratio
            self.rep_torso_min = torso_v
            self.rep_torso_max = torso_v
        elif self.stage == "close" and ratio > self.OPEN_RATIO:
            if self.rep_min_ratio > self.PARTIAL_ROM_RATIO:
                errors.append("fly_partial_rom")
            if self.rep_torso_max - self.rep_torso_min > self.BODY_ENGLISH_DELTA:
                errors.append("fly_body_english")
            self.stage = "open"
            self.rep_min_ratio = 1000.0
        elif self.stage == "close":
            if ratio < self.rep_min_ratio:
                self.rep_min_ratio = ratio
            if torso_v < self.rep_torso_min:
                self.rep_torso_min = torso_v
            if torso_v > self.rep_torso_max:
                self.rep_torso_max = torso_v

        if self.stage == "open":
            if ratio > self.TOO_OPEN_RATIO:
                if not self.too_open_active:
                    self.too_open_active = True
                    errors.append("fly_too_open")
            else:
                self.too_open_active = False

        wrist_y_diff = abs(l_wr[1] - r_wr[1])
        if (wrist_y_diff / shoulder_width) > self.UNEVEN_RATIO:
            if not self.uneven_active:
                self.uneven_active = True
                errors.append("fly_uneven")
        else:
            self.uneven_active = False

        return errors


class SumoDeadliftState:
    """스모 데드리프트 (측면) 룰 기반. 보이는 쪽 신체로 spine 각도 + hip y 트래킹."""
    DOWN_SPINE_ANGLE = 130
    UP_SPINE_ANGLE = 165
    ROUND_BACK_NOSE_DELTA = 0.03  # 코가 어깨 라인보다 아래로 떨어진 정도
    PARTIAL_LOCKOUT_TORSO_ANGLE = 15  # 정점에서 상체-수직 각도가 이보다 크면 미완 락아웃
    BAR_DRIFT_RATIO = 0.2  # 손목-어깨 x거리 / torso height
    HIPS_FIRST_RATIO = 1.5  # 직전 프레임 대비 hip y 감소량 / shoulder y 감소량
    VIS_TH = 0.5

    def __init__(self):
        self.counter = 0
        self.stage = "down"
        self.round_back_active = False
        self.bar_drift_active = False
        self.hips_first_active = False
        self.prev_hip_y = None
        self.prev_shoulder_y = None

    def update(self, landmarks):
        nose = landmarks[0]
        l_sh, r_sh = landmarks[11], landmarks[12]
        l_wr, r_wr = landmarks[15], landmarks[16]
        l_hip, r_hip = landmarks[23], landmarks[24]
        l_knee, r_knee = landmarks[25], landmarks[26]

        l_vis = min(l_sh[3], l_hip[3], l_knee[3], l_wr[3])
        r_vis = min(r_sh[3], r_hip[3], r_knee[3], r_wr[3])
        if max(l_vis, r_vis) < self.VIS_TH or nose[3] < self.VIS_TH:
            return []
        if l_vis >= r_vis:
            sh, hip, knee, wr = l_sh, l_hip, l_knee, l_wr
        else:
            sh, hip, knee, wr = r_sh, r_hip, r_knee, r_wr

        spine_angle = _calc_angle((sh[0], sh[1]), (hip[0], hip[1]), (knee[0], knee[1]))
        torso_v = _torso_angle_from_vertical((sh[0], sh[1]), (hip[0], hip[1]))

        errors = []

        if self.stage == "down" and spine_angle > self.UP_SPINE_ANGLE:
            self.stage = "up"
            self.counter += 1
            if torso_v > self.PARTIAL_LOCKOUT_TORSO_ANGLE:
                errors.append("sumo_partial_lockout")
        elif self.stage == "up" and spine_angle < self.DOWN_SPINE_ANGLE:
            self.stage = "down"

        # 등 굽음: 코가 어깨 라인보다 아래로 떨어짐
        if (nose[1] - sh[1]) > self.ROUND_BACK_NOSE_DELTA:
            if not self.round_back_active:
                self.round_back_active = True
                errors.append("sumo_round_back")
        else:
            self.round_back_active = False

        # 바 경로 이탈: 손목-어깨 x거리 / torso height
        torso_height = abs(sh[1] - hip[1])
        if torso_height > 1e-3:
            drift = abs(wr[0] - sh[0]) / torso_height
            if drift > self.BAR_DRIFT_RATIO:
                if not self.bar_drift_active:
                    self.bar_drift_active = True
                    errors.append("sumo_bar_drift")
            else:
                self.bar_drift_active = False

        # 엉덩이 먼저: hip y 감소율 / shoulder y 감소율 (둘 다 상승 중일 때만)
        if self.prev_hip_y is not None and self.prev_shoulder_y is not None:
            dh = self.prev_hip_y - hip[1]
            ds = self.prev_shoulder_y - sh[1]
            if dh > 0.005 and ds > 0.005 and self.stage == "down":
                if (dh / ds) > self.HIPS_FIRST_RATIO:
                    if not self.hips_first_active:
                        self.hips_first_active = True
                        errors.append("sumo_hips_first")
                else:
                    self.hips_first_active = False
            else:
                self.hips_first_active = False
        self.prev_hip_y = hip[1]
        self.prev_shoulder_y = sh[1]

        return errors


class FrontRaiseState:
    """프론트 레이즈 (측면) 룰 기반. 보이는 쪽 팔의 측면 외전 각도 사용."""
    DOWN_ARM_ANGLE = 30
    UP_ARM_ANGLE = 80
    TOO_HIGH_ANGLE = 105
    PARTIAL_ROM_ANGLE = 70
    LEAN_BACK_TORSO_ANGLE = 15
    BODY_ENGLISH_DELTA = 10.0
    VIS_TH = 0.5

    def __init__(self):
        self.counter = 0
        self.stage = "down"
        self.rep_max_arm = 0.0
        self.rep_torso_min = 1000.0
        self.rep_torso_max = -1000.0
        self.too_high_active = False
        self.lean_back_active = False

    def update(self, landmarks):
        l_sh, r_sh = landmarks[11], landmarks[12]
        l_el, r_el = landmarks[13], landmarks[14]
        l_hip, r_hip = landmarks[23], landmarks[24]

        l_vis = min(l_sh[3], l_el[3], l_hip[3])
        r_vis = min(r_sh[3], r_el[3], r_hip[3])
        if max(l_vis, r_vis) < self.VIS_TH:
            return []
        if l_vis >= r_vis:
            sh, el, hip = l_sh, l_el, l_hip
        else:
            sh, el, hip = r_sh, r_el, r_hip

        below = (sh[0], sh[1] + 0.1)
        arm_angle = _calc_angle(below, (sh[0], sh[1]), (el[0], el[1]))
        torso_v = _torso_angle_from_vertical((sh[0], sh[1]), (hip[0], hip[1]))

        errors = []

        if self.stage == "down" and arm_angle > self.UP_ARM_ANGLE:
            self.stage = "up"
            self.counter += 1
            self.rep_max_arm = arm_angle
            self.rep_torso_min = torso_v
            self.rep_torso_max = torso_v
        elif self.stage == "up" and arm_angle < self.DOWN_ARM_ANGLE:
            if self.rep_max_arm < self.PARTIAL_ROM_ANGLE:
                errors.append("fr_partial_rom")
            if self.rep_torso_max - self.rep_torso_min > self.BODY_ENGLISH_DELTA:
                errors.append("fr_body_english")
            self.stage = "down"
            self.rep_max_arm = 0.0
        elif self.stage == "up":
            if arm_angle > self.rep_max_arm:
                self.rep_max_arm = arm_angle
            if torso_v < self.rep_torso_min:
                self.rep_torso_min = torso_v
            if torso_v > self.rep_torso_max:
                self.rep_torso_max = torso_v

        if self.stage == "up" and arm_angle > self.TOO_HIGH_ANGLE:
            if not self.too_high_active:
                self.too_high_active = True
                errors.append("fr_too_high")
        else:
            if self.stage != "up" or arm_angle <= self.TOO_HIGH_ANGLE:
                self.too_high_active = False

        if torso_v > self.LEAN_BACK_TORSO_ANGLE:
            if not self.lean_back_active:
                self.lean_back_active = True
                errors.append("fr_lean_back")
        else:
            self.lean_back_active = False

        return errors


class CloseGripBenchPressState:
    """클로즈그립 벤치프레스 (정면) 룰 기반."""
    DOWN_ELBOW_ANGLE = 90
    UP_ELBOW_ANGLE = 160
    PARTIAL_ROM_ANGLE = 100
    GRIP_TOO_WIDE_RATIO = 1.3
    UNEVEN_RATIO = 0.15
    ELBOW_FLARE_RATIO = 1.2
    VIS_TH = 0.5

    def __init__(self):
        self.counter = 0
        self.stage = "up"
        self.rep_min_elbow = 1000.0
        self.grip_wide_active = False
        self.uneven_active = False
        self.elbow_flare_active = False

    def update(self, landmarks):
        l_sh, r_sh = landmarks[11], landmarks[12]
        l_el, r_el = landmarks[13], landmarks[14]
        l_wr, r_wr = landmarks[15], landmarks[16]
        if min(l_sh[3], r_sh[3], l_el[3], r_el[3], l_wr[3], r_wr[3]) < self.VIS_TH:
            return []

        l_angle = _calc_angle((l_sh[0], l_sh[1]), (l_el[0], l_el[1]), (l_wr[0], l_wr[1]))
        r_angle = _calc_angle((r_sh[0], r_sh[1]), (r_el[0], r_el[1]), (r_wr[0], r_wr[1]))
        avg_elbow = (l_angle + r_angle) / 2.0

        shoulder_width = abs(l_sh[0] - r_sh[0])
        if shoulder_width < 1e-3:
            return []
        wrist_width = abs(l_wr[0] - r_wr[0])
        elbow_width = abs(l_el[0] - r_el[0])
        grip_ratio = wrist_width / shoulder_width
        elbow_ratio = elbow_width / shoulder_width

        errors = []

        if self.stage == "up" and avg_elbow < self.DOWN_ELBOW_ANGLE:
            self.stage = "down"
            self.rep_min_elbow = avg_elbow
        elif self.stage == "down" and avg_elbow > self.UP_ELBOW_ANGLE:
            if self.rep_min_elbow > self.PARTIAL_ROM_ANGLE:
                errors.append("cgbp_partial_rom")
            self.stage = "up"
            self.counter += 1
            self.rep_min_elbow = 1000.0
        elif self.stage == "down":
            if avg_elbow < self.rep_min_elbow:
                self.rep_min_elbow = avg_elbow

        if grip_ratio > self.GRIP_TOO_WIDE_RATIO:
            if not self.grip_wide_active:
                self.grip_wide_active = True
                errors.append("cgbp_grip_too_wide")
        else:
            self.grip_wide_active = False

        if elbow_ratio > self.ELBOW_FLARE_RATIO:
            if not self.elbow_flare_active:
                self.elbow_flare_active = True
                errors.append("cgbp_elbow_flare")
        else:
            self.elbow_flare_active = False

        if self.stage == "up":
            wrist_y_diff = abs(l_wr[1] - r_wr[1])
            if (wrist_y_diff / shoulder_width) > self.UNEVEN_RATIO:
                if not self.uneven_active:
                    self.uneven_active = True
                    errors.append("cgbp_uneven_press")
            else:
                self.uneven_active = False

        return errors


class SkullCrusherState:
    """라잉 트라이셉스 익스텐션 (측면) 룰 기반. 누운 자세에서 보이는 쪽 팔 사용.
    상완(어깨→팔꿈치)이 천장 방향 수직 유지해야 함.
    sc_incomplete_lockout은 다음 rep 시작 시점에 직전 rep의 'up' 스테이지 최대 elbow 각도 기준으로 판정 (delayed check)."""
    DOWN_ELBOW_ANGLE = 90
    UP_ELBOW_ANGLE = 160
    PARTIAL_ROM_ANGLE = 100
    INCOMPLETE_LOCKOUT_ANGLE = 165
    UPPER_ARM_DRIFT_ANGLE = 20
    VIS_TH = 0.5

    def __init__(self):
        self.counter = 0
        self.stage = "up"
        self.rep_min_elbow = 1000.0
        self.up_max_elbow = 0.0
        self.rep_upper_arm_min = 1000.0
        self.rep_upper_arm_max = -1000.0

    def update(self, landmarks):
        l_sh, r_sh = landmarks[11], landmarks[12]
        l_el, r_el = landmarks[13], landmarks[14]
        l_wr, r_wr = landmarks[15], landmarks[16]

        l_vis = min(l_sh[3], l_el[3], l_wr[3])
        r_vis = min(r_sh[3], r_el[3], r_wr[3])
        if max(l_vis, r_vis) < self.VIS_TH:
            return []
        if l_vis >= r_vis:
            sh, el, wr = l_sh, l_el, l_wr
        else:
            sh, el, wr = r_sh, r_el, r_wr

        elbow_angle = _calc_angle((sh[0], sh[1]), (el[0], el[1]), (wr[0], wr[1]))
        above = (sh[0], sh[1] - 0.1)
        upper_arm_angle = _calc_angle(above, (sh[0], sh[1]), (el[0], el[1]))

        errors = []

        if self.stage == "up" and elbow_angle < self.DOWN_ELBOW_ANGLE:
            # up→down: 직전 rep의 락아웃 검사 (counter>0 일 때만)
            if self.counter > 0 and self.up_max_elbow < self.INCOMPLETE_LOCKOUT_ANGLE:
                errors.append("sc_incomplete_lockout")
            self.up_max_elbow = 0.0
            self.stage = "down"
            self.rep_min_elbow = elbow_angle
            self.rep_upper_arm_min = upper_arm_angle
            self.rep_upper_arm_max = upper_arm_angle
        elif self.stage == "down" and elbow_angle > self.UP_ELBOW_ANGLE:
            # down→up: 이번 rep 완료. partial_rom + upper_arm_drift 검사
            if self.rep_min_elbow > self.PARTIAL_ROM_ANGLE:
                errors.append("sc_partial_rom")
            if self.rep_upper_arm_max - self.rep_upper_arm_min > self.UPPER_ARM_DRIFT_ANGLE:
                errors.append("sc_upper_arm_drift")
            self.stage = "up"
            self.counter += 1
            self.rep_min_elbow = 1000.0
            self.up_max_elbow = elbow_angle
        elif self.stage == "down":
            if elbow_angle < self.rep_min_elbow:
                self.rep_min_elbow = elbow_angle
            if upper_arm_angle < self.rep_upper_arm_min:
                self.rep_upper_arm_min = upper_arm_angle
            if upper_arm_angle > self.rep_upper_arm_max:
                self.rep_upper_arm_max = upper_arm_angle
        elif self.stage == "up":
            if elbow_angle > self.up_max_elbow:
                self.up_max_elbow = elbow_angle

        return errors


class CrunchState:
    """크런치 (측면, 누운 자세) 룰 기반. 어깨-골반-무릎 각도로 사이클 판정."""
    DOWN_ANGLE = 175
    UP_ANGLE = 150
    TOO_HIGH_ANGLE = 110
    PARTIAL_ROM_ANGLE = 165
    HIP_LIFT_DELTA = 0.05
    VIS_TH = 0.5

    def __init__(self):
        self.counter = 0
        self.stage = "down"
        self.rep_min_angle = 1000.0
        self.rep_hip_y_min = 1000.0
        self.rep_hip_y_max = -1000.0

    def update(self, landmarks):
        l_sh, r_sh = landmarks[11], landmarks[12]
        l_hip, r_hip = landmarks[23], landmarks[24]
        l_knee, r_knee = landmarks[25], landmarks[26]

        l_vis = min(l_sh[3], l_hip[3], l_knee[3])
        r_vis = min(r_sh[3], r_hip[3], r_knee[3])
        if max(l_vis, r_vis) < self.VIS_TH:
            return []
        if l_vis >= r_vis:
            sh, hip, knee = l_sh, l_hip, l_knee
        else:
            sh, hip, knee = r_sh, r_hip, r_knee

        body_angle = _calc_angle((sh[0], sh[1]), (hip[0], hip[1]), (knee[0], knee[1]))

        errors = []

        if self.stage == "down" and body_angle < self.UP_ANGLE:
            self.stage = "up"
            self.rep_min_angle = body_angle
            self.rep_hip_y_min = hip[1]
            self.rep_hip_y_max = hip[1]
        elif self.stage == "up" and body_angle > self.DOWN_ANGLE:
            if self.rep_min_angle > self.PARTIAL_ROM_ANGLE:
                errors.append("crunch_partial_rom")
            if self.rep_min_angle < self.TOO_HIGH_ANGLE:
                errors.append("crunch_too_high")
            if (self.rep_hip_y_max - self.rep_hip_y_min) > self.HIP_LIFT_DELTA:
                errors.append("crunch_lower_back_lift")
            self.stage = "down"
            self.counter += 1
            self.rep_min_angle = 1000.0
        elif self.stage == "up":
            if body_angle < self.rep_min_angle:
                self.rep_min_angle = body_angle
            if hip[1] < self.rep_hip_y_min:
                self.rep_hip_y_min = hip[1]
            if hip[1] > self.rep_hip_y_max:
                self.rep_hip_y_max = hip[1]

        return errors


class LegRaiseState:
    """레그 레이즈 (측면, 누운 자세) 룰 기반. 골반→무릎 벡터의 수평 기준 각도로 사이클 판정."""
    DOWN_LEG_ANGLE = 30
    UP_LEG_ANGLE = 70
    PARTIAL_ROM_ANGLE = 65
    PARTIAL_DESCENT_ANGLE = 35
    HIP_LIFT_DELTA = 0.04
    KNEE_FLEX_THRESHOLD = 130
    VIS_TH = 0.5

    def __init__(self):
        self.counter = 0
        self.stage = "down"
        self.rep_max_leg_angle = 0.0
        self.rep_hip_y_min = 1000.0
        self.rep_hip_y_max = -1000.0
        self.rep_knee_angle_min = 1000.0
        self.down_min_leg_angle = 1000.0

    def update(self, landmarks):
        l_hip, r_hip = landmarks[23], landmarks[24]
        l_knee, r_knee = landmarks[25], landmarks[26]
        l_ankle, r_ankle = landmarks[27], landmarks[28]

        l_vis = min(l_hip[3], l_knee[3], l_ankle[3])
        r_vis = min(r_hip[3], r_knee[3], r_ankle[3])
        if max(l_vis, r_vis) < self.VIS_TH:
            return []
        if l_vis >= r_vis:
            hip, knee, ankle = l_hip, l_knee, l_ankle
        else:
            hip, knee, ankle = r_hip, r_knee, r_ankle

        # 다리 각도: 골반→무릎 벡터의 수평 기준 각도. 0°=수평, 90°=수직 위쪽.
        dx = knee[0] - hip[0]
        dy = knee[1] - hip[1]
        leg_angle = abs(np.degrees(np.arctan2(-dy, abs(dx))))

        knee_angle = _calc_angle((hip[0], hip[1]), (knee[0], knee[1]), (ankle[0], ankle[1]))

        errors = []

        if self.stage == "down" and leg_angle > self.UP_LEG_ANGLE:
            # down→up: 직전 down stage가 충분히 내려갔는지 검사 (counter>0)
            if self.counter > 0 and self.down_min_leg_angle > self.PARTIAL_DESCENT_ANGLE:
                errors.append("legraise_partial_descent")
            self.down_min_leg_angle = 1000.0
            self.stage = "up"
            self.rep_max_leg_angle = leg_angle
            self.rep_hip_y_min = hip[1]
            self.rep_hip_y_max = hip[1]
            self.rep_knee_angle_min = knee_angle
        elif self.stage == "up" and leg_angle < self.DOWN_LEG_ANGLE:
            # up→down: rep 완료
            if self.rep_max_leg_angle < self.PARTIAL_ROM_ANGLE:
                errors.append("legraise_partial_rom")
            if (self.rep_hip_y_max - self.rep_hip_y_min) > self.HIP_LIFT_DELTA:
                errors.append("legraise_lower_back_lift")
            if self.rep_knee_angle_min < self.KNEE_FLEX_THRESHOLD:
                errors.append("legraise_excessive_knee_flex")
            self.stage = "down"
            self.counter += 1
            self.rep_max_leg_angle = 0.0
            self.down_min_leg_angle = leg_angle
        elif self.stage == "up":
            if leg_angle > self.rep_max_leg_angle:
                self.rep_max_leg_angle = leg_angle
            if hip[1] < self.rep_hip_y_min:
                self.rep_hip_y_min = hip[1]
            if hip[1] > self.rep_hip_y_max:
                self.rep_hip_y_max = hip[1]
            if knee_angle < self.rep_knee_angle_min:
                self.rep_knee_angle_min = knee_angle
        elif self.stage == "down":
            if leg_angle < self.down_min_leg_angle:
                self.down_min_leg_angle = leg_angle

        return errors


class BicepCurlSideState:
    """바벨 컬 (측면) 룰 기반. 카메라 쪽으로 보이는 팔만 트래킹."""
    DOWN_ELBOW_ANGLE = 150
    UP_ELBOW_ANGLE = 70
    PEAK_CONTRACTION_ANGLE = 60
    LEAN_BACK_ANGLE = 20
    LOOSE_UPPER_ARM_ANGLE = 30
    VIS_TH = 0.5

    def __init__(self):
        self.counter = 0
        self.stage = "down"
        self.peak_min_angle = 1000.0
        self.lean_active = False
        self.loose_active = False

    def update(self, landmarks):
        l_sh, r_sh = landmarks[11], landmarks[12]
        l_el, r_el = landmarks[13], landmarks[14]
        l_wr, r_wr = landmarks[15], landmarks[16]
        l_hip, r_hip = landmarks[23], landmarks[24]

        l_vis = min(l_sh[3], l_el[3], l_wr[3], l_hip[3])
        r_vis = min(r_sh[3], r_el[3], r_wr[3], r_hip[3])
        if max(l_vis, r_vis) < self.VIS_TH:
            return []
        if l_vis >= r_vis:
            sh, el, wr, hip = l_sh, l_el, l_wr, l_hip
        else:
            sh, el, wr, hip = r_sh, r_el, r_wr, r_hip

        sh_xy = (sh[0], sh[1])
        el_xy = (el[0], el[1])
        wr_xy = (wr[0], wr[1])
        hip_xy = (hip[0], hip[1])

        elbow_angle = _calc_angle(sh_xy, el_xy, wr_xy)
        upper_arm_angle = _calc_angle(el_xy, sh_xy, (sh_xy[0], 1.0))
        torso_v = _torso_angle_from_vertical(sh_xy, hip_xy)

        errors = []

        if elbow_angle > self.DOWN_ELBOW_ANGLE:
            if self.stage == "up":
                if self.peak_min_angle != 1000.0 and self.peak_min_angle >= self.PEAK_CONTRACTION_ANGLE:
                    errors.append("weak_peak_contraction")
                self.peak_min_angle = 1000.0
            self.stage = "down"
        elif elbow_angle < self.UP_ELBOW_ANGLE and self.stage == "down":
            self.stage = "up"
            self.counter += 1

        if self.stage == "up" and elbow_angle < self.peak_min_angle:
            self.peak_min_angle = elbow_angle

        if torso_v > self.LEAN_BACK_ANGLE:
            if not self.lean_active:
                self.lean_active = True
                errors.append("lean_back")
        else:
            self.lean_active = False

        if upper_arm_angle > self.LOOSE_UPPER_ARM_ANGLE:
            if not self.loose_active:
                self.loose_active = True
                errors.append("loose_upper_arm")
        else:
            self.loose_active = False

        return errors


class PlankState:
    """플랭크 (측면) 룰 기반. 정적 운동, rep 카운트 없음. 어깨-골반-발목 정렬 기준."""
    HIP_DEVIATION_THRESHOLD = 0.04
    VIS_TH = 0.5

    def __init__(self):
        self.counter = 0
        self.stage = "static"
        self.high_active = False
        self.low_active = False

    def update(self, landmarks):
        l_sh, r_sh = landmarks[11], landmarks[12]
        l_hip, r_hip = landmarks[23], landmarks[24]
        l_ankle, r_ankle = landmarks[27], landmarks[28]

        l_vis = min(l_sh[3], l_hip[3], l_ankle[3])
        r_vis = min(r_sh[3], r_hip[3], r_ankle[3])
        if max(l_vis, r_vis) < self.VIS_TH:
            return []
        if l_vis >= r_vis:
            sh, hip, ankle = l_sh, l_hip, l_ankle
        else:
            sh, hip, ankle = r_sh, r_hip, r_ankle

        # 이미지 좌표는 y가 아래로 증가. 어깨-발목 라인 위 hip 기대 y와의 편차로 판정.
        dx = ankle[0] - sh[0]
        if abs(dx) < 1e-3:
            return []
        t = (hip[0] - sh[0]) / dx
        expected_hip_y = sh[1] + t * (ankle[1] - sh[1])
        deviation = hip[1] - expected_hip_y

        errors = []

        if deviation < -self.HIP_DEVIATION_THRESHOLD:
            if not self.high_active:
                self.high_active = True
                errors.append("plank_high_back")
        else:
            self.high_active = False

        if deviation > self.HIP_DEVIATION_THRESHOLD:
            if not self.low_active:
                self.low_active = True
                errors.append("plank_low_back")
        else:
            self.low_active = False

        return errors


class LungeState:
    """런지 (측면) 룰 기반. 카메라 쪽으로 보이는 앞다리 트래킹."""
    DOWN_KNEE_ANGLE = 120
    UP_KNEE_ANGLE = 165
    KNEE_OVER_TOE_OFFSET = 0.02
    VIS_TH = 0.5

    def __init__(self):
        self.counter = 0
        self.stage = "up"
        self.kot_active = False

    def update(self, landmarks):
        l_hip, r_hip = landmarks[23], landmarks[24]
        l_knee, r_knee = landmarks[25], landmarks[26]
        l_ankle, r_ankle = landmarks[27], landmarks[28]
        l_foot, r_foot = landmarks[31], landmarks[32]

        l_vis = min(l_hip[3], l_knee[3], l_ankle[3], l_foot[3])
        r_vis = min(r_hip[3], r_knee[3], r_ankle[3], r_foot[3])
        if max(l_vis, r_vis) < self.VIS_TH:
            return []
        if l_vis >= r_vis:
            hip, knee, ankle, foot = l_hip, l_knee, l_ankle, l_foot
        else:
            hip, knee, ankle, foot = r_hip, r_knee, r_ankle, r_foot

        knee_angle = _calc_angle((hip[0], hip[1]), (knee[0], knee[1]), (ankle[0], ankle[1]))

        # forward = ankle→foot vector. 무릎이 발끝보다 forward로 더 나가면 knee_over_toe.
        fwd_x = foot[0] - ankle[0]
        fwd_y = foot[1] - ankle[1]
        fwd_len = (fwd_x * fwd_x + fwd_y * fwd_y) ** 0.5
        if fwd_len < 1e-3:
            knee_offset = 0.0
        else:
            knee_dx = knee[0] - foot[0]
            knee_dy = knee[1] - foot[1]
            knee_offset = (knee_dx * fwd_x + knee_dy * fwd_y) / fwd_len

        errors = []

        if self.stage == "up" and knee_angle < self.DOWN_KNEE_ANGLE:
            self.stage = "down"
        elif self.stage == "down" and knee_angle > self.UP_KNEE_ANGLE:
            self.stage = "up"
            self.counter += 1

        if self.stage == "down":
            if knee_offset > self.KNEE_OVER_TOE_OFFSET:
                if not self.kot_active:
                    self.kot_active = True
                    errors.append("knee_over_toe")
            else:
                self.kot_active = False
        else:
            self.kot_active = False

        return errors


RULE_BASED_STATES = {
    "ohp": OHPState,
    "barbell_row": BarbellRowState,
    "incline_bench": InclineBenchState,
    "lat_pulldown": LatPulldownState,
    "side_lateral_raise": SideLateralRaiseState,
    "machine_fly": MachineFlyState,
    "sumo_deadlift": SumoDeadliftState,
    "front_raise": FrontRaiseState,
    "close_grip_bench": CloseGripBenchPressState,
    "skull_crusher": SkullCrusherState,
    "crunch": CrunchState,
    "leg_raise": LegRaiseState,
    "bicep_curl_side": BicepCurlSideState,
    "plank": PlankState,
    "lunge": LungeState,
}


def analyze_video(video_path, exercise_name, frame_skip, yolo_conf, progress_cb=None):
    cfg = EXERCISE_PIPELINES[exercise_name]
    ptype = cfg["type"]
    bundle = load_exercise_bundle(exercise_name)

    yolo = None
    classifier_classes = None
    correct_class_mask = None
    feature_columns = None
    left_arm = None
    right_arm = None
    prev_lunge_stage = None
    rule_state = None

    if ptype == "rf_yolo":
        yolo = load_yolo_model()
        yolo.conf = yolo_conf
        classifier = bundle["model"]
        classifier_classes = [str(c) for c in classifier.classes_]
        correct_class_mask = np.array(
            ["correct" in c.lower() for c in classifier_classes], dtype=bool
        )
    elif ptype == "rule_based":
        rule_state = RULE_BASED_STATES[cfg["state_key"]]()
    else:
        feature_columns = _build_feature_columns(cfg["important_lms"])
        if ptype == "knn_full":
            left_arm = BicepArmState("LEFT", cfg)
            right_arm = BicepArmState("RIGHT", cfg)

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
    error_counter = Counter()
    class_counter = Counter()
    rep_count = 0
    current_stage = ""

    events = []
    current_event = None
    landmarks_by_frame = {}

    def _advance(idx):
        if progress_cb:
            progress_cb(idx + 1, total_frames)

    def _record_errors(errors_in_frame, idx):
        # 다중 오류 프레임: 첫 번째는 연속 이벤트로 이어붙이고 나머지는 단독 이벤트로 추가
        nonlocal current_event
        for ek in errors_in_frame:
            error_counter[ek] += 1
        if not errors_in_frame:
            if current_event is not None:
                events.append(current_event)
                current_event = None
            return
        primary = errors_in_frame[0]
        if current_event is not None and current_event["key"] == primary:
            current_event["end_frame"] = idx
        else:
            if current_event is not None:
                events.append(current_event)
            current_event = {"key": primary, "start_frame": idx, "end_frame": idx}
        for ek in errors_in_frame[1:]:
            events.append({"key": ek, "start_frame": idx, "end_frame": idx})

    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_skip != 0:
                _advance(frame_idx)
                frame_idx += 1
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = frame_rgb.shape[:2]

            if ptype == "rf_yolo":
                yolo_results = yolo(frame_rgb)
                preds = yolo_results.pred[0]
                crop = None
                bbox = None
                if preds is not None and len(preds) > 0:
                    best = preds[preds[:, 4].argmax()]
                    x1, y1, x2, y2 = [int(v.item()) for v in best[:4]]
                    x1 = max(0, x1); y1 = max(0, y1)
                    x2 = min(w, x2); y2 = min(h, y2)
                    if x2 > x1 and y2 > y1:
                        crop = frame_rgb[y1:y2, x1:x2]
                        bbox = (x1, y1, x2, y2)

                if crop is None or crop.size == 0:
                    _advance(frame_idx); frame_idx += 1; continue

                pose_result = pose.process(crop)
                if pose_result.pose_landmarks is None:
                    _advance(frame_idx); frame_idx += 1; continue

                landmarks_full = [
                    (lm.x, lm.y, lm.z, lm.visibility)
                    for lm in pose_result.pose_landmarks.landmark
                ]
                landmarks_by_frame[frame_idx] = {
                    "landmarks": landmarks_full,
                    "crop_bbox": bbox,
                }
                row = [v for lm in landmarks_full for v in lm]
                X = pd.DataFrame([row])
                try:
                    probs = classifier.predict_proba(X)[0]
                    pred_class = classifier_classes[int(np.argmax(probs))]
                except Exception:
                    _advance(frame_idx); frame_idx += 1; continue

                class_counter[str(pred_class)] += 1
                _, stage, error_key = classify_posture(pred_class)

                if stage == "down":
                    current_stage = "down"
                elif stage == "up" and current_stage == "down":
                    current_stage = "up"
                    rep_count += 1

                _record_errors([error_key] if error_key else [], frame_idx)

                if stage in ("up", "down"):
                    analyzed += 1
                    correct_prob_sum += float(probs[correct_class_mask].sum())
            else:
                pose_result = pose.process(frame_rgb)
                if pose_result.pose_landmarks is None:
                    _advance(frame_idx); frame_idx += 1; continue

                landmarks_full = [
                    (lm.x, lm.y, lm.z, lm.visibility)
                    for lm in pose_result.pose_landmarks.landmark
                ]
                landmarks_by_frame[frame_idx] = {
                    "landmarks": landmarks_full,
                    "crop_bbox": (0, 0, w, h),
                }

                if ptype == "rule_based":
                    errors_in_frame = rule_state.update(landmarks_full)
                    rep_count = rule_state.counter
                    class_counter[f"{cfg['state_key']}_stage_{rule_state.stage}"] += 1
                    _record_errors(errors_in_frame, frame_idx)
                    analyzed += 1
                    correct_prob_sum += 0.0 if errors_in_frame else 1.0
                    _advance(frame_idx); frame_idx += 1; continue

                try:
                    subset_row = _extract_subset_row(landmarks_full, cfg["important_lms"])
                    X = pd.DataFrame([subset_row], columns=feature_columns)
                    X_scaled = pd.DataFrame(bundle["scaler"].transform(X))
                except Exception:
                    _advance(frame_idx); frame_idx += 1; continue

                errors_in_frame = []
                correct_score = 0.0

                if ptype == "knn_full":
                    try:
                        probs = bundle["model"].predict_proba(X_scaled)[0]
                        classes = list(bundle["model"].classes_)
                        idx_max = int(np.argmax(probs))
                        pred_class = str(classes[idx_max])
                        pred_prob = float(probs[idx_max])
                    except Exception:
                        _advance(frame_idx); frame_idx += 1; continue

                    class_counter[f"bicep_{pred_class}"] += 1
                    if pred_prob >= cfg["prob_threshold"]:
                        mapped = cfg["posture_class_map"].get(pred_class)
                        if mapped is not None:
                            errors_in_frame.append(mapped)
                    if "C" in classes:
                        correct_score = float(probs[classes.index("C")])

                    le = left_arm.update(landmarks_full)
                    re = right_arm.update(landmarks_full)
                    if le and le not in errors_in_frame:
                        errors_in_frame.append(le)
                    if re and re not in errors_in_frame:
                        errors_in_frame.append(re)
                    rep_count = max(left_arm.counter, right_arm.counter)

                elif ptype == "lr_full":
                    try:
                        probs = bundle["model"].predict_proba(X_scaled)[0]
                        classes = list(bundle["model"].classes_)
                        idx_max = int(np.argmax(probs))
                        pred_int = int(classes[idx_max])
                        pred_prob = float(probs[idx_max])
                    except Exception:
                        _advance(frame_idx); frame_idx += 1; continue

                    class_counter[f"plank_{pred_int}"] += 1
                    if pred_prob >= cfg["prob_threshold"]:
                        mapped = cfg["posture_class_map"].get(pred_int)
                        if mapped is not None:
                            errors_in_frame.append(mapped)
                    if 0 in classes:
                        correct_score = float(probs[classes.index(0)])

                elif ptype == "svc_lr_full":
                    try:
                        stage_probs = bundle["stage_model"].predict_proba(X_scaled)[0]
                        stage_classes = list(bundle["stage_model"].classes_)
                        s_idx = int(np.argmax(stage_probs))
                        stage_pred = str(stage_classes[s_idx])
                        stage_prob = float(stage_probs[s_idx])
                    except Exception:
                        _advance(frame_idx); frame_idx += 1; continue

                    class_counter[f"lunge_stage_{stage_pred}"] += 1
                    new_stage = None
                    if stage_prob >= cfg["prob_threshold"]:
                        new_stage = cfg["stage_map"].get(stage_pred)
                    if new_stage == "down" and prev_lunge_stage in ("init", "mid"):
                        rep_count += 1
                    if new_stage is not None:
                        prev_lunge_stage = new_stage

                    if prev_lunge_stage == "down":
                        try:
                            err_probs = bundle["err_model"].predict_proba(X_scaled)[0]
                            err_classes = list(bundle["err_model"].classes_)
                            e_idx = int(np.argmax(err_probs))
                            err_pred = err_classes[e_idx]
                            err_prob = float(err_probs[e_idx])
                            if err_prob >= cfg["prob_threshold"]:
                                mapped = cfg["err_class_map"].get(err_pred)
                                if mapped is not None:
                                    errors_in_frame.append(mapped)
                            correct_cls = cfg.get("err_correct_class")
                            if correct_cls is not None and correct_cls in err_classes:
                                correct_score = float(err_probs[err_classes.index(correct_cls)])
                            else:
                                correct_score = 1.0 if not errors_in_frame else 0.0
                        except Exception:
                            correct_score = 1.0
                    else:
                        correct_score = 1.0

                _record_errors(errors_in_frame, frame_idx)
                analyzed += 1
                correct_prob_sum += correct_score

            _advance(frame_idx)
            frame_idx += 1
    finally:
        if current_event is not None:
            events.append(current_event)
        cap.release()
        pose.close()

    event_groups = {}
    for ev in events:
        dur_frames = ev["end_frame"] - ev["start_frame"] + 1
        dur_sec = dur_frames / fps if fps > 0 else 0.0
        start_sec = ev["start_frame"] / fps if fps > 0 else 0.0
        event_groups.setdefault(ev["key"], []).append(
            {
                "duration_sec": dur_sec,
                "start_sec": start_sec,
                "start_frame": ev["start_frame"],
                "end_frame": ev["end_frame"],
            }
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
        "landmarks_by_frame": landmarks_by_frame,
        "pipeline_type": ptype,
        "static": bool(cfg.get("static", False)),
    }


def render_results(result, exercise_name, penalty_per_type, min_duration_sec):
    event_groups = result.get("event_groups", {})
    score, significant, filtered = compute_score_from_events(
        event_groups, penalty_per_type, min_duration_sec, result=result
    )

    st.subheader("분석 결과")

    is_plank = bool(result.get("static"))
    analyzed_frames = max(int(result.get("analyzed_frames", 0)), 1)
    wrong_frames = sum(result.get("error_counter", {}).values())
    correct_ratio_pct = max(analyzed_frames - wrong_frames, 0) / analyzed_frames * 100.0

    col1, col2, col3 = st.columns(3)
    col1.metric("전체 점수", f"{score:.0f} / 100")
    if is_plank:
        col2.metric("정자세 유지 비율", f"{correct_ratio_pct:.0f} %")
    else:
        col2.metric("감지된 rep 수", f"{result['rep_count']} 회")
    col3.metric("분석된 프레임", f"{result['analyzed_frames']} / {result['total_frames']}")

    if is_plank:
        st.caption(
            f"점수 공식: (정자세 프레임 / 분석 프레임) × 100  |  "
            f"플랭크는 정적 운동이라 rep 카운트 없이 자세 유지 비율로 채점합니다."
        )
    else:
        st.caption(
            f"점수 공식: 100 − (유의미 오류 유형 수 {len(significant)} × 유형당 감점 {penalty_per_type:.0f})  |  "
            f"유의미 기준: 최소 지속 {min_duration_sec:.1f}초 이상"
        )

    if result["analyzed_frames"] == 0:
        st.warning(
            "분석 가능한 프레임이 없습니다. 촬영 각도나 인물이 잘 잡혔는지 확인하고 다시 시도해주세요."
        )
        return

    annotated_path = st.session_state.get("annotated_video_path")
    if annotated_path and os.path.exists(annotated_path):
        st.markdown("### 오류 구간 표시 영상")
        try:
            with open(annotated_path, "rb") as f:
                video_b64 = base64.b64encode(f.read()).decode("ascii")
            st.markdown(
                f"""
                <video autoplay muted loop controls playsinline
                       style="width:100%; max-height:600px; border-radius:8px; background:#000;">
                    <source src="data:video/mp4;base64,{video_b64}" type="video/mp4">
                </video>
                """,
                unsafe_allow_html=True,
            )
        except Exception:
            st.video(annotated_path)
        st.caption(
            f"감지된 오류 구간 동안 해당 신체 부위에 빨간 원이 그려지고 화면 상단에 교정 문구가 표시됩니다. "
            f"진한 빨강은 유의미 오류({min_duration_sec:.1f}초 이상, 점수 반영), "
            f"주황은 짧은 튕김 오류(점수 미반영)입니다. "
            f"피드백 영상은 0.6배속으로 재생되어 자세를 천천히 확인할 수 있습니다. (자동재생을 위해 음소거됨)"
        )

    render_gymscore_feedback(
        result, exercise_name, score, significant, event_groups, penalty_per_type
    )

    st.markdown("### 자세별 피드백")
    if not event_groups:
        st.success(
            f"{exercise_name} 자세가 전반적으로 양호합니다. 감지된 자세 오류가 없습니다."
        )
    else:
        sorted_keys = sorted(
            event_groups.keys(),
            key=lambda k: sum(ev["duration_sec"] for ev in event_groups[k]),
            reverse=True,
        )
        for error_key in sorted_keys:
            evs = event_groups[error_key]
            total_sec = sum(ev["duration_sec"] for ev in evs)
            max_sec = max((ev["duration_sec"] for ev in evs), default=0.0)
            message = FEEDBACK_MESSAGES.get(error_key, error_key)
            counted = error_key in significant
            prefix = "감점 반영" if counted else "무시됨 (지속 짧음)"
            tag = f"[{prefix} · {len(evs)}회 감지 · 총 {total_sec:.1f}초 · 최장 {max_sec:.1f}초]"
            if counted:
                st.error(f"{tag} {message}")
            else:
                st.info(f"{tag} {message}")

    st.markdown("### 오류 유형별 지속 시간")
    if event_groups:
        chart_df = pd.DataFrame(
            {
                "오류 유형": list(event_groups.keys()),
                "총 지속 시간(초)": [
                    sum(ev["duration_sec"] for ev in evs)
                    for evs in event_groups.values()
                ],
            }
        ).set_index("오류 유형")
        st.bar_chart(chart_df)
    else:
        st.info("집계된 오류가 없어 그래프를 표시하지 않습니다.")

    with st.expander("전체 클래스 분포 보기"):
        class_df = pd.DataFrame(
            {
                "클래스": list(result["class_counter"].keys()),
                "프레임 수": list(result["class_counter"].values()),
            }
        ).sort_values("프레임 수", ascending=False)
        st.dataframe(class_df, use_container_width=True)


def main():
    st.set_page_config(
        page_title="운동 영상 자세 분석 서비스",
        layout="centered",
        initial_sidebar_state="auto",
    )

    st.title("자세 분석")
    st.caption("영상을 올리면 AI가 운동 자세를 분석해 점수와 피드백을 알려드려요.")

    selected_category = st.radio(
        "부위 선택",
        list(EXERCISE_CATEGORIES.keys()),
        horizontal=True,
    )
    exercise_name = st.radio(
        "운동 선택",
        EXERCISE_CATEGORIES[selected_category],
        horizontal=True,
    )

    _guide_text = CAMERA_GUIDE[exercise_name]
    _guide_img = _find_guide_image(exercise_name)
    if _guide_img:
        _ext = os.path.splitext(_guide_img)[1].lstrip(".").lower()
        if _ext == "jpg":
            _ext = "jpeg"
        with open(_guide_img, "rb") as _f:
            _img_b64 = base64.b64encode(_f.read()).decode()
        st.markdown(
            f"""
            <style>
            .guide-card {{
                background: #ffffff;
                border: 1px solid #ececec;
                border-radius: 14px;
                padding: 16px;
                box-shadow: 0 6px 20px rgba(15, 23, 42, 0.06);
                margin-bottom: 18px;
            }}
            .guide-card .guide-img {{
                width: 100%;
                height: auto;
                max-height: 360px;
                object-fit: contain;
                border-radius: 10px;
                background: #f7f7f8;
                display: block;
            }}
            .guide-card .guide-caption {{
                margin-top: 12px;
                font-size: 0.82rem;
                line-height: 1.55;
                color: #5b6470;
                letter-spacing: -0.01em;
            }}
            section[data-testid="stFileUploadDropzone"],
            div[data-testid="stFileUploadDropzone"] {{
                border-radius: 12px;
                border: 1.5px dashed #d6d8de;
                background: #fafbfc;
                transition: border-color 0.18s ease, background 0.18s ease;
            }}
            section[data-testid="stFileUploadDropzone"]:hover,
            div[data-testid="stFileUploadDropzone"]:hover {{
                border-color: #9aa1ac;
                background: #f3f5f8;
            }}
            </style>
            <div class="guide-card">
                <img class="guide-img" src="data:image/{_ext};base64,{_img_b64}" />
                <div class="guide-caption">{_guide_text}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.info(_guide_text)

    uploaded_file = st.file_uploader(
        "운동 영상 업로드 (mp4 / mov / avi)",
        type=["mp4", "mov", "avi"],
    )

    with st.sidebar:
        st.header("분석 설정 (사용자에게 보이지않게 내장 해주세요)")
        frame_skip = st.slider(
            "프레임 건너뛰기 (n 프레임마다 1번 분석)",
            min_value=1,
            max_value=10,
            value=3,
            help="값이 클수록 빠르지만 세밀한 분석은 줄어듭니다.",
        )
        yolo_conf = st.slider(
            "YOLO 사람 검출 신뢰도",
            min_value=0.1,
            max_value=0.9,
            value=0.5,
            step=0.05,
        )
        st.divider()
        st.subheader("채점 설정")
        penalty_per_type = st.slider(
            "오류 유형당 감점",
            min_value=5,
            max_value=50,
            value=25,
            step=5,
            help="감지된 오류 유형 1개당 100점에서 차감되는 점수.",
        )
        min_duration_sec = st.slider(
            "유의미 오류 최소 지속 시간(초)",
            min_value=0.2,
            max_value=3.0,
            value=1.0,
            step=0.1,
            help="이 시간보다 짧은 튕김 오류는 감점에 반영되지 않습니다.",
        )

    start = st.button("분석 시작", type="primary", disabled=uploaded_file is None)

    if start and uploaded_file is not None:
        suffix = os.path.splitext(uploaded_file.name)[1] or ".mp4"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.read())
            temp_path = tmp.name

        progress_bar = st.progress(0.0, text="분석을 준비하는 중입니다...")
        status = st.empty()

        def update_progress(current, total):
            ratio = min(current / max(total, 1), 1.0)
            progress_bar.progress(ratio, text=f"프레임 처리 중 {current} / {total}")

        try:
            status.info("영상을 분석하고 있습니다. 잠시만 기다려주세요.")
            result = analyze_video(
                video_path=temp_path,
                exercise_name=exercise_name,
                frame_skip=frame_skip,
                yolo_conf=yolo_conf,
                progress_cb=update_progress,
            )
            progress_bar.progress(1.0, text="분석 완료")

            event_groups_pre = result.get("event_groups", {})
            _, significant_pre, _ = compute_score_from_events(
                event_groups_pre, penalty_per_type, min_duration_sec, result=result
            )

            status.info("오류 구간 표시 영상을 만드는 중입니다...")
            progress_bar.progress(0.0, text="영상 합성 시작")
            annotated_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            annotated_tmp.close()
            annotated_path = annotated_tmp.name

            ok = annotate_video_with_errors(
                input_path=temp_path,
                output_path=annotated_path,
                event_groups=event_groups_pre,
                significant_keys=significant_pre,
                frame_skip=frame_skip,
                landmarks_by_frame=result.get("landmarks_by_frame"),
                progress_cb=lambda c, t: progress_bar.progress(
                    min(c / max(t, 1), 1.0), text=f"영상 합성 중 {c} / {t}"
                ),
            )

            prev = st.session_state.get("annotated_video_path")
            if prev and prev != annotated_path and os.path.exists(prev):
                try:
                    os.unlink(prev)
                except OSError:
                    pass

            if ok:
                st.session_state["annotated_video_path"] = annotated_path
            else:
                st.session_state.pop("annotated_video_path", None)
                try:
                    os.unlink(annotated_path)
                except OSError:
                    pass

            progress_bar.progress(1.0, text="완료")
            status.success("분석이 완료되었습니다.")
            st.session_state["analysis_result"] = result
            st.session_state["analysis_exercise"] = exercise_name
        except Exception as e:
            status.error(f"분석 중 오류가 발생했습니다: {e}")
            st.exception(e)
            st.session_state.pop("analysis_result", None)
            st.session_state.pop("annotated_video_path", None)
        finally:
            try:
                os.unlink(temp_path)
            except OSError:
                pass

    if "analysis_result" in st.session_state:
        render_results(
            st.session_state["analysis_result"],
            st.session_state.get("analysis_exercise", exercise_name),
            penalty_per_type,
            min_duration_sec,
        )


if __name__ == "__main__":
    main()
