import json
import math
import os
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


USER_DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "user_data.json")
BARBELL_WEIGHT_KG = 20.0
PLATE_WEIGHTS_KG = [20.0, 15.0, 10.0, 5.0, 2.5, 1.25]
MIN_WEIGHT_INCREMENT_KG = 2.5
STARTING_WORKING_WEIGHT_PCT = 0.50
TRAINING_MAX_PCT = 0.90


EXERCISE_NAMES_KO = {
    "squat": "스쿼트",
    "bench": "벤치프레스",
    "deadlift": "데드리프트",
    "ohp": "오버헤드 프레스",
    "row": "바벨 로우",
    "incline_bench": "인클라인 벤치프레스",
    "front_squat": "프론트 스쿼트",
    "close_grip_bench": "클로즈 그립 벤치프레스",
    "sumo_deadlift": "스모 데드리프트",
}


FEEDBACK_MESSAGES = {
    "session_success_linear": "이번 세션 모든 세트 성공! 다음 세션부터 {increment}kg 증량됩니다.",
    "session_partial_linear": "일부 세트에서 실패했습니다. 현재 무게 유지하고 다음 세션 재시도하세요.",
    "session_deload": "3회 연속 실패로 디로드 적용: {old_weight}kg → {new_weight}kg (-10%).",
    "weekly_progression": "주간 증량 적용: +{increment}kg.",
    "nsuns_amrap_pr": "AMRAP {reps}회! 다음 세션 TM 증량됩니다.",
    "nsuns_amrap_stall": "AMRAP {reps}회. TM 유지하고 다음 주 재시도하세요.",
    "nsuns_amrap_fail": "AMRAP에서 목표 반복 수 미달. TM 5% 디로드 적용됩니다.",
    "one_rm_required": "이 루틴을 시작하려면 메인 리프트 1RM을 먼저 입력하세요.",
    "tm_initialized": "TM(Training Max) = 1RM × 90% 로 자동 설정되었습니다.",
}


_SL_5X5 = [
    {"target_reps": 5, "weight_pct": 1.0, "is_amrap": False},
    {"target_reps": 5, "weight_pct": 1.0, "is_amrap": False},
    {"target_reps": 5, "weight_pct": 1.0, "is_amrap": False},
    {"target_reps": 5, "weight_pct": 1.0, "is_amrap": False},
    {"target_reps": 5, "weight_pct": 1.0, "is_amrap": False},
]

_SL_1X5 = [
    {"target_reps": 5, "weight_pct": 1.0, "is_amrap": False},
]


_MADCOW_RAMP_5X5 = [
    {"target_reps": 5, "weight_pct": 0.500, "is_amrap": False},
    {"target_reps": 5, "weight_pct": 0.625, "is_amrap": False},
    {"target_reps": 5, "weight_pct": 0.750, "is_amrap": False},
    {"target_reps": 5, "weight_pct": 0.875, "is_amrap": False},
    {"target_reps": 5, "weight_pct": 1.000, "is_amrap": False},
]

_MADCOW_LIGHT_4X5 = [
    {"target_reps": 5, "weight_pct": 0.50, "is_amrap": False},
    {"target_reps": 5, "weight_pct": 0.60, "is_amrap": False},
    {"target_reps": 5, "weight_pct": 0.70, "is_amrap": False},
    {"target_reps": 5, "weight_pct": 0.80, "is_amrap": False},
]

_MADCOW_INTENSITY = [
    {"target_reps": 5, "weight_pct": 0.500, "is_amrap": False},
    {"target_reps": 5, "weight_pct": 0.625, "is_amrap": False},
    {"target_reps": 5, "weight_pct": 0.750, "is_amrap": False},
    {"target_reps": 5, "weight_pct": 0.875, "is_amrap": False},
    {"target_reps": 3, "weight_pct": 1.050, "is_amrap": False, "is_pr_set": True},
    {"target_reps": 8, "weight_pct": 0.700, "is_amrap": False, "is_backoff": True},
]


_NSUNS_T1_BENCH = [
    {"target_reps": 5, "weight_pct": 0.75, "is_amrap": False},
    {"target_reps": 3, "weight_pct": 0.85, "is_amrap": False},
    {"target_reps": 1, "weight_pct": 0.95, "is_amrap": True},
    {"target_reps": 3, "weight_pct": 0.90, "is_amrap": False},
    {"target_reps": 3, "weight_pct": 0.85, "is_amrap": False},
    {"target_reps": 3, "weight_pct": 0.80, "is_amrap": False},
    {"target_reps": 5, "weight_pct": 0.75, "is_amrap": False},
    {"target_reps": 5, "weight_pct": 0.70, "is_amrap": False},
    {"target_reps": 5, "weight_pct": 0.65, "is_amrap": True},
]

_NSUNS_T1_SQUAT = [
    {"target_reps": 5, "weight_pct": 0.75, "is_amrap": False},
    {"target_reps": 3, "weight_pct": 0.85, "is_amrap": False},
    {"target_reps": 1, "weight_pct": 0.95, "is_amrap": True},
    {"target_reps": 3, "weight_pct": 0.90, "is_amrap": False},
    {"target_reps": 3, "weight_pct": 0.85, "is_amrap": False},
    {"target_reps": 3, "weight_pct": 0.80, "is_amrap": False},
    {"target_reps": 3, "weight_pct": 0.75, "is_amrap": False},
    {"target_reps": 3, "weight_pct": 0.70, "is_amrap": False},
    {"target_reps": 3, "weight_pct": 0.65, "is_amrap": True},
]

_NSUNS_T2_6X6 = [
    {"target_reps": 6, "weight_pct": 0.50, "is_amrap": False},
    {"target_reps": 6, "weight_pct": 0.60, "is_amrap": False},
    {"target_reps": 6, "weight_pct": 0.70, "is_amrap": False},
    {"target_reps": 6, "weight_pct": 0.60, "is_amrap": False},
    {"target_reps": 6, "weight_pct": 0.50, "is_amrap": False},
    {"target_reps": 6, "weight_pct": 0.40, "is_amrap": False},
]


_NSUNS_DAY_BENCH_OHP = {
    "id": "1",
    "name_ko": "Day 1 (벤치 + OHP)",
    "lifts": [
        {"lift_id": "bench", "tier": "T1", "sets": _NSUNS_T1_BENCH},
        {"lift_id": "ohp", "tier": "T2", "tm_source": "ohp", "sets": _NSUNS_T2_6X6},
    ],
}
_NSUNS_DAY_SQUAT_SUMO = {
    "id": "2",
    "name_ko": "Day 2 (스쿼트 + 스모 데드)",
    "lifts": [
        {"lift_id": "squat", "tier": "T1", "sets": _NSUNS_T1_SQUAT},
        {"lift_id": "sumo_deadlift", "tier": "T2", "tm_source": "deadlift", "sets": _NSUNS_T2_6X6},
    ],
}
_NSUNS_DAY_OHP_INCLINE = {
    "id": "3",
    "name_ko": "Day 3 (OHP + 인클라인)",
    "lifts": [
        {"lift_id": "ohp", "tier": "T1", "sets": _NSUNS_T1_BENCH},
        {"lift_id": "incline_bench", "tier": "T2", "tm_source": "bench", "sets": _NSUNS_T2_6X6},
    ],
}
_NSUNS_DAY_DEAD_FRONT = {
    "id": "4",
    "name_ko": "Day 4 (데드 + 프론트 스쿼트)",
    "lifts": [
        {"lift_id": "deadlift", "tier": "T1", "sets": _NSUNS_T1_SQUAT},
        {"lift_id": "front_squat", "tier": "T2", "tm_source": "squat", "sets": _NSUNS_T2_6X6},
    ],
}
_NSUNS_DAY_BENCH_CG = {
    "id": "5",
    "name_ko": "Day 5 (벤치 + 클로즈그립)",
    "lifts": [
        {"lift_id": "bench", "tier": "T1", "sets": _NSUNS_T1_BENCH},
        {"lift_id": "close_grip_bench", "tier": "T2", "tm_source": "bench", "sets": _NSUNS_T2_6X6},
    ],
}
_NSUNS_DAY_EXTRA_SQUAT = {
    "id": "6",
    "name_ko": "Day 6 (스쿼트 T2 - 보조)",
    "lifts": [
        {"lift_id": "squat", "tier": "T2", "tm_source": "squat", "sets": _NSUNS_T2_6X6},
    ],
}


ROUTINE_DEFINITIONS = {
    "stronglifts_5x5": {
        "name_ko": "StrongLifts 5x5",
        "level": "초급",
        "frequency_per_week": 3,
        "weight_base": "working_weight",
        "tm_required": False,
        "progression_type": "linear_session",
        "main_lifts": ["squat", "bench", "deadlift", "ohp", "row"],
        "days": [
            {
                "id": "A",
                "name_ko": "Workout A",
                "lifts": [
                    {"lift_id": "squat", "sets": _SL_5X5},
                    {"lift_id": "bench", "sets": _SL_5X5},
                    {"lift_id": "row", "sets": _SL_5X5},
                ],
            },
            {
                "id": "B",
                "name_ko": "Workout B",
                "lifts": [
                    {"lift_id": "squat", "sets": _SL_5X5},
                    {"lift_id": "ohp", "sets": _SL_5X5},
                    {"lift_id": "deadlift", "sets": _SL_1X5},
                ],
            },
        ],
        "progression_rules": {
            "increment_kg": {
                "squat": 2.5, "bench": 2.5, "row": 2.5, "ohp": 2.5, "deadlift": 5.0,
            },
            "deload_fails": 3,
            "deload_pct": 0.10,
        },
    },
    "madcow_5x5": {
        "name_ko": "Madcow 5x5",
        "level": "중급",
        "frequency_per_week": 3,
        "weight_base": "working_weight",
        "tm_required": False,
        "progression_type": "linear_weekly",
        "main_lifts": ["squat", "bench", "deadlift", "ohp", "row"],
        "days": [
            {
                "id": "1",
                "name_ko": "Day 1 (월, 볼륨)",
                "lifts": [
                    {"lift_id": "squat", "sets": _MADCOW_RAMP_5X5},
                    {"lift_id": "bench", "sets": _MADCOW_RAMP_5X5},
                    {"lift_id": "row", "sets": _MADCOW_RAMP_5X5},
                ],
            },
            {
                "id": "2",
                "name_ko": "Day 2 (수, 가벼움)",
                "lifts": [
                    {"lift_id": "squat", "sets": _MADCOW_LIGHT_4X5},
                    {"lift_id": "ohp", "sets": _MADCOW_RAMP_5X5},
                    {"lift_id": "deadlift", "sets": _MADCOW_LIGHT_4X5},
                ],
            },
            {
                "id": "3",
                "name_ko": "Day 3 (금, 인텐시티)",
                "lifts": [
                    {"lift_id": "squat", "sets": _MADCOW_INTENSITY},
                    {"lift_id": "bench", "sets": _MADCOW_INTENSITY},
                    {"lift_id": "row", "sets": _MADCOW_INTENSITY},
                ],
            },
        ],
        "progression_rules": {
            "increment_kg_per_week": {
                "squat": 2.5, "bench": 2.5, "row": 2.5, "ohp": 2.5, "deadlift": 2.5,
            },
        },
    },
    "nsuns_4day": {
        "name_ko": "nSuns 5/3/1 LP (4-day)",
        "level": "중상급",
        "frequency_per_week": 4,
        "weight_base": "training_max",
        "tm_required": True,
        "progression_type": "amrap_weekly",
        "main_lifts": ["bench", "squat", "ohp", "deadlift"],
        "days": [
            _NSUNS_DAY_BENCH_OHP,
            _NSUNS_DAY_SQUAT_SUMO,
            _NSUNS_DAY_OHP_INCLINE,
            _NSUNS_DAY_DEAD_FRONT,
        ],
        "progression_rules": {
            "tm_increment_kg": {
                "bench": 2.27, "ohp": 2.27, "squat": 4.54, "deadlift": 4.54,
            },
            "amrap_min_for_increment": {
                "bench": 1, "ohp": 1, "squat": 1, "deadlift": 1,
            },
            "tm_deload_pct": 0.05,
        },
    },
    "nsuns_5day": {
        "name_ko": "nSuns 5/3/1 LP (5-day, 표준)",
        "level": "중상급",
        "frequency_per_week": 5,
        "weight_base": "training_max",
        "tm_required": True,
        "progression_type": "amrap_weekly",
        "main_lifts": ["bench", "squat", "ohp", "deadlift"],
        "days": [
            _NSUNS_DAY_BENCH_OHP,
            _NSUNS_DAY_SQUAT_SUMO,
            _NSUNS_DAY_OHP_INCLINE,
            _NSUNS_DAY_DEAD_FRONT,
            _NSUNS_DAY_BENCH_CG,
        ],
        "progression_rules": {
            "tm_increment_kg": {
                "bench": 2.27, "ohp": 2.27, "squat": 4.54, "deadlift": 4.54,
            },
            "amrap_min_for_increment": {
                "bench": 1, "ohp": 1, "squat": 1, "deadlift": 1,
            },
            "tm_deload_pct": 0.05,
        },
    },
    "nsuns_6day": {
        "name_ko": "nSuns 5/3/1 LP (6-day)",
        "level": "상급",
        "frequency_per_week": 6,
        "weight_base": "training_max",
        "tm_required": True,
        "progression_type": "amrap_weekly",
        "main_lifts": ["bench", "squat", "ohp", "deadlift"],
        "days": [
            _NSUNS_DAY_BENCH_OHP,
            _NSUNS_DAY_SQUAT_SUMO,
            _NSUNS_DAY_OHP_INCLINE,
            _NSUNS_DAY_DEAD_FRONT,
            _NSUNS_DAY_BENCH_CG,
            _NSUNS_DAY_EXTRA_SQUAT,
        ],
        "progression_rules": {
            "tm_increment_kg": {
                "bench": 2.27, "ohp": 2.27, "squat": 4.54, "deadlift": 4.54,
            },
            "amrap_min_for_increment": {
                "bench": 1, "ohp": 1, "squat": 1, "deadlift": 1,
            },
            "tm_deload_pct": 0.05,
        },
    },
}


ROUTINE_FAMILIES = {
    "stronglifts": {
        "name_ko": "StrongLifts 5x5",
        "variants": [
            {"routine_id": "stronglifts_5x5", "label_ko": "기본"},
        ],
    },
    "madcow": {
        "name_ko": "Madcow 5x5",
        "variants": [
            {"routine_id": "madcow_5x5", "label_ko": "기본"},
        ],
    },
    "nsuns": {
        "name_ko": "nSuns 5/3/1 LP",
        "variants": [
            {"routine_id": "nsuns_4day", "label_ko": "4-day (시간 부족)"},
            {"routine_id": "nsuns_5day", "label_ko": "5-day (표준)"},
            {"routine_id": "nsuns_6day", "label_ko": "6-day (고볼륨)"},
        ],
    },
}


def _init_default_user_data():
    return {
        "user": {
            "unit": "kg",
            "active_routine": None,
            "current_day_index": 0,
        },
        "one_rep_max": {},
        "training_max": {},
        "working_weights": {},
        "consecutive_fails": {},
        "sessions": [],
    }


def _load_user_data():
    if not os.path.exists(USER_DATA_PATH):
        return _init_default_user_data()
    try:
        with open(USER_DATA_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return _init_default_user_data()
    default = _init_default_user_data()
    for key, value in default.items():
        data.setdefault(key, value)
    return data


def _save_user_data(data):
    os.makedirs(os.path.dirname(USER_DATA_PATH), exist_ok=True)
    with open(USER_DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _round_to_nearest_plate(weight, step=None):
    if step is None:
        step = MIN_WEIGHT_INCREMENT_KG
    if weight is None or weight < 0:
        return 0.0
    return round(weight / step) * step


def _floor_to_plate_step(weight, step=None):
    if step is None:
        step = MIN_WEIGHT_INCREMENT_KG
    if weight is None or weight < 0:
        return 0.0
    return math.floor(weight / step) * step


def _calculate_plate_breakdown(weight_kg):
    if weight_kg is None or weight_kg < BARBELL_WEIGHT_KG:
        return []
    per_side = (weight_kg - BARBELL_WEIGHT_KG) / 2.0
    plates = []
    remaining = per_side
    for plate in PLATE_WEIGHTS_KG:
        while remaining >= plate - 1e-6:
            plates.append(plate)
            remaining -= plate
    return plates


def _estimate_1rm_epley(weight, reps):
    if weight is None or weight <= 0 or reps is None or reps <= 0:
        return 0.0
    if reps == 1:
        return float(weight)
    return float(weight) * (1.0 + reps / 30.0)


def _build_1rm_chart_data(sessions, lift_id):
    points = []
    for sess in sessions:
        date_str = sess.get("date", "")
        if not date_str:
            continue
        try:
            dt = datetime.strptime(date_str.split(" ")[0], "%Y-%m-%d")
        except ValueError:
            continue
        for lift in sess.get("lifts", []):
            if lift.get("lift_id") != lift_id:
                continue
            best_1rm = 0.0
            for s in lift.get("sets", []):
                actual = s.get("actual_reps") or 0
                weight = s.get("weight") or 0
                if actual > 0 and weight > 0:
                    est = _estimate_1rm_epley(weight, actual)
                    if est > best_1rm:
                        best_1rm = est
            if best_1rm > 0:
                points.append((dt, best_1rm))
    return points


def _calculate_session_weights(routine_id, day_idx, user_data):
    routine = ROUTINE_DEFINITIONS[routine_id]
    day = routine["days"][day_idx]
    base_source = routine["weight_base"]
    base_dict = (
        user_data["training_max"]
        if base_source == "training_max"
        else user_data["working_weights"]
    )

    result_lifts = []
    for lift_def in day["lifts"]:
        lift_id = lift_def["lift_id"]
        base_key = lift_def.get("tm_source", lift_id)
        base_weight = base_dict.get(base_key)

        result_sets = []
        for set_idx, set_def in enumerate(lift_def["sets"]):
            if base_weight is None:
                actual_weight = None
            else:
                actual_weight = _round_to_nearest_plate(base_weight * set_def["weight_pct"])
            result_sets.append({
                "set_idx": set_idx,
                "target_reps": set_def["target_reps"],
                "weight": actual_weight,
                "weight_pct": set_def["weight_pct"],
                "is_amrap": set_def.get("is_amrap", False),
                "is_pr_set": set_def.get("is_pr_set", False),
                "is_backoff": set_def.get("is_backoff", False),
            })

        result_lifts.append({
            "lift_id": lift_id,
            "lift_name_ko": EXERCISE_NAMES_KO.get(lift_id, lift_id),
            "tier": lift_def.get("tier"),
            "base_weight": base_weight,
            "base_key": base_key,
            "sets": result_sets,
        })

    return {
        "day_id": day["id"],
        "day_name_ko": day["name_ko"],
        "lifts": result_lifts,
    }


def _stronglifts_progress(user_data, routine, lift_result):
    lift_id = lift_result["lift_id"]
    rules = routine["progression_rules"]

    all_success = all(
        (s.get("actual_reps") or 0) >= s.get("target_reps", 0)
        for s in lift_result["sets"]
    )

    current_weight = user_data["working_weights"].get(lift_id, 0.0)
    fails = user_data["consecutive_fails"].get(lift_id, 0)

    if all_success:
        increment = rules["increment_kg"].get(lift_id, MIN_WEIGHT_INCREMENT_KG)
        new_weight = _round_to_nearest_plate(current_weight + increment)
        user_data["working_weights"][lift_id] = new_weight
        user_data["consecutive_fails"][lift_id] = 0
        return {
            "event": "success",
            "lift_id": lift_id,
            "old_weight": current_weight,
            "new_weight": new_weight,
            "message": FEEDBACK_MESSAGES["session_success_linear"].format(increment=increment),
        }

    fails += 1
    if fails >= rules["deload_fails"]:
        new_weight = _floor_to_plate_step(current_weight * (1.0 - rules["deload_pct"]))
        user_data["working_weights"][lift_id] = new_weight
        user_data["consecutive_fails"][lift_id] = 0
        return {
            "event": "deload",
            "lift_id": lift_id,
            "old_weight": current_weight,
            "new_weight": new_weight,
            "message": FEEDBACK_MESSAGES["session_deload"].format(
                old_weight=current_weight, new_weight=new_weight
            ),
        }

    user_data["consecutive_fails"][lift_id] = fails
    return {
        "event": "stall",
        "lift_id": lift_id,
        "old_weight": current_weight,
        "new_weight": current_weight,
        "message": FEEDBACK_MESSAGES["session_partial_linear"],
    }


def _madcow_progress(user_data, routine, lift_id):
    rules = routine["progression_rules"]
    increment = rules["increment_kg_per_week"].get(lift_id, MIN_WEIGHT_INCREMENT_KG)
    current_weight = user_data["working_weights"].get(lift_id, 0.0)
    new_weight = _round_to_nearest_plate(current_weight + increment)
    user_data["working_weights"][lift_id] = new_weight
    return {
        "event": "weekly_progress",
        "lift_id": lift_id,
        "old_weight": current_weight,
        "new_weight": new_weight,
        "message": FEEDBACK_MESSAGES["weekly_progression"].format(increment=increment),
    }


def _nsuns_progress(user_data, routine, lift_result):
    lift_id = lift_result["lift_id"]
    rules = routine["progression_rules"]

    amrap_sets = [s for s in lift_result["sets"] if s.get("is_amrap")]
    if not amrap_sets:
        return None

    primary = amrap_sets[0]
    actual_reps = primary.get("actual_reps") or 0
    min_for_increment = rules["amrap_min_for_increment"].get(lift_id, 1)

    current_tm = user_data["training_max"].get(lift_id, 0.0)

    if actual_reps >= min_for_increment:
        increment = rules["tm_increment_kg"].get(lift_id, MIN_WEIGHT_INCREMENT_KG)
        new_tm = _round_to_nearest_plate(current_tm + increment)
        user_data["training_max"][lift_id] = new_tm
        return {
            "event": "nsuns_pr",
            "lift_id": lift_id,
            "old_weight": current_tm,
            "new_weight": new_tm,
            "amrap_reps": actual_reps,
            "message": FEEDBACK_MESSAGES["nsuns_amrap_pr"].format(reps=actual_reps),
        }

    if actual_reps == 0:
        new_tm = _floor_to_plate_step(current_tm * (1.0 - rules["tm_deload_pct"]))
        user_data["training_max"][lift_id] = new_tm
        return {
            "event": "nsuns_deload",
            "lift_id": lift_id,
            "old_weight": current_tm,
            "new_weight": new_tm,
            "amrap_reps": actual_reps,
            "message": FEEDBACK_MESSAGES["nsuns_amrap_fail"],
        }

    return {
        "event": "nsuns_stall",
        "lift_id": lift_id,
        "old_weight": current_tm,
        "new_weight": current_tm,
        "amrap_reps": actual_reps,
        "message": FEEDBACK_MESSAGES["nsuns_amrap_stall"].format(reps=actual_reps),
    }


def _apply_progression(user_data, routine_id, day_idx, session_result):
    routine = ROUTINE_DEFINITIONS[routine_id]
    events = []

    if routine["progression_type"] == "linear_session":
        for lift_result in session_result["lifts"]:
            events.append(_stronglifts_progress(user_data, routine, lift_result))

    elif routine["progression_type"] == "linear_weekly":
        last_day_idx = len(routine["days"]) - 1
        if day_idx == last_day_idx:
            for lift_id in routine["main_lifts"]:
                events.append(_madcow_progress(user_data, routine, lift_id))

    elif routine["progression_type"] == "amrap_weekly":
        for lift_result in session_result["lifts"]:
            if lift_result.get("tier") != "T1":
                continue
            event = _nsuns_progress(user_data, routine, lift_result)
            if event:
                events.append(event)

    num_days = len(routine["days"])
    user_data["user"]["current_day_index"] = (day_idx + 1) % num_days

    return events


def _append_session_log(user_data, session_result):
    user_data["sessions"].append(session_result)


def _initialize_weights_from_1rm(user_data, routine, force=False):
    is_tm_based = routine["weight_base"] == "training_max"
    target_dict = user_data["training_max"] if is_tm_based else user_data["working_weights"]
    calc_pct = TRAINING_MAX_PCT if is_tm_based else STARTING_WORKING_WEIGHT_PCT

    for lift_id, rm in user_data["one_rep_max"].items():
        if rm is None or rm <= 0:
            continue
        if not force and target_dict.get(lift_id, 0) > 0:
            continue
        value = max(BARBELL_WEIGHT_KG, _round_to_nearest_plate(rm * calc_pct))
        target_dict[lift_id] = value
        user_data["consecutive_fails"].setdefault(lift_id, 0)


def _get_family_id_for_routine(routine_id):
    for family_id, family in ROUTINE_FAMILIES.items():
        for variant in family["variants"]:
            if variant["routine_id"] == routine_id:
                return family_id
    return next(iter(ROUTINE_FAMILIES.keys()))


def _render_sidebar(user_data):
    st.sidebar.title("운동 루틴 프로그램")

    routine_keys = list(ROUTINE_DEFINITIONS.keys())
    current_routine_id = user_data["user"].get("active_routine")
    if current_routine_id not in routine_keys:
        current_routine_id = routine_keys[0]

    current_family_id = _get_family_id_for_routine(current_routine_id)
    family_keys = list(ROUTINE_FAMILIES.keys())
    family_idx = family_keys.index(current_family_id)

    selected_family_id = st.sidebar.selectbox(
        "루틴",
        options=family_keys,
        index=family_idx,
        format_func=lambda k: ROUTINE_FAMILIES[k]["name_ko"],
        key="sidebar_family_select",
    )

    family = ROUTINE_FAMILIES[selected_family_id]
    variants = family["variants"]

    if len(variants) > 1:
        variant_ids = [v["routine_id"] for v in variants]
        if current_routine_id in variant_ids:
            variant_default = variant_ids.index(current_routine_id)
        else:
            variant_default = 0
        selected_id = st.sidebar.radio(
            "버전 선택",
            options=variant_ids,
            index=variant_default,
            format_func=lambda vid: next(v["label_ko"] for v in variants if v["routine_id"] == vid),
            key="sidebar_variant_select",
        )
    else:
        selected_id = variants[0]["routine_id"]

    routine = ROUTINE_DEFINITIONS[selected_id]
    st.sidebar.caption(
        f"난이도: {routine['level']}  ·  주 {routine['frequency_per_week']}회  ·  {len(routine['days'])}개 Day"
    )

    if selected_id != user_data["user"].get("active_routine"):
        user_data["user"]["active_routine"] = selected_id
        user_data["user"]["current_day_index"] = 0
        _initialize_weights_from_1rm(user_data, routine, force=False)
        _save_user_data(user_data)
        _clear_session_state_for_session()
        st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.subheader("1RM 입력 (kg)")

    pending_1rms = {}
    for lift_id in routine["main_lifts"]:
        lift_name = EXERCISE_NAMES_KO.get(lift_id, lift_id)
        current_1rm = user_data["one_rep_max"].get(lift_id, 0.0)
        new_1rm = st.sidebar.number_input(
            lift_name,
            min_value=0.0,
            max_value=500.0,
            value=float(current_1rm),
            step=2.5,
            key=f"input_1rm_{lift_id}",
        )
        pending_1rms[lift_id] = new_1rm

    col_save, col_reset = st.sidebar.columns(2)
    save_clicked = col_save.button("저장", use_container_width=True, key="sidebar_save_1rm")
    reset_clicked = col_reset.button("초기화", use_container_width=True, key="sidebar_reset")

    if save_clicked:
        for lift_id, val in pending_1rms.items():
            user_data["one_rep_max"][lift_id] = val
        _initialize_weights_from_1rm(user_data, routine, force=False)
        _save_user_data(user_data)
        st.sidebar.success("1RM 저장 완료")
        st.rerun()

    if reset_clicked:
        for lift_id, val in pending_1rms.items():
            user_data["one_rep_max"][lift_id] = val
        _initialize_weights_from_1rm(user_data, routine, force=True)
        user_data["user"]["current_day_index"] = 0
        for lift_id in routine["main_lifts"]:
            user_data["consecutive_fails"][lift_id] = 0
        _save_user_data(user_data)
        _clear_session_state_for_session()
        st.sidebar.success("재계산 완료")
        st.rerun()

    st.sidebar.markdown("---")
    base_label = "Training Max" if routine["weight_base"] == "training_max" else "작업 무게"
    st.sidebar.subheader(f"현재 {base_label} (kg)")
    base_dict = (
        user_data["training_max"]
        if routine["weight_base"] == "training_max"
        else user_data["working_weights"]
    )
    for lift_id in routine["main_lifts"]:
        lift_name = EXERCISE_NAMES_KO.get(lift_id, lift_id)
        weight = base_dict.get(lift_id)
        if weight is None or weight <= 0:
            st.sidebar.text(f"{lift_name}: 미설정")
        else:
            st.sidebar.text(f"{lift_name}: {weight:g} kg")

    return selected_id


def _clear_session_state_for_session():
    for key in list(st.session_state.keys()):
        if key.startswith("set_done_") or key.startswith("amrap_"):
            del st.session_state[key]


_EVENT_PREFIX_KO = {
    "success": "[증량]",
    "stall": "[정체]",
    "deload": "[디로드]",
    "weekly_progress": "[주간 증량]",
    "nsuns_pr": "[TM 증량]",
    "nsuns_stall": "[정체]",
    "nsuns_deload": "[디로드]",
}


def _render_set_row(set_info, day_idx, lift_id):
    set_idx = set_info["set_idx"]
    weight = set_info["weight"]
    target = set_info["target_reps"]
    is_amrap = set_info["is_amrap"]
    is_pr = set_info.get("is_pr_set", False)
    is_backoff = set_info.get("is_backoff", False)

    rep_suffix = "+" if is_amrap else ""
    flag_parts = []
    if is_amrap:
        flag_parts.append("(AMRAP)")
    if is_pr:
        flag_parts.append("[PR]")
    if is_backoff:
        flag_parts.append("[백오프]")
    flag_text = " " + " ".join(flag_parts) if flag_parts else ""

    plates = _calculate_plate_breakdown(weight)
    if not plates:
        plate_text = "바벨만"
    else:
        plate_text = "+".join(f"{p:g}" for p in plates)

    col_label, col_plates, col_input = st.columns([4, 3, 2])
    col_label.markdown(f"세트 {set_idx + 1} · **{weight:g} kg** × {target}{rep_suffix}{flag_text}")
    col_plates.text(f"한쪽: {plate_text}")

    if is_amrap:
        col_input.number_input(
            "reps",
            min_value=0, max_value=50, value=int(target), step=1,
            key=f"amrap_{day_idx}_{lift_id}_{set_idx}",
            label_visibility="collapsed",
        )
    else:
        col_input.checkbox(
            "완료",
            value=False,
            key=f"set_done_{day_idx}_{lift_id}_{set_idx}",
        )


def _render_lift_card(lift, day_idx):
    tier = lift.get("tier")
    tier_text = f"  [{tier}]" if tier else ""

    base_text = f"기준 무게: {lift['base_weight']:g} kg"
    if lift.get("base_key") and lift["base_key"] != lift["lift_id"]:
        src_name = EXERCISE_NAMES_KO.get(lift["base_key"], lift["base_key"])
        base_text += f"  ({src_name} 기준)"

    st.markdown(f"### {lift['lift_name_ko']}{tier_text}")
    st.caption(base_text)
    for s in lift["sets"]:
        _render_set_row(s, day_idx, lift["lift_id"])
    st.markdown("---")


def _render_today_session(user_data, routine_id):
    routine = ROUTINE_DEFINITIONS[routine_id]
    num_days = len(routine["days"])
    day_idx = user_data["user"].get("current_day_index", 0) % num_days

    base_dict = (
        user_data["training_max"]
        if routine["weight_base"] == "training_max"
        else user_data["working_weights"]
    )

    missing_lifts = set()
    for lift_def in routine["days"][day_idx]["lifts"]:
        base_key = lift_def.get("tm_source", lift_def["lift_id"])
        if not base_dict.get(base_key, 0):
            missing_lifts.add(EXERCISE_NAMES_KO.get(base_key, base_key))

    if missing_lifts:
        st.warning(f"다음 운동의 1RM이 아직 입력되지 않았습니다: {', '.join(sorted(missing_lifts))}")
        st.info("사이드바에서 1RM을 입력하고 [저장] 버튼을 눌러주세요.")
        return

    if "last_session_events" in st.session_state:
        events = st.session_state.pop("last_session_events")
        if events:
            with st.expander("직전 세션 결과", expanded=True):
                for e in events:
                    prefix = _EVENT_PREFIX_KO.get(e["event"], "[알림]")
                    lift_name = EXERCISE_NAMES_KO.get(e["lift_id"], e["lift_id"])
                    st.write(f"{prefix} **{lift_name}**: {e['message']}")

    session_plan = _calculate_session_weights(routine_id, day_idx, user_data)

    st.subheader(session_plan["day_name_ko"])
    st.caption(f"Day {day_idx + 1} / {num_days}")

    if st.button("모두 완료 체크 (AMRAP 제외)", key="btn_check_all_sets"):
        for lift in session_plan["lifts"]:
            for s in lift["sets"]:
                if s["is_amrap"]:
                    continue
                key = f"set_done_{day_idx}_{lift['lift_id']}_{s['set_idx']}"
                st.session_state[key] = True
        st.rerun()

    for lift in session_plan["lifts"]:
        _render_lift_card(lift, day_idx)

    if st.button("세션 완료", type="primary", use_container_width=True, key="btn_complete_session"):
        _complete_session(user_data, routine_id, day_idx, session_plan)


def _complete_session(user_data, routine_id, day_idx, session_plan):
    session_result = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "routine_id": routine_id,
        "day_idx": day_idx,
        "day_id": session_plan["day_id"],
        "day_name_ko": session_plan["day_name_ko"],
        "lifts": [],
    }

    for lift in session_plan["lifts"]:
        lift_id = lift["lift_id"]
        lift_result = {
            "lift_id": lift_id,
            "lift_name_ko": lift["lift_name_ko"],
            "tier": lift.get("tier"),
            "base_weight": lift["base_weight"],
            "sets": [],
        }
        for s in lift["sets"]:
            set_idx = s["set_idx"]
            if s["is_amrap"]:
                key = f"amrap_{day_idx}_{lift_id}_{set_idx}"
                actual_reps = int(st.session_state.get(key, 0))
            else:
                key = f"set_done_{day_idx}_{lift_id}_{set_idx}"
                done = bool(st.session_state.get(key, False))
                actual_reps = int(s["target_reps"]) if done else 0
            lift_result["sets"].append({
                "set_idx": set_idx,
                "target_reps": s["target_reps"],
                "actual_reps": actual_reps,
                "weight": s["weight"],
                "weight_pct": s["weight_pct"],
                "is_amrap": s["is_amrap"],
                "is_pr_set": s.get("is_pr_set", False),
                "is_backoff": s.get("is_backoff", False),
            })
        session_result["lifts"].append(lift_result)

    events = _apply_progression(user_data, routine_id, day_idx, session_result)
    _append_session_log(user_data, session_result)
    _save_user_data(user_data)
    _clear_session_state_for_session()
    st.session_state["last_session_events"] = events
    st.rerun()


def _render_history_tab(user_data):
    sessions = user_data.get("sessions", [])
    if not sessions:
        st.info("아직 기록된 세션이 없습니다. 세션을 완료하면 여기에 표시됩니다.")
        return

    st.caption(f"총 {len(sessions)}개 세션")

    for sess_idx, sess in enumerate(reversed(sessions)):
        routine_id = sess.get("routine_id", "")
        routine_name = ROUTINE_DEFINITIONS.get(routine_id, {}).get("name_ko", routine_id)
        title = f"{sess.get('date', '?')}  ·  {sess.get('day_name_ko', '')}  ·  {routine_name}"

        with st.expander(title, expanded=(sess_idx == 0)):
            rows = []
            for lift in sess.get("lifts", []):
                lift_name = lift.get("lift_name_ko", lift.get("lift_id", "?"))
                tier = lift.get("tier") or ""
                for s in lift.get("sets", []):
                    target = s.get("target_reps", 0)
                    actual = s.get("actual_reps", 0) or 0
                    weight = s.get("weight") or 0
                    flags = []
                    if s.get("is_amrap"):
                        flags.append("AMRAP")
                    if s.get("is_pr_set"):
                        flags.append("PR")
                    if s.get("is_backoff"):
                        flags.append("백오프")
                    if actual >= target and target > 0:
                        status = "성공"
                    elif actual > 0:
                        status = "부분"
                    else:
                        status = "실패"
                    rows.append({
                        "운동": lift_name,
                        "Tier": tier,
                        "세트": s.get("set_idx", 0) + 1,
                        "무게(kg)": weight,
                        "목표": target,
                        "실제": actual,
                        "상태": status,
                        "플래그": ", ".join(flags),
                    })

            if rows:
                df = pd.DataFrame(rows)
                st.dataframe(df, use_container_width=True, hide_index=True)
            else:
                st.text("이 세션에는 기록된 세트가 없습니다.")


def _render_chart_tab(user_data):
    sessions = user_data.get("sessions", [])
    if not sessions:
        st.info("아직 기록된 세션이 없습니다. 세션을 완료하면 1RM 추이가 표시됩니다.")
        return

    lift_ids_with_data = set()
    for sess in sessions:
        for lift in sess.get("lifts", []):
            lid = lift.get("lift_id")
            if lid:
                lift_ids_with_data.add(lid)

    if not lift_ids_with_data:
        st.info("세션 데이터가 비어있습니다.")
        return

    st.caption("Epley 공식: 추정 1RM = 무게 × (1 + reps / 30)")

    fig = go.Figure()
    plotted_any = False
    for lift_id in sorted(lift_ids_with_data):
        points = _build_1rm_chart_data(sessions, lift_id)
        if not points:
            continue
        dates = [p[0] for p in points]
        values = [p[1] for p in points]
        name = EXERCISE_NAMES_KO.get(lift_id, lift_id)
        fig.add_trace(go.Scatter(
            x=dates, y=values,
            mode="lines+markers",
            name=name,
        ))
        plotted_any = True

    if not plotted_any:
        st.info("플롯할 데이터 포인트가 없습니다. AMRAP 세트나 정상 세트 reps가 기록된 세션이 필요합니다.")
        return

    fig.update_layout(
        title="추정 1RM 추이",
        xaxis_title="날짜",
        yaxis_title="추정 1RM (kg)",
        height=480,
        hovermode="x unified",
    )
    chart_config = {
        "displaylogo": False,
        "modeBarButtonsToRemove": [
            "select2d", "lasso2d",
            "zoomIn2d", "zoomOut2d",
            "autoScale2d",
            "toggleSpikelines",
            "hoverClosestCartesian", "hoverCompareCartesian",
        ],
    }
    st.plotly_chart(fig, use_container_width=True, config=chart_config)


def _render_settings_tab(user_data):
    st.subheader("데이터 파일")
    st.code(USER_DATA_PATH)

    st.subheader("JSON 백업")
    backup_json = json.dumps(user_data, ensure_ascii=False, indent=2)
    st.download_button(
        "user_data.json 다운로드",
        data=backup_json,
        file_name=f"user_data_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json",
        use_container_width=True,
        key="btn_backup_download",
    )

    st.subheader("JSON 복원")
    uploaded = st.file_uploader("백업 JSON 파일 업로드", type=["json"], key="settings_upload")
    if uploaded is not None:
        try:
            content = uploaded.read().decode("utf-8")
            new_data = json.loads(content)
        except (UnicodeDecodeError, json.JSONDecodeError) as ex:
            st.error(f"파일 파싱 실패: {ex}")
        else:
            expected_keys = {
                "user", "one_rep_max", "training_max",
                "working_weights", "consecutive_fails", "sessions",
            }
            if not expected_keys.issubset(new_data.keys()):
                st.error("유효하지 않은 백업 파일입니다. 필수 키가 누락되어 있습니다.")
            else:
                if st.button("복원 적용", type="primary", key="btn_restore_apply"):
                    _save_user_data(new_data)
                    _clear_session_state_for_session()
                    st.success("복원 완료. 페이지를 새로고침하세요.")
                    st.rerun()

    st.markdown("---")
    st.subheader("전체 초기화")
    st.caption("모든 1RM, 작업 무게, 세션 기록을 삭제합니다.")

    if not st.session_state.get("confirm_reset_all"):
        if st.button("전체 초기화", key="btn_reset_request"):
            st.session_state["confirm_reset_all"] = True
            st.rerun()
    else:
        st.warning("정말 모든 데이터를 삭제하시겠습니까?")
        col_yes, col_no = st.columns(2)
        if col_yes.button("예, 삭제", type="primary", key="btn_reset_confirm"):
            _save_user_data(_init_default_user_data())
            _clear_session_state_for_session()
            st.session_state.pop("confirm_reset_all", None)
            st.success("초기화 완료")
            st.rerun()
        if col_no.button("취소", key="btn_reset_cancel"):
            st.session_state.pop("confirm_reset_all", None)
            st.rerun()


def _run_app():
    st.set_page_config(page_title="운동 루틴 프로그램", layout="wide")
    user_data = _load_user_data()
    routine_id = _render_sidebar(user_data)

    routine = ROUTINE_DEFINITIONS[routine_id]
    st.title(routine["name_ko"])
    st.caption("Boostcamp 스타일 루틴 트래커 - 5x5 / nSuns 지원")

    tab_session, tab_history, tab_chart, tab_settings = st.tabs([
        "오늘 세션",
        "세션 기록",
        "1RM 차트",
        "설정",
    ])

    with tab_session:
        _render_today_session(user_data, routine_id)

    with tab_history:
        _render_history_tab(user_data)

    with tab_chart:
        _render_chart_tab(user_data)

    with tab_settings:
        _render_settings_tab(user_data)


if __name__ == "__main__":
    _run_app()
