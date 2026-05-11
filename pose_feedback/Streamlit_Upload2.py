import os
import sys
import subprocess
import tempfile
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

YOLO_WEIGHTS_PATH = "./models/best_big_bounding.pt"
EXERCISE_MODEL_PATHS = {
    "벤치프레스": "./models/benchpress/benchpress.pkl",
    "스쿼트": "./models/squat/squat.pkl",
    "데드리프트": "./models/deadlift/deadlift.pkl",
}

CAMERA_GUIDE = {
    "벤치프레스": "정면에서 촬영하세요. 카메라를 발 아래쪽에 두고 바가 정면으로 보이도록 배치하면 그립 너비와 허리 아치를 판정하기 좋습니다.",
    "스쿼트": "측면에서 촬영하세요. 무릎·허리 라인이 한눈에 보이게 옆에서 찍어야 무릎 안쪽 꺾임과 척추 각도를 판정할 수 있습니다.",
    "데드리프트": "측면에서 촬영하세요. 바·허리·무릎이 한 라인에 들어오도록 옆면을 잡으면 척추 중립 여부를 판정하기 좋습니다.",
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

ERROR_CATEGORY_MAP = {
    "excessive_arch": "Posture",
    "spine_neutral": "Posture",
    "arms_spread": "Movement Quality",
    "arms_narrow": "Movement Quality",
    "caved_in_knees": "Stability",
    "feet_spread": "Stability",
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
def load_exercise_model(exercise_name):
    path = EXERCISE_MODEL_PATHS[exercise_name]
    with open(path, "rb") as f:
        return pickle.load(f)


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


def compute_score_from_events(event_groups, penalty_per_type, min_duration_sec):
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


def analyze_video(video_path, exercise_name, frame_skip, yolo_conf, progress_cb=None):
    yolo = load_yolo_model()
    yolo.conf = yolo_conf
    classifier = load_exercise_model(exercise_name)
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
    error_counter = Counter()
    class_counter = Counter()
    rep_count = 0
    current_stage = ""

    events = []
    current_event = None

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

            row = extract_landmark_row(pose_result.pose_landmarks)
            X = pd.DataFrame([row])
            try:
                probs = classifier.predict_proba(X)[0]
                pred_class = classifier_classes[int(np.argmax(probs))]
            except Exception:
                frame_idx += 1
                if progress_cb:
                    progress_cb(frame_idx, total_frames)
                continue

            class_counter[str(pred_class)] += 1
            _, stage, error_key = classify_posture(pred_class)

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

    event_groups = {}
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


def render_results(result, exercise_name, penalty_per_type, min_duration_sec):
    event_groups = result.get("event_groups", {})
    score, significant, filtered = compute_score_from_events(
        event_groups, penalty_per_type, min_duration_sec
    )

    st.subheader("분석 결과")

    col1, col2, col3 = st.columns(3)
    col1.metric("전체 점수", f"{score:.0f} / 100")
    col2.metric("감지된 rep 수", f"{result['rep_count']} 회")
    col3.metric("분석된 프레임", f"{result['analyzed_frames']} / {result['total_frames']}")

    st.caption(
        f"점수 공식: 100 − (유의미 오류 유형 수 {len(significant)} × 유형당 감점 {penalty_per_type:.0f})  |  "
        f"유의미 기준: 최소 지속 {min_duration_sec:.1f}초 이상"
    )

    if result["analyzed_frames"] == 0:
        st.warning(
            "분석 가능한 프레임이 없습니다. 촬영 각도나 인물이 잘 잡혔는지 확인하고 다시 시도해주세요."
        )
        return

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

    st.title("운동 영상 자세 분석 서비스")
    st.caption("업로드한 영상을 YOLOv5 + MediaPipe + Random Forest로 분석해 점수와 피드백을 제공합니다.")

    exercise_name = st.radio(
        "운동 종류를 선택하세요",
        list(EXERCISE_MODEL_PATHS.keys()),
        horizontal=True,
    )

    st.info(CAMERA_GUIDE[exercise_name])

    uploaded_file = st.file_uploader(
        "운동 영상 업로드 (mp4 / mov / avi)",
        type=["mp4", "mov", "avi"],
    )

    with st.sidebar:
        st.header("분석 설정")
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
            status.success("분석이 완료되었습니다.")
            st.session_state["analysis_result"] = result
            st.session_state["analysis_exercise"] = exercise_name
        except Exception as e:
            status.error(f"분석 중 오류가 발생했습니다: {e}")
            st.exception(e)
            st.session_state.pop("analysis_result", None)
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
