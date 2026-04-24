import argparse
import csv
import math
import platform
import statistics
import time
import urllib.request
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np


MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)

# Minimal connections for optional drawing in Tasks mode
SIMPLE_CONNECTIONS = [
    (7, 11),   # ear-shoulder left
    (11, 23),  # shoulder-hip left
    (8, 12),   # ear-shoulder right
    (12, 24),  # shoulder-hip right
    (11, 12),  # shoulders line
    (23, 24),  # hips line
]


@dataclass
class Point:
    x: float
    y: float
    visibility: float = 1.0


def ensure_model_file(model_path: Path) -> Path:
    if model_path.exists():
        return model_path

    model_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Downloading pose model to: {model_path}")
    urllib.request.urlretrieve(MODEL_URL, model_path)
    return model_path


def select_body_side(landmarks: List[Point]) -> Optional[Tuple[Point, Point, Point]]:
    """Select the more reliable body side (left or right) by visibility."""
    left = (
        landmarks[7],   # left ear
        landmarks[11],  # left shoulder
        landmarks[23],  # left hip
    )
    right = (
        landmarks[8],   # right ear
        landmarks[12],  # right shoulder
        landmarks[24],  # right hip
    )

    left_score = sum(p.visibility for p in left) / 3.0
    right_score = sum(p.visibility for p in right) / 3.0

    selected = left if left_score >= right_score else right
    if min(p.visibility for p in selected) < 0.55:
        return None
    return selected


def tilt_from_vertical(p1: Point, p2: Point) -> float:
    """Return tilt angle in degrees from a vertical line; 0 means perfectly vertical."""
    dx = p1.x - p2.x
    dy = p1.y - p2.y
    radians = math.atan2(abs(dx), abs(dy) + 1e-9)
    return math.degrees(radians)


def beep_alert() -> None:
    system = platform.system().lower()
    if "windows" in system:
        import winsound

        winsound.Beep(950, 250)
        winsound.Beep(1100, 250)
    else:
        print("\a", end="", flush=True)


def put_text(frame, text: str, y: int, color=(255, 255, 255), scale=0.7, thickness=2):
    cv2.putText(
        frame,
        text,
        (20, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def ensure_log_writer(path: Optional[str]):
    if not path:
        return None, None

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    is_new = not output.exists()
    fp = output.open("a", newline="", encoding="utf-8")
    writer = csv.writer(fp)
    if is_new:
        writer.writerow(["timestamp", "neck_angle", "torso_angle", "is_bad_posture", "bad_duration_sec"])
    return fp, writer


def parse_args():
    parser = argparse.ArgumentParser(description="AI Posture Correction monitor using MediaPipe Pose")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--width", type=int, default=1280, help="Capture width")
    parser.add_argument("--height", type=int, default=720, help="Capture height")
    parser.add_argument("--calibration-seconds", type=float, default=7.0, help="Initial calibration duration in seconds")
    parser.add_argument("--alert-delay", type=float, default=6.0, help="Trigger alert after this many bad seconds")
    parser.add_argument("--alert-cooldown", type=float, default=10.0, help="Minimum seconds between alerts")
    parser.add_argument("--log-file", type=str, default="logs/posture_log.csv", help="CSV log file path")
    parser.add_argument("--show-landmarks", action="store_true", help="Draw simplified landmarks")
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/pose_landmarker_lite.task",
        help="Path to MediaPipe Pose Landmarker task model",
    )
    return parser.parse_args()


def to_points_from_tasks_landmarks(raw_landmarks) -> List[Point]:
    points = []
    for lm in raw_landmarks:
        vis = float(getattr(lm, "visibility", 1.0))
        points.append(Point(float(lm.x), float(lm.y), vis))
    return points


def draw_simple_landmarks(frame, points: List[Point]):
    h, w = frame.shape[:2]

    for i, j in SIMPLE_CONNECTIONS:
        pi, pj = points[i], points[j]
        xi, yi = int(pi.x * w), int(pi.y * h)
        xj, yj = int(pj.x * w), int(pj.y * h)
        cv2.line(frame, (xi, yi), (xj, yj), (50, 180, 255), 2)

    key_ids = [7, 8, 11, 12, 23, 24]
    for idx in key_ids:
        p = points[idx]
        x, y = int(p.x * w), int(p.y * h)
        cv2.circle(frame, (x, y), 4, (220, 240, 255), -1)


def main():
    args = parse_args()

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Try another camera index with --camera.")

    # MediaPipe Tasks API (compatible with recent mediapipe versions)
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision

    model_path = ensure_model_file(Path(args.model_path))

    base_options = mp_python.BaseOptions(model_asset_path=str(model_path))
    options = mp_vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.55,
        min_pose_presence_confidence=0.55,
        min_tracking_confidence=0.55,
    )
    landmarker = mp_vision.PoseLandmarker.create_from_options(options)

    log_fp, log_writer = ensure_log_writer(args.log_file)

    calibration_neck = []
    calibration_torso = []
    baseline_neck = None
    baseline_torso = None

    neck_smooth = deque(maxlen=6)
    torso_smooth = deque(maxlen=6)

    start_time = time.time()
    bad_start = None
    last_alert_time = 0.0

    frame_index = 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int((frame_index / fps) * 1000)
            frame_index += 1

            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            current_neck = None
            current_torso = None
            is_bad = False
            bad_duration = 0.0

            if result.pose_landmarks:
                points = to_points_from_tasks_landmarks(result.pose_landmarks[0])

                if args.show_landmarks:
                    draw_simple_landmarks(frame, points)

                selected = select_body_side(points)
                if selected is not None:
                    ear, shoulder, hip = selected
                    neck_angle = tilt_from_vertical(ear, shoulder)
                    torso_angle = tilt_from_vertical(shoulder, hip)

                    neck_smooth.append(neck_angle)
                    torso_smooth.append(torso_angle)

                    current_neck = float(statistics.mean(neck_smooth))
                    current_torso = float(statistics.mean(torso_smooth))

                    elapsed = time.time() - start_time
                    if elapsed < args.calibration_seconds:
                        calibration_neck.append(current_neck)
                        calibration_torso.append(current_torso)
                    elif baseline_neck is None and calibration_neck and calibration_torso:
                        baseline_neck = float(np.median(calibration_neck))
                        baseline_torso = float(np.median(calibration_torso))

                    if baseline_neck is not None and baseline_torso is not None:
                        neck_threshold = max(18.0, baseline_neck + 8.0)
                        torso_threshold = max(12.0, baseline_torso + 7.0)

                        neck_bad = current_neck > neck_threshold
                        torso_bad = current_torso > torso_threshold
                        is_bad = neck_bad or torso_bad

                        if is_bad:
                            if bad_start is None:
                                bad_start = time.time()
                            bad_duration = time.time() - bad_start
                        else:
                            bad_start = None
                            bad_duration = 0.0

                        if is_bad and bad_duration >= args.alert_delay:
                            now = time.time()
                            if now - last_alert_time >= args.alert_cooldown:
                                beep_alert()
                                last_alert_time = now

                        status_color = (50, 210, 80) if not is_bad else (30, 30, 255)
                        status_text = "GOOD POSTURE" if not is_bad else "BAD POSTURE"

                        put_text(frame, f"Status: {status_text}", 35, color=status_color, scale=0.9, thickness=3)
                        put_text(frame, f"Neck angle: {current_neck:.1f} deg", 70)
                        put_text(frame, f"Torso angle: {current_torso:.1f} deg", 100)
                        put_text(frame, f"Bad duration: {bad_duration:.1f}s / {args.alert_delay:.1f}s", 130)
                        put_text(frame, f"Neck threshold: {neck_threshold:.1f} deg", 160, color=(220, 220, 220), scale=0.62, thickness=1)
                        put_text(frame, f"Torso threshold: {torso_threshold:.1f} deg", 185, color=(220, 220, 220), scale=0.62, thickness=1)

                        if log_writer:
                            log_writer.writerow(
                                [
                                    datetime.now().isoformat(timespec="seconds"),
                                    f"{current_neck:.3f}",
                                    f"{current_torso:.3f}",
                                    int(is_bad),
                                    f"{bad_duration:.3f}",
                                ]
                            )

            elapsed = time.time() - start_time
            if elapsed < args.calibration_seconds:
                remaining = max(0.0, args.calibration_seconds - elapsed)
                put_text(frame, "Calibration: sit in your best posture", 35, color=(0, 215, 255), scale=0.9, thickness=3)
                put_text(frame, f"Hold steady... {remaining:.1f}s", 70, color=(0, 215, 255), scale=0.8)
                put_text(frame, "Tip: keep your side visible to the camera", 100, color=(0, 215, 255), scale=0.65)
            elif baseline_neck is None or baseline_torso is None:
                put_text(frame, "Calibration failed: keep body visible and press C", 35, color=(0, 165, 255), scale=0.8, thickness=2)

            put_text(frame, "Keys: Q=quit, C=recalibrate", frame.shape[0] - 20, color=(220, 220, 220), scale=0.65, thickness=1)

            cv2.imshow("AI Posture Correction", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("c"):
                start_time = time.time()
                baseline_neck = None
                baseline_torso = None
                calibration_neck.clear()
                calibration_torso.clear()
                bad_start = None

    finally:
        if log_fp:
            log_fp.close()
        landmarker.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
