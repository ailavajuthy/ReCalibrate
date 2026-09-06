import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque

mp_face_mesh = mp.solutions.face_mesh

CONSEC_FRAMES_FOR_BLINK = 2
CALIBRATION_SECONDS = 20
ROLLING_WINDOW_SECONDS = 60
THRESHOLD_FRACTION = 0.75

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]


def euclidean(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))


def eye_aspect_ratio(landmarks, eye_indices, image_width, image_height):
    p1, p2, p3, p4, p5, p6 = [
        (landmarks[i].x * image_width, landmarks[i].y * image_height) for i in eye_indices
    ]
    vertical = euclidean(p2, p6) + euclidean(p3, p5)
    horizontal = euclidean(p1, p4)
    if horizontal < 1e-3:
        return None
    return vertical / (2.0 * horizontal)


def main():
    cap = cv2.VideoCapture(0)
    baseline_ear = None
    blink_threshold = None
    calibration_samples = []
    calibrating_until = time.time() + CALIBRATION_SECONDS

    consec_low_frames = 0
    blink_timestamps = deque()

    print("Calibrating... look at the screen normally (don't force blinks).")

    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as face_mesh:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            status_text = "No face detected"
            status_color = (200, 200, 200)

            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark

                left_ear = eye_aspect_ratio(landmarks, LEFT_EYE, w, h)
                right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE, w, h)

                if left_ear is not None and right_ear is not None:
                    ear = (left_ear + right_ear) / 2.0
                    now = time.time()

                    if now < calibrating_until:
                        calibration_samples.append(ear)
                        remaining = int(calibrating_until - now) + 1
                        status_text = f"Calibrating... blink normally ({remaining}s)"
                        status_color = (255, 200, 0)
                    elif baseline_ear is None:
                        baseline_ear = float(np.median(calibration_samples))
                        blink_threshold = baseline_ear * THRESHOLD_FRACTION
                        print(f"Calibration done. Baseline EAR = {baseline_ear:.3f}, "
                              f"blink threshold = {blink_threshold:.3f}")

                    else:
                        while blink_timestamps and now - blink_timestamps[0] > ROLLING_WINDOW_SECONDS:
                            blink_timestamps.popleft()

                        if ear < blink_threshold:
                            consec_low_frames += 1
                        else:
                            if consec_low_frames >= CONSEC_FRAMES_FOR_BLINK:
                                blink_timestamps.append(now)
                            consec_low_frames = 0

                        elapsed = min(now - (calibrating_until), ROLLING_WINDOW_SECONDS)
                        elapsed = max(elapsed, 1e-3)
                        blinks_per_min = len(blink_timestamps) * (60.0 / min(elapsed, ROLLING_WINDOW_SECONDS))

                        status_text = f"EAR: {ear:.2f}  |  Blinks/min: {blinks_per_min:.1f}"
                        status_color = (0, 200, 0) if blinks_per_min >= 10 else (0, 165, 255)
                        if blinks_per_min < 6 and elapsed >= 15:
                            status_text += "  -- LOW, take a break!"
                            status_color = (0, 0, 255)

            cv2.putText(frame, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, status_color, 2)
            cv2.putText(frame, "Press 'c' to recalibrate, 'q' to quit", (20, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv2.imshow("Blink Rate Detector - StudySync", frame)

            key = cv2.waitKey(5) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                baseline_ear = None
                blink_threshold = None
                calibration_samples = []
                consec_low_frames = 0
                blink_timestamps.clear()
                calibrating_until = time.time() + CALIBRATION_SECONDS
                print("Recalibrating... look at the screen normally.")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()