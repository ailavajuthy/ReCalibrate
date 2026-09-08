import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque

mp_face_mesh=mp.solutions.face_mesh
mp_pose=mp.solutions.pose

CONSEC_FRAMES_FOR_BLINK=2
THRESHOLD_FRACTION=0.75
ROLLING_WINDOW_SECONDS=60

POSTURE_MULTIPLIER=1.3
POSTURE_WINDOW_SECONDS=60

CALIBRATION_SECONDS=20

LEFT_EYE=[33,160,158,133,153,144]
RIGHT_EYE=[362,385,387,263,373,380]


def euclidean(a,b):
    return np.linalg.norm(np.array(a)-np.array(b))


def eye_aspect_ratio(landmarks,eye_indices,image_width,image_height):
    p1,p2,p3,p4,p5,p6=[
        (landmarks[i].x*image_width,landmarks[i].y*image_height) for i in eye_indices
    ]
    vertical=euclidean(p2,p6)+euclidean(p3,p5)
    horizontal=euclidean(p1,p4)
    if horizontal<1e-3:
        return None
    return vertical/(2.0*horizontal)


def posture_ratios(pose_landmarks,image_width,image_height):
    L_EAR,R_EAR=mp_pose.PoseLandmark.LEFT_EAR,mp_pose.PoseLandmark.RIGHT_EAR
    L_SH,R_SH=mp_pose.PoseLandmark.LEFT_SHOULDER,mp_pose.PoseLandmark.RIGHT_SHOULDER

    l_ear,r_ear=pose_landmarks[L_EAR.value],pose_landmarks[R_EAR.value]
    l_sh,r_sh=pose_landmarks[L_SH.value],pose_landmarks[R_SH.value]

    if l_ear.visibility>=r_ear.visibility:
        ear,shoulder=l_ear,l_sh
    else:
        ear,shoulder=r_ear,r_sh

    shoulder_width=abs(l_sh.x-r_sh.x)*image_width
    if shoulder_width<1e-3:
        return None,None

    horizontal_offset=abs(ear.x-shoulder.x)*image_width/shoulder_width
    vertical_neck=abs(ear.y-shoulder.y)*image_height/shoulder_width

    return horizontal_offset,vertical_neck


def combined_posture_score(h_ratio,v_ratio,baseline_h,baseline_v):
    h_badness=h_ratio/baseline_h
    v_badness=baseline_v/max(v_ratio,1e-3)
    return (h_badness+v_badness)/2.0


def main():
    cap=cv2.VideoCapture(0)

    calibrating_until=time.time()+CALIBRATION_SECONDS
    ear_calibration_samples=[]
    posture_calibration_samples=[]
    baseline_ear=None
    blink_threshold=None
    baseline_posture_h=None
    baseline_posture_v=None

    consec_low_frames=0
    blink_timestamps=deque()

    posture_window=deque(maxlen=POSTURE_WINDOW_SECONDS)
    last_posture_sample_time=0
    posture_alert_active=False

    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as face_mesh,mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:

        while cap.isOpened():
            success,frame=cap.read()
            if not success:
                break

            frame=cv2.flip(frame,1)
            h,w,_=frame.shape
            rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

            face_results=face_mesh.process(rgb)
            pose_results=pose.process(rgb)

            now=time.time()
            blink_status="No face detected"
            blink_color=(200,200,200)
            posture_status="No person detected"
            posture_color=(200,200,200)

            if face_results.multi_face_landmarks:
                landmarks=face_results.multi_face_landmarks[0].landmark
                left_ear=eye_aspect_ratio(landmarks,LEFT_EYE,w,h)
                right_ear=eye_aspect_ratio(landmarks,RIGHT_EYE,w,h)

                if left_ear is not None and right_ear is not None:
                    ear=(left_ear+right_ear)/2.0

                    if now<calibrating_until:
                        ear_calibration_samples.append(ear)
                        remaining=int(calibrating_until-now)+1
                        blink_status=f"Calibrating... ({remaining}s)"
                        blink_color=(255,200,0)
                    elif baseline_ear is None:
                        baseline_ear=float(np.median(ear_calibration_samples))
                        blink_threshold=baseline_ear*THRESHOLD_FRACTION
                    else:
                        while blink_timestamps and now-blink_timestamps[0]>ROLLING_WINDOW_SECONDS:
                            blink_timestamps.popleft()

                        if ear<blink_threshold:
                            consec_low_frames+=1
                        else:
                            if consec_low_frames>=CONSEC_FRAMES_FOR_BLINK:
                                blink_timestamps.append(now)
                            consec_low_frames=0

                        elapsed=max(now-calibrating_until,1e-3)
                        blinks_per_min=len(blink_timestamps)*(60.0/min(elapsed,ROLLING_WINDOW_SECONDS))

                        blink_status=f"EAR: {ear:.2f}  Blinks/min: {blinks_per_min:.1f}"
                        blink_color=(0,200,0) if blinks_per_min>=10 else (0,165,255)
                        if blinks_per_min<6 and elapsed>=15:
                            blink_status+="  -- LOW, take a break!"
                            blink_color=(0,0,255)

            if pose_results.pose_landmarks:
                h_ratio,v_ratio=posture_ratios(pose_results.pose_landmarks.landmark,w,h)

                if h_ratio is not None and v_ratio is not None:
                    if now<calibrating_until:
                        posture_calibration_samples.append((h_ratio,v_ratio))
                        remaining=int(calibrating_until-now)+1
                        posture_status=f"Calibrating... ({remaining}s)"
                        posture_color=(255,200,0)
                    elif baseline_posture_h is None:
                        baseline_posture_h=float(np.median([s[0] for s in posture_calibration_samples]))
                        baseline_posture_v=float(np.median([s[1] for s in posture_calibration_samples]))
                    else:
                        score=combined_posture_score(h_ratio,v_ratio,baseline_posture_h,baseline_posture_v)

                        if now-last_posture_sample_time>=1.0:
                            posture_window.append(score)
                            last_posture_sample_time=now

                        is_poor_now=score>POSTURE_MULTIPLIER
                        window_full=len(posture_window)==POSTURE_WINDOW_SECONDS
                        all_poor_in_window=window_full and all(
                            s>POSTURE_MULTIPLIER for s in posture_window
                        )

                        posture_alert_active=all_poor_in_window

                        if posture_alert_active:
                            posture_status=f"POOR POSTURE for 1+ min (score {score:.2f}) -- sit up!"
                            posture_color=(0,0,255)
                        elif is_poor_now:
                            posture_status=f"Drifting (score {score:.2f})"
                            posture_color=(0,165,255)
                        else:
                            posture_status=f"Good posture (score {score:.2f})"
                            posture_color=(0,200,0)

            cv2.putText(frame,blink_status,(20,40),cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,blink_color,2)
            cv2.putText(frame,posture_status,(20,75),cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,posture_color,2)
            cv2.putText(frame,"Press 'c' to recalibrate, 'q' to quit",(20,h-20),
                        cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)

            cv2.imshow("StudySync Detector - Blink + Posture",frame)

            key=cv2.waitKey(5)&0xFF
            if key==ord('q'):
                break
            elif key==ord('c'):
                calibrating_until=time.time()+CALIBRATION_SECONDS
                ear_calibration_samples=[]
                posture_calibration_samples=[]
                baseline_ear=None
                blink_threshold=None
                baseline_posture_h=None
                baseline_posture_v=None
                consec_low_frames=0
                blink_timestamps.clear()
                posture_window.clear()
                last_posture_sample_time=0
                posture_alert_active=False

    cap.release()
    cv2.destroyAllWindows()


if __name__=="__main__":
    main()
