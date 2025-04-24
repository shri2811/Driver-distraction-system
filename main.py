import time
import pprint
import cv2
import mediapipe as mp
import numpy as np
import winsound 
import threading
import time

from attention_scorer import AttentionScorer as AttScorer
from eye_detector import EyeDetector as EyeDet
from parser import get_args
from pose_estimation import HeadPoseEstimator as HeadPoseEst
from utils import get_landmarks, load_camera_parameters

def draw_alert_box(frame, text, position, color):
    """ Draws a semi-transparent alert box with text """
    x, y = position
    box_width = 400
    box_height = 50

    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + box_width, y + box_height), color, -1)
    alpha = 0.5  # Transparency level
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # Display text inside the box
    cv2.putText(frame, text, (x + 20, y + 35), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

last_alert_time = 0
def play_alert_sound():
    """Plays a beep sound once every 3 seconds when an alert is triggered."""
    global last_alert_time
    current_time = time.time()

    if current_time - last_alert_time > 3:  # Allow beep only every 3 seconds
        threading.Thread(target=winsound.Beep, args=(1000, 2000), daemon=True).start()
        last_alert_time = current_time

def main():
    args = get_args()

    if not cv2.useOptimized():
        try:
            cv2.setUseOptimized(True)  # Enable OpenCV optimization
        except:
            print("Could not set OpenCV optimization, script may be slower")

    camera_matrix, dist_coeffs = load_camera_parameters(args.camera_params) if args.camera_params else (None, None)

    if args.verbose:
        print("Arguments and Parameters used:\n")
        pprint.pp(vars(args), indent=4)

    Detector = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        refine_landmarks=True,
    )

    Eye_det = EyeDet(show_processing=args.show_eye_proc)
    Head_pose = HeadPoseEst(show_axis=args.show_axis, camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)

    t_now = time.perf_counter()
    Scorer = AttScorer(
        t_now=t_now, gaze_time_thresh=args.gaze_time_thresh, ear_thresh=0.2,
        roll_thresh=args.roll_thresh, pitch_thresh=args.pitch_thresh, yaw_thresh=args.yaw_thresh,
        gaze_thresh=args.gaze_thresh, pose_time_thresh=args.pose_time_thresh, verbose=args.verbose,
    )

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    prev_time = time.perf_counter()
    fps = 0.0

    # --- Yawning Detection Variables ---
    mar_threshold = 0.2  # Adjust based on testing
    yawn_counter = 0
    yawn_time_threshold = 10
    face_missing_counter = 0  
    face_missing_threshold = 50   # Yawning must persist for 30 frames (~1 second)

    while True:
        t_now = time.perf_counter()
        elapsed_time = t_now - prev_time
        prev_time = t_now

        if elapsed_time > 0:
            fps = np.round(1 / elapsed_time, 3)

        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame from camera")
            break

        if args.camera == 0:
            frame = cv2.flip(frame, 2)
        e1 = cv2.getTickCount()
        frame_size = frame.shape[1], frame.shape[0]

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = np.expand_dims(gray, axis=2)
        gray = np.concatenate([gray, gray, gray], axis=2)

        lms = Detector.process(gray).multi_face_landmarks
        if lms:
            face_missing_counter = 0 
            landmarks = get_landmarks(lms)

            Eye_det.show_eye_keypoints(color_frame=frame, landmarks=landmarks, frame_size=(frame.shape[1], frame.shape[0]))

            cnn_eye_state = Eye_det.get_EAR(frame=gray, landmarks=landmarks)
            tired, perclos_score = Scorer.get_PERCLOS(t_now, fps, cnn_eye_state)
            gaze = Eye_det.get_Gaze_Score(frame=gray, landmarks=landmarks, frame_size=(frame.shape[1], frame.shape[0]))
            frame_det, roll, pitch, yaw = Head_pose.get_pose(frame=frame, landmarks=landmarks, frame_size=(frame.shape[1], frame.shape[0]))

            asleep, looking_away, distracted = Scorer.eval_scores(
                t_now=t_now, cnn_eye_state=cnn_eye_state, gaze_score=gaze,
                head_roll=roll, head_pitch=pitch, head_yaw=yaw
            )

            # --- Yawning Detection ---
            mar = Eye_det.calculate_MAR(landmarks)  # Get MAR score

            if mar > mar_threshold:
                yawn_counter += 1
            else:
                yawn_counter = 0  # Reset counter if mouth closes
            
            # Check if yawning has persisted for long enough
            if yawn_counter > yawn_time_threshold:
                draw_alert_box(frame, "Yawning Detected! Stay Alert!", (200, 250), (0, 0, 255))
                play_alert_sound()

            # Display MAR on screen
            cv2.putText(frame, f"MAR: {round(mar, 3)}", (10, 140), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)   

            if frame_det is not None:
                frame = frame_det
            if asleep or looking_away or distracted:
                play_alert_sound()

            cv2.putText(frame, f"Eye State: {'Open' if cnn_eye_state > 0.5 else 'Closed'}", (10, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, f"Gaze Score: {round(gaze, 3)}", (10, 80), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, f"PERCLOS: {round(perclos_score, 3)}", (10, 110), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            if roll is not None:
                cv2.putText(frame, f"Roll: {roll.round(1)[0]}", (450, 40), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 255), 2, cv2.LINE_AA)
            if pitch is not None:
                cv2.putText(frame, f"Pitch: {pitch.round(1)[0]}", (450, 70), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 255), 2, cv2.LINE_AA)
            if yaw is not None:
                cv2.putText(frame, f"Yaw: {yaw.round(1)[0]}", (450, 100), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 255), 2, cv2.LINE_AA)

       

            #if tired:
                #draw_alert_box(frame, "TIRED! WAKE UP!", (200, 50), (0, 0, 255))
            if asleep:
                draw_alert_box(frame, "ASLEEP! WAKE UP!", (200, 100), (0, 0, 255))
            if looking_away:
                draw_alert_box(frame, "LOOKING AWAY!", (200, 150), (0, 255, 255))
            if distracted:
                draw_alert_box(frame, "DISTRACTED", (200, 200), (0, 0, 255))
        else:
            
        # No face detected, increase missing counter
            face_missing_counter += 1

        # Show "No Person Detected" warning
            draw_alert_box(frame, "No Person Detected!", (200, 200), (0, 0, 255))
        
        # Trigger an alert beep immediately
            play_alert_sound()
        e2 = cv2.getTickCount()
        proc_time_frame_ms = ((e2 - e1) / cv2.getTickFrequency()) * 1000
        #if args.show_fps:
            #cv2.putText(
             #   frame,
                #"FPS:" + str(round(fps)),
                #(10, 400),
               # cv2.FONT_HERSHEY_PLAIN,
               # 2,
               # (255, 0, 255),
                #1,
            #)
        if args.show_proc_time:
            cv2.putText(
                frame,
                "PROC. TIME FRAME:" + str(round(proc_time_frame_ms, 0)) + "ms",
                (10, 430),
                cv2.FONT_HERSHEY_PLAIN,
                2,
                (255, 0, 255),
                1,
            )
        

        cv2.imshow("Driver Monitoring", frame)
        if cv2.waitKey(20) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
