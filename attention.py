import time
from tensorflow.keras.models import load_model

# Load the trained CNN model
MODEL_PATH = "bestModel100.h5"  # Update this if needed
cnn_model = load_model(MODEL_PATH, compile=False)

class AttentionScorer:
    """
    Attention Scorer class that contains methods for estimating eye closure, gaze score, PERCLOS, and head pose over time,
    now using CNN-based eye closure detection instead of EAR.
    """

    def __init__(
        self,
        t_now,
        ear_thresh,  # Kept for compatibility, but not used
        gaze_thresh,
        perclos_thresh=0.1,
        roll_thresh=60,
        pitch_thresh=20,
        yaw_thresh=30,
        ear_time_thresh=2.0,
        gaze_time_thresh=2.0,
        pose_time_thresh=4.0,
        verbose=False,
    ):
        """
        Initialize the AttentionScorer object with the given thresholds and parameters.
        """
        self.PERCLOS_TIME_PERIOD = 60  # Constant for PERCLOS time period

        self.ear_thresh = ear_thresh  # Not used, but kept for compatibility
        self.gaze_thresh = gaze_thresh
        self.perclos_thresh = perclos_thresh
        self.roll_thresh = roll_thresh
        self.pitch_thresh = pitch_thresh
        self.yaw_thresh = yaw_thresh
        self.ear_time_thresh = ear_time_thresh
        self.gaze_time_thresh = gaze_time_thresh
        self.pose_time_thresh = pose_time_thresh

        self.last_time_eye_opened = t_now
        self.last_time_looked_ahead = t_now
        self.last_time_attended = t_now
        self.prev_time = t_now

        self.closure_time = 0
        self.not_look_ahead_time = 0
        self.distracted_time = 0
        self.eye_closure_counter = 0

        self.verbose = verbose

    def detect_eye_state(self, eye_img):
        """
        Predicts whether the eye is open or closed using the CNN model.
        """
        if eye_img is None or eye_img.size == 0:
            return 1  # Assume open eyes if detection fails

        eye_img = eye_img.astype("float32") / 255.0  # Normalize
        eye_img = eye_img.reshape(1, 64, 64, 1)  # Reshape to (1, 64, 64, 1)
        prediction = cnn_model.predict(eye_img)
        eye_state = 1 if prediction > 0.5 else 0  # Open (1) or Closed (0)
        return eye_state

    def eval_scores(
        self, t_now, cnn_eye_state, gaze_score, head_roll, head_pitch, head_yaw
    ):
        """
        Evaluate the driver's state of attention based on CNN-predicted eye closure, gaze, and head pose.
        """
        asleep = False
        looking_away = False
        distracted = False

        if self.closure_time >= self.ear_time_thresh:
            asleep = True

        if self.not_look_ahead_time >= self.gaze_time_thresh:
            looking_away = True

        if self.distracted_time >= self.pose_time_thresh:
            distracted = True

        if cnn_eye_state == 0:  # CNN detects closed eyes
            self.closure_time = t_now - self.last_time_eye_opened
        else:
            self.last_time_eye_opened = t_now
            self.closure_time = 0.0

        if gaze_score is not None and gaze_score > self.gaze_thresh:
            self.not_look_ahead_time = t_now - self.last_time_looked_ahead
        else:
            self.last_time_looked_ahead = t_now
            self.not_look_ahead_time = 0.0

        if (
            abs(head_roll) > self.roll_thresh
            or abs(head_pitch) > self.pitch_thresh
            or abs(head_yaw) > self.yaw_thresh
        ):
            self.distracted_time = t_now - self.last_time_attended
        else:
            self.last_time_attended = t_now
            self.distracted_time = 0.0

        if self.verbose:
            print(
                f"Eye Closed: {asleep}\tLooking Away: {looking_away}\tDistracted: {distracted}"
            )

        return asleep, looking_away, distracted

    def get_PERCLOS(self, t_now, fps, cnn_eye_state):
        """
        Compute the PERCLOS (Percentage of Eye Closure) score over a given time period.
        """
        delta = t_now - self.prev_time
        tired = False

        all_frames_numbers_in_perclos_duration = int(self.PERCLOS_TIME_PERIOD * fps)

        if cnn_eye_state == 0:
            self.eye_closure_counter += 1

        perclos_score = self.eye_closure_counter / all_frames_numbers_in_perclos_duration

        if perclos_score >= self.perclos_thresh:
            tired = True

        if delta >= self.PERCLOS_TIME_PERIOD:
            self.eye_closure_counter = 0
            self.prev_time = t_now

        return tired, perclos_score
