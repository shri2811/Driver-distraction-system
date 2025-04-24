import cv2
import numpy as np
from numpy import linalg as LA
from tensorflow.keras.models import load_model
from numpy.linalg import norm


# Load the trained CNN model
MODEL_PATH = "bestModel100.h5"  # Update this if needed
cnn_model = load_model(MODEL_PATH)
print("Model loaded successfully. Summary:")
cnn_model.summary()


# Define class mapping
CLASS_NAMES = {1: "Open Eye", 0: "Closed Eye"}
MOUTH_LANDMARKS = [78, 191, 80, 95, 88, 178, 87, 14, 317, 402, 310, 415, 311, 324]

class EyeDetector:
    def __init__(self, show_processing: bool = False):
        """
        Eye detector class using a CNN model for eye closure detection.
        """
        self.show_processing = show_processing

    def preprocess_eye(self, eye_img):
        """
        Preprocesses the eye image for CNN input.
        """
        eye_img = cv2.cvtColor(eye_img, cv2.COLOR_BGR2GRAY)
        eye_img = cv2.resize(eye_img, (64, 64))  # Resize to match CNN input
        eye_img = eye_img.astype("float32") / 255.0  # Normalize
        eye_img = eye_img.reshape(1, 64, 64, 1)  # Reshape to (1, 64, 64, 1)
        return eye_img

    def detect_eye_state(self, eye_img):
        """
        Predicts whether the eye is open or closed using the CNN model.
        """
        if eye_img is None or eye_img.size == 0:
            print("Eye image not found. Defaulting to open.")
            return 0  # Assume open eyes if detection fails
        eye_img = cv2.cvtColor(eye_img, cv2.COLOR_BGR2GRAY) if len(eye_img.shape) == 3 else eye_img  
        eye_img = cv2.resize(eye_img, (64, 64))  # Resize
        eye_img = eye_img.astype("float32") / 255.0  # Normalize
        eye_img = np.expand_dims(eye_img, axis=(0, -1))

        prediction = cnn_model.predict(eye_img,verbose=0)
        return 1 if prediction > 0.3 else 0 

    def get_EAR(self, frame, landmarks):
        """
        Extracts eye regions, applies CNN, and returns binary eye state.
        """
        left_eye = self.extract_eye_region(frame, landmarks, is_left=True)
        right_eye = self.extract_eye_region(frame, landmarks, is_left=False)
        
        left_eye_state = self.detect_eye_state(left_eye)
        right_eye_state = self.detect_eye_state(right_eye)
        
        return (left_eye_state + right_eye_state) / 2  # Average of both eyes
    
    def extract_eye_region(self, frame, landmarks, is_left=True):
        """Extracts eye region from normalized landmarks and scales to image size."""
        eye_lms = [33, 133, 160, 144, 158, 153] if is_left else [362, 263, 385, 380, 387, 373]

        # Convert normalized landmarks to pixel coordinates
        h, w = frame.shape[:2]  # Image height and width
        x_min = int(min([landmarks[i][0] * w for i in eye_lms]))
        y_min = int(min([landmarks[i][1] * h for i in eye_lms]))
        x_max = int(max([landmarks[i][0] * w for i in eye_lms]))
        y_max = int(max([landmarks[i][1] * h for i in eye_lms]))

        # Debugging
        #print(f"Eye ({'Left' if is_left else 'Right'}) Region: x_min={x_min}, y_min={y_min}, x_max={x_max}, y_max={y_max}")

        # Ensure valid ROI (Region of Interest)
        if x_min >= x_max or y_min >= y_max:
            print("Invalid eye region detected! Returning None.")
            return None

        # Extract the eye region
        eye_img = frame[y_min:y_max, x_min:x_max]

        # Show extracted eye region
        #if eye_img is not None and eye_img.size > 0:
           # cv2.imshow(f"Extracted Eye ({'Left' if is_left else 'Right'})", cv2.resize(eye_img, (100, 100)))
            #cv2.waitKey(1)
        #else:
           # print("Eye image extraction failed!")

        return eye_img if eye_img is not None and eye_img.size > 0 else None

    def _calc_1eye_score(self, landmarks, eye_lms_nums, eye_iris_num, frame_size, frame):
        """Gets each eye score and its picture."""
        iris = landmarks[eye_iris_num, :2]

        eye_x_min = landmarks[eye_lms_nums, 0].min()
        eye_y_min = landmarks[eye_lms_nums, 1].min()
        eye_x_max = landmarks[eye_lms_nums, 0].max()
        eye_y_max = landmarks[eye_lms_nums, 1].max()

        eye_center = np.array(((eye_x_min + eye_x_max) / 2, (eye_y_min + eye_y_max) / 2))
        eye_gaze_score = LA.norm(iris - eye_center) / eye_center[0]

        eye_x_min_frame = int(eye_x_min * frame_size[0])
        eye_y_min_frame = int(eye_y_min * frame_size[1])
        eye_x_max_frame = int(eye_x_max * frame_size[0])
        eye_y_max_frame = int(eye_y_max * frame_size[1])

        eye = frame[eye_y_min_frame:eye_y_max_frame, eye_x_min_frame:eye_x_max_frame]

        return eye_gaze_score, eye if eye.size > 0 else None
    
    def show_eye_keypoints(self, color_frame, landmarks, frame_size):
   
        for n in [33, 133, 160, 144, 158, 153, 362, 263, 385, 380, 387, 373]:
            x = int(landmarks[n, 0] * frame_size[0])
            y = int(landmarks[n, 1] * frame_size[1])
            cv2.circle(color_frame, (x, y), 2, (0, 0, 255), -1)
    
    # Draw iris keypoints
        for iris in [468, 473]:  # Left and Right iris indices
            x = int(landmarks[iris, 0] * frame_size[0])
            y = int(landmarks[iris, 1] * frame_size[1])
            cv2.circle(color_frame, (x, y), 3, (255, 255, 255), -1)

        return color_frame

# Mouth landmarks from Mediapipe Face Mesh


    def calculate_MAR(self,landmarks):   
        A = norm(landmarks[13] - landmarks[14])  # Upper lip to lower lip
        B = norm(landmarks[41] - landmarks[270])  # Left to right mouth width
        mar = A / B
        return mar


    def get_Gaze_Score(self, frame, landmarks, frame_size):
        """
        Computes the average Gaze Score for the eyes
        """
        left_gaze_score, left_eye = self._calc_1eye_score(
            landmarks, [33, 133, 160, 144, 158, 153], 468, frame_size, frame
        )
        right_gaze_score, right_eye = self._calc_1eye_score(
            landmarks, [362, 263, 385, 380, 387, 373], 473, frame_size, frame
        )

        avg_gaze_score = (left_gaze_score + right_gaze_score) / 2

        #if self.show_processing and (left_eye is not None) and (right_eye is not None):
            #left_eye = cv2.resize(left_eye, (100, 100))
            #right_eye = cv2.resize(right_eye, (100, 100))
            #cv2.imshow("Left Eye", left_eye)
            #cv2.imshow("Right Eye", right_eye)

        return avg_gaze_score
