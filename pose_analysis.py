import numpy as np
from ultralytics import YOLO
import cv2
import mediapipe as mp
import subprocess
import time

# Load YOLO Pose Detection model
model = YOLO("yolov8n-pose.pt", verbose=False)  # Adjust for accuracy if needed

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)

def detect_activity(keypoints, face_landmarks, previous_keypoints=None):
    """
    Detect activity based on pose keypoints and facial landmarks.
    :param keypoints: A numpy array of shape (16, 2) representing (x, y) keypoints.
    :param face_landmarks: A list of facial landmarks for detecting lip movement.
    :param previous_keypoints: A numpy array of shape (16, 2) representing keypoints from the previous frame.
    :return: Detected activity as a string.
    """
    def distance(pt1, pt2):
        return np.linalg.norm(np.array(pt1) - np.array(pt2))

    # Keypoint indices
    NOSE, L_SHOULDER, R_SHOULDER, L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANKLE, R_ANKLE, L_WRIST, R_WRIST = (
        0, 5, 6, 11, 12, 13, 14, 15, 16, 9, 10
    )

    if keypoints is None or len(keypoints) == 0:
        return "None"

    # Extract important keypoints
    nose = keypoints[NOSE]
    left_shoulder = keypoints[L_SHOULDER]
    right_shoulder = keypoints[R_SHOULDER]
    left_hip = keypoints[L_HIP]
    right_hip = keypoints[R_HIP]
    left_knee = keypoints[L_KNEE]
    right_knee = keypoints[R_KNEE]
    left_ankle = keypoints[L_ANKLE]
    right_ankle = keypoints[R_ANKLE]
    left_wrist = keypoints[L_WRIST]
    right_wrist = keypoints[R_WRIST]

    # Calculate distances and movement metrics
    torso_length = distance(nose, (left_hip + right_hip) / 2)
    arm_movement = distance(left_wrist, left_shoulder) + distance(right_wrist, right_shoulder)
    leg_movement = distance(left_knee, left_ankle) + distance(right_knee, right_ankle)
    total_movement = arm_movement + leg_movement

    # Detect mouth movement for talking
    is_talking = False
    if face_landmarks:
        upper_lip = face_landmarks[13]  # MediaPipe index for upper lip
        lower_lip = face_landmarks[14]  # MediaPipe index for lower lip
        lip_distance = distance(upper_lip, lower_lip)
        is_talking = lip_distance > 5  # Define a suitable threshold

    # Detect activities
    if is_talking:
        return "Talking"
    elif total_movement > torso_length * 1.2:  # Detect dancing based on large synchronized movement
        return "Dancing"
    elif leg_movement > torso_length * 0.3:
        return "Running"
    elif arm_movement < torso_length * 0.2 and leg_movement < torso_length * 0.1:
        return "Standing"
    elif leg_movement > 0.1 and leg_movement < torso_length * 0.3:
        return "Walking"
    else:
        return "Other Activity"

def process_gif(gif_path, confidence_score):
    """
    Detect keypoints in a GIF and classify activities.
    :param gif_path: Path to the input GIF.
    """
    cap = cv2.VideoCapture(gif_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_path = f"annotated_{gif_path}"
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    previous_keypoints = None
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Perform pose detection
        results = model.predict(source=frame, conf=0.5, save=False, verbose=False)
        if results is None or len(results) == 0 or not hasattr(results[0], 'keypoints') or results[0].keypoints is None:
            continue
        for result in results:
            for pose in result.keypoints.xy:  # Loop through detected people
                keypoints = np.array(pose)

                # Detect facial landmarks using MediaPipe
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                face_results = face_mesh.process(rgb_frame)

                face_landmarks = []
                if face_results.multi_face_landmarks:
                    for face_landmark in face_results.multi_face_landmarks:
                        for landmark in face_landmark.landmark:
                            x = int(landmark.x * frame.shape[1])
                            y = int(landmark.y * frame.shape[0])
                            face_landmarks.append((x, y))

                activity = detect_activity(keypoints, face_landmarks, previous_keypoints)

                # Annotate the activity on the frame
                cv2.putText(frame, activity, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

                # Draw keypoints
                for x, y in keypoints:
                    cv2.circle(frame, (int(x), int(y)), 5, (255, 0, 0), -1)

                # Draw facial landmarks
                for x, y in face_landmarks:
                    cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

                previous_keypoints = keypoints  # Update for temporal analysis

        out.write(frame)
        # cv2.imshow("Pose Activity Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
            break

    out.release()
    repaired_path = f"repaired_{output_path}"

    # Define the ffmpeg command
    command = [
        'ffmpeg', '-y',
        '-i', output_path,
        '-c:v', 'libx264',
        '-c:a', 'aac',
        repaired_path
    ]

    if retry_file_access(output_path):
        # Run the command
        try:
            subprocess.run(command, check=True)
            print("Video processed successfully")
        except subprocess.CalledProcessError as e:
            print(f"Error occurred: {e}")

    return repaired_path

def retry_file_access(file_path, retries=3, delay=2):
    for i in range(retries):
        try:
            # Try accessing the file
            with open(file_path, 'rb'):
                return True
        except IOError:
            print(f"File is not ready yet. Retrying... {i+1}/{retries}")
            time.sleep(delay)
    print("File is not accessible after multiple retries.")
    return False
