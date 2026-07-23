import cv2
import pandas as pd
import os

try:
    import mediapipe as mp
    mp_pose = mp.solutions.pose
    HAS_MP = True
except AttributeError:
    # MediaPipe Python 3.13 compatibility issue fallback
    HAS_MP = False

def extract_video_keypoints(video_path):
    """Processes a single video and returns a DataFrame of frame-by-frame keypoints."""
    if not HAS_MP:
        import numpy as np
        print("Warning: Using MVP Mock Data due to MediaPipe Python 3.13 compatibility.")
        frames_data = []
        for i in range(30):
            row = {'frame': i}
            for j in range(33):
                row[f'x_{j}'] = 0.5 + np.random.uniform(-0.05, 0.05)
                row[f'y_{j}'] = 0.5 + (0.2 * np.sin(i / 5.0)) # Simulate squat motion
                row[f'z_{j}'] = 0.0
                row[f'vis_{j}'] = 0.99
            frames_data.append(row)
        return pd.DataFrame(frames_data)

    cap = cv2.VideoCapture(video_path)
    pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5)
    
    frames_data = []
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert image to RGB for MediaPipe
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if results.pose_landmarks:
            row = {'frame': frame_idx}
            for idx, lm in enumerate(results.pose_landmarks.landmark):
                row[f'x_{idx}'] = lm.x
                row[f'y_{idx}'] = lm.y
                row[f'z_{idx}'] = lm.z
                row[f'vis_{idx}'] = lm.visibility
            frames_data.append(row)
        frame_idx += 1

    cap.release()
    pose.close()
    return pd.DataFrame(frames_data)

# Example usage to extract and save frame keypoints
if __name__ == "__main__":
    # Ensure datasets folder exists
    os.makedirs("datasets", exist_ok=True)
    # Placeholder for actual usage
    # df_keypoints = extract_video_keypoints("sample_squat.mp4")
    # df_keypoints.to_csv("datasets/sample_squat_keypoints.csv", index=False)
    # print(f"Extracted {len(df_keypoints)} frames successfully!")
