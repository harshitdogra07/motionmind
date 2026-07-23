import numpy as np
import pandas as pd

def calculate_angle(a, b, c):
    """Calculates 2D angle (degrees) at joint B given coordinates A, B, C."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360.0 - angle
    return angle

def extract_features_by_skill(df_keypoints: pd.DataFrame, skill_name: str) -> dict:
    """Routes feature calculation depending on the skill selected."""
    skill = skill_name.lower()

    if "squat" in skill:
        knee_angles, hip_angles, back_tilts = [], [], []
        for _, row in df_keypoints.iterrows():
            hip = [row['x_23'], row['y_23']]
            knee = [row['x_25'], row['y_25']]
            ankle = [row['x_27'], row['y_27']]
            shoulder = [row['x_11'], row['y_11']]
            vert = [row['x_23'], row['y_23'] - 0.5]

            knee_angles.append(calculate_angle(hip, knee, ankle))
            hip_angles.append(calculate_angle(shoulder, hip, knee))
            back_tilts.append(calculate_angle(shoulder, hip, vert))

        knee_angles, hip_angles, back_tilts = np.array(knee_angles), np.array(hip_angles), np.array(back_tilts)

        return {
            'min_knee_angle': float(np.min(knee_angles)),
            'max_knee_angle': float(np.max(knee_angles)),
            'knee_rom': float(np.max(knee_angles) - np.min(knee_angles)),
            'min_hip_angle': float(np.min(hip_angles)),
            'max_back_tilt': float(np.max(back_tilts)),
            'knee_wobble_std': float(np.std(np.diff(knee_angles))) if len(knee_angles) > 1 else 0.0,
            'rep_duration_frames': len(df_keypoints)
        }

    elif "box" in skill or "jab" in skill:
        elbow_angles, shoulder_angles, wrist_speeds, guard_drops = [], [], [], []
        prev_wrist = None

        for _, row in df_keypoints.iterrows():
            shoulder = [row['x_11'], row['y_11']]
            elbow = [row['x_13'], row['y_13']]
            wrist = [row['x_15'], row['y_15']]
            hip = [row['x_23'], row['y_23']]

            elbow_angles.append(calculate_angle(shoulder, elbow, wrist))
            shoulder_angles.append(calculate_angle(hip, shoulder, elbow))
            
            # Guard Drop: distance between non-punching wrist (right) and nose (index 0)
            nose = [row['x_0'], row['y_0']]
            r_wrist = [row['x_16'], row['y_16']]
            guard_drops.append(np.linalg_norm(np.array(r_wrist) - np.array(nose)))

            if prev_wrist is not None:
                speed = np.linalg_norm(np.array(wrist) - np.array(prev_wrist))
                wrist_speeds.append(speed)
            prev_wrist = wrist

        elbow_angles = np.array(elbow_angles)
        return {
            'min_elbow_angle': float(np.min(elbow_angles)),
            'max_elbow_angle': float(np.max(elbow_angles)),
            'elbow_rom': float(np.max(elbow_angles) - np.min(elbow_angles)),
            'max_shoulder_angle': float(np.max(shoulder_angles)),
            'max_wrist_speed': float(np.max(wrist_speeds)) if wrist_speeds else 0.1,
            'hip_rotation_range': 25.0,  # Fallback baseline
            'guard_drop_max': float(np.max(guard_drops))
        }

    elif "basket" in skill or "shoot" in skill:
        elbow_angles, shoulder_angles, knee_angles, wrist_heights = [], [], [], []
        for _, row in df_keypoints.iterrows():
            shoulder = [row['x_11'], row['y_11']]
            elbow = [row['x_13'], row['y_13']]
            wrist = [row['x_15'], row['y_15']]
            hip = [row['x_23'], row['y_23']]
            knee = [row['x_25'], row['y_25']]
            ankle = [row['x_27'], row['y_27']]

            elbow_angles.append(calculate_angle(shoulder, elbow, wrist))
            shoulder_angles.append(calculate_angle(hip, shoulder, elbow))
            knee_angles.append(calculate_angle(hip, knee, ankle))
            wrist_heights.append(row['y_11'] - row['y_15']) # Height relative to shoulder

        elbow_angles = np.array(elbow_angles)
        return {
            'min_elbow_angle': float(np.min(elbow_angles)),
            'max_elbow_angle': float(np.max(elbow_angles)),
            'elbow_rom': float(np.max(elbow_angles) - np.min(elbow_angles)),
            'max_shoulder_angle': float(np.max(shoulder_angles)),
            'min_knee_angle': float(np.min(knee_angles)),
            'max_wrist_height': float(np.max(wrist_heights)),
            'max_release_angle': 52.0  # Estimated release angle
        }

    else:
        raise ValueError(f"Unsupported skill: {skill_name}")
