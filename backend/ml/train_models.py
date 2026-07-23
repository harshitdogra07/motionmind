import numpy as np
import pandas as pd
from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import mean_absolute_error, accuracy_score
from sklearn.model_selection import LeaveOneOut
import joblib
import os

def train_and_export():
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # =========================================================================
    # SKILL 1: SQUAT
    # Features: min_knee_angle, max_knee_angle, knee_rom, min_hip_angle,
    #           max_back_tilt, knee_wobble_std, rep_duration_frames
    # Mistakes: 0=Good, 1=Shallow Depth, 2=Back Rounded, 3=Unstable Knees
    # =========================================================================
    squat_X = pd.DataFrame([
        {'min_knee_angle': 78,  'max_knee_angle': 175, 'knee_rom': 95, 'min_hip_angle': 82,  'max_back_tilt': 15, 'knee_wobble_std': 1.2, 'rep_duration_frames': 45},  # Good
        {'min_knee_angle': 76,  'max_knee_angle': 176, 'knee_rom': 100,'min_hip_angle': 80,  'max_back_tilt': 14, 'knee_wobble_std': 1.0, 'rep_duration_frames': 48},  # Good
        {'min_knee_angle': 110, 'max_knee_angle': 170, 'knee_rom': 60, 'min_hip_angle': 105, 'max_back_tilt': 18, 'knee_wobble_std': 1.5, 'rep_duration_frames': 30},  # Shallow
        {'min_knee_angle': 115, 'max_knee_angle': 168, 'knee_rom': 53, 'min_hip_angle': 110, 'max_back_tilt': 20, 'knee_wobble_std': 1.3, 'rep_duration_frames': 28},  # Shallow
        {'min_knee_angle': 75,  'max_knee_angle': 172, 'knee_rom': 98, 'min_hip_angle': 80,  'max_back_tilt': 45, 'knee_wobble_std': 3.8, 'rep_duration_frames': 50},  # Back Rounded
        {'min_knee_angle': 77,  'max_knee_angle': 174, 'knee_rom': 97, 'min_hip_angle': 78,  'max_back_tilt': 50, 'knee_wobble_std': 4.2, 'rep_duration_frames': 52},  # Back Rounded
        {'min_knee_angle': 80,  'max_knee_angle': 178, 'knee_rom': 92, 'min_hip_angle': 85,  'max_back_tilt': 12, 'knee_wobble_std': 6.5, 'rep_duration_frames': 42},  # Unstable Knees
        {'min_knee_angle': 82,  'max_knee_angle': 176, 'knee_rom': 94, 'min_hip_angle': 83,  'max_back_tilt': 16, 'knee_wobble_std': 7.1, 'rep_duration_frames': 40},  # Unstable Knees
    ])
    squat_quality = np.array([95, 97, 60, 55, 68, 62, 72, 65])
    squat_mistake = np.array([0, 0, 1, 1, 2, 2, 3, 3])

    squat_q_model = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1)
    squat_q_model.fit(squat_X, squat_quality)
    squat_m_model = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, use_label_encoder=False, eval_metric='mlogloss')
    squat_m_model.fit(squat_X, squat_mistake)

    # Evaluate
    sq_q_pred = squat_q_model.predict(squat_X)
    sq_m_pred = squat_m_model.predict(squat_X)
    sq_mae = mean_absolute_error(squat_quality, sq_q_pred)
    sq_acc = accuracy_score(squat_mistake, sq_m_pred) * 100
    print(f"[SQUAT]   Quality MAE: {sq_mae:.2f} | Mistake Accuracy: {sq_acc:.1f}%")

    joblib.dump(squat_q_model, os.path.join(current_dir, "squat_quality.pkl"))
    joblib.dump(squat_m_model, os.path.join(current_dir, "squat_mistake.pkl"))

    # =========================================================================
    # SKILL 2: BOXING (Jab)
    # Features: min_elbow_angle, max_elbow_angle, elbow_rom, max_shoulder_angle,
    #           max_wrist_speed, hip_rotation_range, guard_drop_max
    # Mistakes: 0=Good, 1=Elbow Flared, 2=No Hip Rotation, 3=Guard Dropped
    # =========================================================================
    boxing_X = pd.DataFrame([
        {'min_elbow_angle': 165, 'max_elbow_angle': 170, 'elbow_rom': 85,  'max_shoulder_angle': 78, 'max_wrist_speed': 0.12, 'hip_rotation_range': 25, 'guard_drop_max': 0.08},  # Good
        {'min_elbow_angle': 168, 'max_elbow_angle': 172, 'elbow_rom': 88,  'max_shoulder_angle': 80, 'max_wrist_speed': 0.14, 'hip_rotation_range': 28, 'guard_drop_max': 0.07},  # Good
        {'min_elbow_angle': 140, 'max_elbow_angle': 155, 'elbow_rom': 60,  'max_shoulder_angle': 95, 'max_wrist_speed': 0.09, 'hip_rotation_range': 22, 'guard_drop_max': 0.10},  # Elbow Flared
        {'min_elbow_angle': 135, 'max_elbow_angle': 150, 'elbow_rom': 55,  'max_shoulder_angle': 100,'max_wrist_speed': 0.08, 'hip_rotation_range': 20, 'guard_drop_max': 0.11},  # Elbow Flared
        {'min_elbow_angle': 162, 'max_elbow_angle': 168, 'elbow_rom': 80,  'max_shoulder_angle': 75, 'max_wrist_speed': 0.07, 'hip_rotation_range': 8,  'guard_drop_max': 0.09},  # No Hip Rotation
        {'min_elbow_angle': 160, 'max_elbow_angle': 166, 'elbow_rom': 78,  'max_shoulder_angle': 72, 'max_wrist_speed': 0.06, 'hip_rotation_range': 6,  'guard_drop_max': 0.10},  # No Hip Rotation
        {'min_elbow_angle': 164, 'max_elbow_angle': 169, 'elbow_rom': 82,  'max_shoulder_angle': 76, 'max_wrist_speed': 0.11, 'hip_rotation_range': 24, 'guard_drop_max': 0.22},  # Guard Dropped
        {'min_elbow_angle': 166, 'max_elbow_angle': 171, 'elbow_rom': 84,  'max_shoulder_angle': 79, 'max_wrist_speed': 0.13, 'hip_rotation_range': 26, 'guard_drop_max': 0.25},  # Guard Dropped
    ])
    boxing_quality = np.array([94, 96, 62, 58, 70, 65, 68, 63])
    boxing_mistake = np.array([0, 0, 1, 1, 2, 2, 3, 3])

    boxing_q_model = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1)
    boxing_q_model.fit(boxing_X, boxing_quality)
    boxing_m_model = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, use_label_encoder=False, eval_metric='mlogloss')
    boxing_m_model.fit(boxing_X, boxing_mistake)

    bx_q_pred = boxing_q_model.predict(boxing_X)
    bx_m_pred = boxing_m_model.predict(boxing_X)
    bx_mae = mean_absolute_error(boxing_quality, bx_q_pred)
    bx_acc = accuracy_score(boxing_mistake, bx_m_pred) * 100
    print(f"[BOXING]  Quality MAE: {bx_mae:.2f} | Mistake Accuracy: {bx_acc:.1f}%")

    joblib.dump(boxing_q_model, os.path.join(current_dir, "boxing_quality.pkl"))
    joblib.dump(boxing_m_model, os.path.join(current_dir, "boxing_mistake.pkl"))

    # =========================================================================
    # SKILL 3: BASKETBALL SHOOTING
    # Features: min_elbow_angle, max_elbow_angle, elbow_rom, max_shoulder_angle,
    #           min_knee_angle, max_wrist_height, max_release_angle
    # Mistakes: 0=Good, 1=Low Elbow, 2=No Leg Drive, 3=Poor Release Angle
    # =========================================================================
    bball_X = pd.DataFrame([
        {'min_elbow_angle': 85,  'max_elbow_angle': 165, 'elbow_rom': 80,  'max_shoulder_angle': 160, 'min_knee_angle': 100, 'max_wrist_height': 0.15, 'max_release_angle': 55},  # Good
        {'min_elbow_angle': 82,  'max_elbow_angle': 168, 'elbow_rom': 86,  'max_shoulder_angle': 165, 'min_knee_angle': 95,  'max_wrist_height': 0.18, 'max_release_angle': 58},  # Good
        {'min_elbow_angle': 120, 'max_elbow_angle': 145, 'elbow_rom': 25,  'max_shoulder_angle': 130, 'min_knee_angle': 105, 'max_wrist_height': 0.08, 'max_release_angle': 45},  # Low Elbow
        {'min_elbow_angle': 125, 'max_elbow_angle': 140, 'elbow_rom': 15,  'max_shoulder_angle': 125, 'min_knee_angle': 100, 'max_wrist_height': 0.06, 'max_release_angle': 42},  # Low Elbow
        {'min_elbow_angle': 88,  'max_elbow_angle': 162, 'elbow_rom': 74,  'max_shoulder_angle': 155, 'min_knee_angle': 160, 'max_wrist_height': 0.12, 'max_release_angle': 50},  # No Leg Drive
        {'min_elbow_angle': 90,  'max_elbow_angle': 160, 'elbow_rom': 70,  'max_shoulder_angle': 150, 'min_knee_angle': 165, 'max_wrist_height': 0.10, 'max_release_angle': 48},  # No Leg Drive
        {'min_elbow_angle': 84,  'max_elbow_angle': 164, 'elbow_rom': 80,  'max_shoulder_angle': 158, 'min_knee_angle': 98,  'max_wrist_height': 0.14, 'max_release_angle': 25},  # Poor Release Angle
        {'min_elbow_angle': 86,  'max_elbow_angle': 166, 'elbow_rom': 80,  'max_shoulder_angle': 162, 'min_knee_angle': 102, 'max_wrist_height': 0.16, 'max_release_angle': 20},  # Poor Release Angle
    ])
    bball_quality = np.array([93, 96, 58, 52, 72, 68, 65, 60])
    bball_mistake = np.array([0, 0, 1, 1, 2, 2, 3, 3])

    bball_q_model = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1)
    bball_q_model.fit(bball_X, bball_quality)
    bball_m_model = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, use_label_encoder=False, eval_metric='mlogloss')
    bball_m_model.fit(bball_X, bball_mistake)

    bb_q_pred = bball_q_model.predict(bball_X)
    bb_m_pred = bball_m_model.predict(bball_X)
    bb_mae = mean_absolute_error(bball_quality, bb_q_pred)
    bb_acc = accuracy_score(bball_mistake, bb_m_pred) * 100
    print(f"[BBALL]   Quality MAE: {bb_mae:.2f} | Mistake Accuracy: {bb_acc:.1f}%")

    joblib.dump(bball_q_model, os.path.join(current_dir, "basketball_quality.pkl"))
    joblib.dump(bball_m_model, os.path.join(current_dir, "basketball_mistake.pkl"))

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n========== TRAINING SUMMARY ==========")
    print(f"  Squat      ->  Quality MAE: {sq_mae:.2f},  Mistake Acc: {sq_acc:.1f}%")
    print(f"  Boxing     ->  Quality MAE: {bx_mae:.2f},  Mistake Acc: {bx_acc:.1f}%")
    print(f"  Basketball ->  Quality MAE: {bb_mae:.2f},  Mistake Acc: {bb_acc:.1f}%")
    print("=======================================")
    print("All 6 models exported successfully!")


if __name__ == "__main__":
    train_and_export()
