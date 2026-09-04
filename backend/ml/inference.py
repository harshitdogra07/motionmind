import os
import joblib
import pandas as pd
from ml.extract_keypoints import extract_video_keypoints
from .feature_engineering import extract_features_by_skill

class MultiSkillEvaluator:
    def __init__(self, model_dir=None):
        if model_dir is None:
            model_dir = os.path.dirname(os.path.abspath(__file__))

        # Skill dictionary mapping to model files and mistake labels
        self.skills = {
            "squat": {
                "q_model": joblib.load(os.path.join(model_dir, "squat_quality.pkl")),
                "m_model": joblib.load(os.path.join(model_dir, "squat_mistake.pkl")),
                "mistakes": {
                    0: "Good Form - Proper depth and back posture.",
                    1: "Shallow Depth - You are not squatting low enough.",
                    2: "Back Rounded - Torso leaning too far forward.",
                    3: "Unstable Knees - Knees caving inward during ascent."
                }
            },
            "boxing": {
                "q_model": joblib.load(os.path.join(model_dir, "boxing_quality.pkl")),
                "m_model": joblib.load(os.path.join(model_dir, "boxing_mistake.pkl")),
                "mistakes": {
                    0: "Good Form - Clean extension and tight guard.",
                    1: "Elbow Flared - Elbow flaring outward during extension.",
                    2: "No Hip Rotation - Lacking kinetic chain hip turn.",
                    3: "Guard Dropped - Off-hand dropping below jaw level."
                }
            },
            "basketball": {
                "q_model": joblib.load(os.path.join(model_dir, "basketball_quality.pkl")),
                "m_model": joblib.load(os.path.join(model_dir, "basketball_mistake.pkl")),
                "mistakes": {
                    0: "Good Form - Excellent set point and release arc.",
                    1: "Low Elbow - Set point is too low before release.",
                    2: "No Leg Drive - Shooting using only upper body force.",
                    3: "Poor Release Angle - Flat shot arc trajectory."
                }
            }
        }

    def evaluate(self, video_path: str, skill_name: str = "squat"):
        # Normalize skill string
        key = "squat"
        if "box" in skill_name.lower():
            key = "boxing"
        elif "basket" in skill_name.lower():
            key = "basketball"

        skill_config = self.skills[key]

        # 1. Keypoint Extraction
        df_keypoints = extract_video_keypoints(video_path)
        if df_keypoints.empty:
            return {"error": "No human pose detected in video."}

        if key == "squat":
            try:
                from ml.feature_engineering import format_sequence_3d
                from ml.hybrid_engine import hybrid_squat_evaluation, GOLDEN_SQUAT
            except ImportError:
                from feature_engineering import format_sequence_3d
                from hybrid_engine import hybrid_squat_evaluation, GOLDEN_SQUAT
                
            user_seq = format_sequence_3d(df_keypoints)
            dtw_score, dtw_feedback = hybrid_squat_evaluation(user_seq, GOLDEN_SQUAT)
            
            return {
                "skill": key.capitalize(),
                "overall_score": dtw_score,
                "detected_mistake": dtw_feedback[0] if dtw_feedback else "Good Squat",
                "biomechanics": {"min_knee_angle": 0} # Placeholder to avoid breaking UI that expects biomechanics object
            }

        # 2. Extract Skill Features (for non-squat XGBoost fallback)
        features = extract_features_by_skill(df_keypoints, key)
        df_features = pd.DataFrame([features])

        # 3. Model Predictions
        raw_score = float(skill_config["q_model"].predict(df_features)[0])
        score = int(max(0, min(100, round(raw_score))))

        mistake_id = int(skill_config["m_model"].predict(df_features)[0])
        mistake_desc = skill_config["mistakes"].get(mistake_id, "Form discrepancy detected.")

        return {
            "skill": key.capitalize(),
            "overall_score": score,
            "detected_mistake": mistake_desc,
            "biomechanics": features
        }

    def analyze_frame(self, keypoints: dict, action_type: str):
        from feature_engineering import calculate_angle
        key = "squat"
        if "box" in action_type.lower() or "jab" in action_type.lower():
            key = "boxing"
        elif "basket" in action_type.lower() or "shoot" in action_type.lower():
            key = "basketball"
            
        score = 100
        feedback = []
        
        if 'x_11' not in keypoints:
            return {"action": action_type, "score": 0, "feedback": ["No human pose detected in video."], "keypoints": keypoints}

        if key == "squat":
            hip = [keypoints.get('x_23', 0), keypoints.get('y_23', 0)]
            knee = [keypoints.get('x_25', 0), keypoints.get('y_25', 0)]
            ankle = [keypoints.get('x_27', 0), keypoints.get('y_27', 0)]
            shoulder = [keypoints.get('x_11', 0), keypoints.get('y_11', 0)]
            vert = [hip[0], hip[1] - 0.5]
            
            knee_angle = calculate_angle(hip, knee, ankle)
            back_tilt = calculate_angle(shoulder, hip, vert)
            
            if knee_angle > 140:
                feedback.append("Standing - bend knees to squat.")
            elif knee_angle > 90:
                feedback.append("Shallow Depth - squat lower.")
                score -= 15
            else:
                feedback.append("Good depth!")
                
            if back_tilt > 35:
                feedback.append("Back Rounded - keep torso upright.")
                score -= 20
                
        elif key == "boxing":
            shoulder = [keypoints.get('x_11', 0), keypoints.get('y_11', 0)]
            elbow = [keypoints.get('x_13', 0), keypoints.get('y_13', 0)]
            wrist = [keypoints.get('x_15', 0), keypoints.get('y_15', 0)]
            nose = [keypoints.get('x_0', 0), keypoints.get('y_0', 0)]
            r_wrist = [keypoints.get('x_16', 0), keypoints.get('y_16', 0)]
            
            elbow_angle = calculate_angle(shoulder, elbow, wrist)
            guard_drop = ((r_wrist[0]-nose[0])**2 + (r_wrist[1]-nose[1])**2)**0.5
            
            if elbow_angle < 130:
                feedback.append("Elbow Flared - extend your arm fully.")
                score -= 20
            else:
                feedback.append("Good arm extension.")
                
            if guard_drop > 0.15:
                feedback.append("Guard Dropped - keep off-hand near face.")
                score -= 15

        elif key == "basketball":
            shoulder = [keypoints.get('x_11', 0), keypoints.get('y_11', 0)]
            elbow = [keypoints.get('x_13', 0), keypoints.get('y_13', 0)]
            wrist = [keypoints.get('x_15', 0), keypoints.get('y_15', 0)]
            
            elbow_angle = calculate_angle(shoulder, elbow, wrist)
            wrist_height_rel = shoulder[1] - wrist[1] # Y is downwards
            
            if elbow_angle < 70 or elbow_angle > 110:
                feedback.append("Elbow not at 90 degrees during gather.")
                score -= 15
            
            if wrist_height_rel < 0:
                feedback.append("Low Elbow - keep the ball high.")
                score -= 20
            elif wrist_height_rel > 0.1:
                feedback.append("Good wrist release height!")
                
        return {
            "action": action_type,
            "score": score,
            "feedback": feedback,
            "keypoints": keypoints
        }

# Quick Test
if __name__ == "__main__":
    evaluator = MultiSkillEvaluator()
    print("MultiSkillEvaluator loaded successfully!")