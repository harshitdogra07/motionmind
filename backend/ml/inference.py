import os
import joblib
import pandas as pd
from extract_keypoints import extract_video_keypoints
from feature_engineering import extract_features_by_skill

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

        # 2. Extract Skill Features
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

# Quick Test
if __name__ == "__main__":
    evaluator = MultiSkillEvaluator()
    print("MultiSkillEvaluator loaded successfully!")