import numpy as np
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
try:
    from .feature_engineering import calculate_3d_angle
except ImportError:
    from feature_engineering import calculate_3d_angle

def hybrid_squat_evaluation(user_sequence, golden_sequence):
    """
    user_sequence: NumPy array of shape (frames, 33, 3)
    golden_sequence: Same shape
    """
    if len(user_sequence) == 0 or len(golden_sequence) == 0:
        return 0, ["No valid sequence detected."]

    min_knee_angle = 180.0
    
    for frame in user_sequence:
        hip, knee, ankle = frame[23], frame[25], frame[27]
        # Avoid zero-length vectors if joints are completely zeroed out
        if np.linalg.norm(hip - knee) < 1e-5 or np.linalg.norm(ankle - knee) < 1e-5:
            continue
        angle = calculate_3d_angle(hip, knee, ankle)
        if angle < min_knee_angle:
            min_knee_angle = angle
            
    core_joints = [23, 24, 25, 26, 27, 28]
    
    user_flat = np.array([f[core_joints].flatten() for f in user_sequence])
    golden_flat = np.array([f[core_joints].flatten() for f in golden_sequence])
    
    distance, _ = fastdtw(user_flat, golden_flat, dist=euclidean)
    normalized_distance = distance / len(user_flat)
    
    raw_score = 100 * (1 - (normalized_distance / 1.5))
    final_score = max(0, min(100, raw_score))
    
    feedback = []
    
    if min_knee_angle > 100:
        final_score = min(final_score, 50)
        feedback.append(f"Shallow Squat: You only reached {int(min_knee_angle)}°. Drop hips below 90°.")
    elif min_knee_angle < 70:
        feedback.append(f"Deep Squat: Excellent depth achieved ({int(min_knee_angle)}°).")
    else:
        feedback.append(f"Good Depth: Parallel reached at {int(min_knee_angle)}°.")
        
    return int(final_score), feedback

def get_mock_golden_sequence(frames=30):
    seq = np.zeros((frames, 33, 3))
    for i in range(frames):
        seq[i, :, :] = 0.5
        progress = np.sin(np.pi * i / (frames - 1)) if frames > 1 else 0
        seq[i, 23, 1] += 0.2 * progress
        seq[i, 24, 1] += 0.2 * progress
        seq[i, 25, 2] -= 0.1 * progress
        seq[i, 26, 2] -= 0.1 * progress
    return seq

GOLDEN_SQUAT = get_mock_golden_sequence()

if __name__ == "__main__":
    # Test script to verify logic
    user_seq = get_mock_golden_sequence(35)
    user_seq += np.random.normal(0, 0.01, user_seq.shape)
    
    score, feedback = hybrid_squat_evaluation(user_seq, GOLDEN_SQUAT)
    print(f"Test Score: {score}")
    print(f"Test Feedback: {feedback}")
