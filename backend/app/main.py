from fastapi import FastAPI, UploadFile, File, HTTPException, Form, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from enum import Enum
import base64
import cv2
import numpy as np
import json
from ml.extract_keypoints import extract_image_keypoints
from ml.feature_engineering import KeypointSmoother

try:
    import mediapipe as mp
    mp_pose = mp.solutions.pose
    HAS_MP = True
except AttributeError:
    HAS_MP = False
import uvicorn
import shutil
import os
from ml.inference import MultiSkillEvaluator

app = FastAPI(title="MotionMind API")

class ActionPayload(BaseModel):
    action_type: str  # e.g., 'squat', 'jab', 'shoot'
    user_id: str

active_filters = {}
# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

evaluator = None

@app.on_event("startup")
def load_models():
    global evaluator
    try:
        evaluator = MultiSkillEvaluator()
        print("ML Models loaded successfully.")
    except Exception as e:
        print(f"Warning: ML Models not loaded. {e}")

@app.get("/")
def read_root():
    return {"message": "Welcome to MotionMind API"}

@app.post("/evaluate")
async def evaluate_video(file: UploadFile = File(...), skill: str = Form("squat")):
    global evaluator
    if not evaluator:
        raise HTTPException(status_code=500, detail="ML Models not loaded on server.")

    temp_video_path = f"temp_{file.filename}"
    with open(temp_video_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        results = evaluator.evaluate(temp_video_path, skill_name=skill)

        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)

        if "error" in results:
            raise HTTPException(status_code=400, detail=results["error"])

        return results

    except Exception as e:
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws/analyze")
async def websocket_analyze(websocket: WebSocket):
    await websocket.accept()
    session_id = str(id(websocket))
    active_filters[session_id] = KeypointSmoother(maxlen=5)
    
    pose = None
    if HAS_MP:
        pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5)
        
    try:
        while True:
            # 1. Receive JSON payload from iOS
            raw_data = await websocket.receive_text()
            payload = json.loads(raw_data)
            
            action = payload.get("action", "squat")
            encoded_frame = payload.get("frame")
            
            if not encoded_frame:
                continue
                
            # 2. Decode base64
            img_data = base64.b64decode(encoded_frame)
            np_arr = np.frombuffer(img_data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            if frame is None:
                await websocket.send_json({"error": "Failed to decode image data"})
                continue
                
            # 3. Process the frame (Real ML pipeline)
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            raw_keypoints = extract_image_keypoints(img_rgb, pose)
            
            if not raw_keypoints:
                await websocket.send_json({
                    "status": "success",
                    "action": action, 
                    "score": 0, 
                    "feedback": ["No human pose detected."]
                })
                continue
                
            smoothed_kp = active_filters[session_id].smooth(raw_keypoints)
            
            # 4. Route to multi-action logic
            result = evaluator.analyze_frame(smoothed_kp, action)
            
            # 5. Stream results back to iOS client
            await websocket.send_json({
                "status": "success",
                "action": action,
                "score": result["score"],
                "feedback": result["feedback"]
            })
            
    except WebSocketDisconnect:
        if session_id in active_filters:
            del active_filters[session_id]
        print(f"Client {session_id} disconnected.")
    finally:
        if pose:
            pose.close()

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
