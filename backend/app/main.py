from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import shutil
import os
from ml.inference import MultiSkillEvaluator

app = FastAPI(title="MotionMind API")

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
        results = evaluator.evaluate(temp_video_path, skill=skill)

        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)

        if "error" in results:
            raise HTTPException(status_code=400, detail=results["error"])

        return results

    except Exception as e:
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
