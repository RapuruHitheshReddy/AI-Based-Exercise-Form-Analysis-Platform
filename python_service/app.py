import os
import shutil
import traceback
import uvicorn
from datetime import datetime
from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from exercise import run_web_analysis

# ===========================================================
# ⚙️ App Configuration (Absolute Paths + CORS)
# ===========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

app = FastAPI(
    title="🏋️ Exercise Form Analyzer API",
    description="AI + biomechanical feedback system for workout videos",
    version="4.2"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # ✅ Allow all (change to frontend domain in production)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Always serve results from correct absolute path
app.mount("/results", StaticFiles(directory=RESULTS_DIR), name="results")


# ===========================================================
# 🗂️ Helper: Categorize Generated Files for Web URLs
# ===========================================================
def categorize_files(output_folder: str) -> dict:
    """
    Walk through the results folder and return categorized file URLs.
    (Ensures web-safe relative URLs like /results/session/file.mp4)
    """
    categorized = {"videos": {}, "csv": {}, "plots": {}, "reports": {}}
    session_base_path = os.path.relpath(output_folder, RESULTS_DIR).replace("\\", "/")

    for root, _, files in os.walk(output_folder):
        for f in files:
            rel_path = os.path.relpath(os.path.join(root, f), output_folder).replace("\\", "/")
            url = f"/results/{session_base_path}/{rel_path}"

            if f.endswith(".mp4"):
                if "annotated_full" in f:
                    categorized["videos"]["annotated_full"] = url
                elif "annotated_simple" in f:
                    categorized["videos"]["annotated_simple"] = url
                elif "stick" in f:
                    categorized["videos"]["stick_figure"] = url
                elif "source_" not in f:
                    categorized["videos"][f] = url

            elif f.endswith(".csv"):
                categorized["csv"][f.split('.')[0].replace('data_', '')] = url

            elif f.endswith(".png"):
                categorized["plots"][f.split('.')[0].replace('_video1', '')] = url

            elif f.endswith(".json"):
                categorized["reports"]["summary_json"] = url
            elif f.endswith(".txt"):
                categorized["reports"]["report_text"] = url

    return categorized


# ===========================================================
# 🚀 API Endpoint: /analyze
# ===========================================================
@app.post("/analyze")
async def analyze_exercise(
    exercise_name: str = Form(...),
    video: UploadFile = Form(...)
):
    """
    Main endpoint:
    - Saves uploaded video
    - Runs biomechanical + AI analysis
    - Returns URLs + metrics for frontend visualization
    """
    try:
        # 1️⃣ Save uploaded video safely
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_filename = "".join(c for c in video.filename if c.isalnum() or c in ('.', '_')).rstrip()
        video_filename = f"{timestamp}_{safe_filename}"
        video_path = os.path.join(UPLOAD_DIR, video_filename)

        with open(video_path, "wb") as f:
            shutil.copyfileobj(video.file, f)

        print(f"📥 Received '{video.filename}' for exercise '{exercise_name}'")

        # 2️⃣ Run main analysis (AI + Biomechanics)
        df, output_folder, summary = run_web_analysis(video_path, exercise_name, RESULTS_DIR)

        if not summary or "error" in summary:
            return JSONResponse(
                status_code=500,
                content={"error": "❌ Analysis failed.", "details": summary.get("error", "Unknown")}
            )

        # 3️⃣ Categorize generated outputs
        files = categorize_files(output_folder)

        # 4️⃣ Build JSON response
        response_data = {
            "exercise_name": exercise_name,
            "session_id": summary.get("session_id"),
            "session_folder_url": f"/results/{summary.get('session_id')}",
            "ai_form_score": summary.get("ai_form_score"),
            "similarity": summary.get("similarity"),
            "total_reps": summary.get("total_reps"),
            "feedback": summary.get("feedback"),
            "timestamp": summary.get("timestamp"),
            "files": files
        }

        print(f"✅ Completed analysis for '{exercise_name}' → {output_folder}")
        return JSONResponse(content=response_data)

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"error": f"Server error: {str(e)}"}
        )


# ===========================================================
# 🌐 Root Endpoint (Health Check)
# ===========================================================
@app.get("/")
async def root():
    return {
        "message": "🏋️‍♂️ Exercise Analyzer API is running!",
        "upload_dir": UPLOAD_DIR,
        "results_dir": RESULTS_DIR,
        "results_url_public": "/results",
        "endpoint": "/analyze"
    }


# ===========================================================
# 🧩 Entry Point (Stable Run)
# ===========================================================
if __name__ == "__main__":
    print("🚀 Starting FastAPI server...")
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
