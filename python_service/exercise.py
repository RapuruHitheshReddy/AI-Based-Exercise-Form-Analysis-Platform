"""
exercise.py — Unified AI + Biomechanical Analysis
-------------------------------------------------
Generates:
  • Annotated videos (full/simple)
  • Stick figure video
  • All analytics plots + CSVs
  • AI-based form score (Siamese model)
  • Web-ready summary JSON
"""

import os
import cv2
import json
import numpy as np
import pandas as pd
import mediapipe as mp
import matplotlib
matplotlib.use('Agg') # Non-interactive backend
import matplotlib.pyplot as plt
from datetime import datetime
import subprocess
import shutil
import traceback
from collections import deque

# Attempt to import the AI model, but don't fail if it's not there
try:
    from form_learner_model import score_user_video
except ImportError:
    print("WARNING: 'form_learner_model.py' not found. AI scoring will be disabled.")
    def score_user_video(video_path):
        return {"form_score": 0.0, "similarity": 0.0, "error": "Model not found"}


# ===========================================================
# 🎨 Biomechanical Analysis - Config, Helpers & Definitions
# ===========================================================

# --- MediaPipe & UI Initialization (MOVED HERE TO FIX ERROR) ---
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# --- UI & Drawing Config ---
BODY_COLORS = {"torso": (0,140,255), "arms": (255,0,0), "legs": (0,0,255)}
TEXT_COLOR = (255, 255, 255)
UI_BG_COLOR = (50, 50, 50)
UI_ALPHA = 0.6


# --- Biomechanical Definitions (Completed for 22 Exercises) ---

# Note: All exercise names are normalized to lowercase and no spaces
# This is a robust way to look them up (e.g., "barbellbicepscurl")

# MediaPipe Pose Landmark IDs:
# 11: left_shoulder, 12: right_shoulder
# 13: left_elbow,   14: right_elbow
# 15: left_wrist,   16: right_wrist
# 23: left_hip,     24: right_hip
# 25: left_knee,    26: right_knee
# 27: left_ankle,   28: right_ankle

EXERCISE_ANGLES = {
    # --- Legs ---
    "squat": [(11,23,25), (12,24,26), (23,25,27), (24,26,28)], # Hips, Knees
    "deadlift": [(11,23,25), (12,24,26), (23,25,27), (24,26,28)], # Hips, Knees
    "romaniandeadlift": [(11,23,25), (12,24,26), (23,25,27)], # Hips, Knees (Knees bend less)
    "hipthrust": [(11,23,25), (12,24,26), (23,25,27), (24,26,28)], # Hips, Knees
    "legextension": [(23,25,27), (24,26,28)], # Knees
    
    # --- Chest ---
    "benchpress": [(11,13,15), (12,14,16)], # Elbows
    "declinebenchpress": [(11,13,15), (12,14,16)], # Elbows
    "inclinebenchpress": [(11,13,15), (12,14,16)], # Elbows
    "push-up": [(11,13,15), (12,14,16), (11,23,25)], # Elbows, Hips (for sag)
    "chestflymachine": [(23,11,13), (24,12,14)], # Shoulder Adduction (Hip-Shoulder-Elbow)
    
    # --- Back ---
    "pullup": [(11,13,15), (12,14,16)], # Elbows
    "latpulldown": [(11,13,15), (12,14,16)], # Elbows
    "tbarrow": [(11,23,25), (12,24,26), (11,13,15), (12,14,16)], # Hips, Elbows

    # --- Arms ---
    "barbellbicepscurl": [(11,13,15), (12,14,16)], # Elbows
    "hammercurl": [(11,13,15), (12,14,16)], # Elbows
    "triceppushdown": [(11,13,15), (12,14,16)], # Elbows
    "tricepdips": [(11,13,15), (12,14,16), (11,23,25)], # Elbows, Hips
    
    # --- Shoulders ---
    "shoulderpress": [(11,13,15), (12,14,16)], # Elbows
    "lateralraise": [(23,11,13), (24,12,14)], # Shoulder Abduction (Hip-Shoulder-Elbow)

    # --- Core ---
    "legraises": [(11,23,25), (23,25,27)], # Hip, Knee (for straightness)
    "plank": [(11,23,25), (23,25,27)], # Hip, Knee (for straightness)
    "russiantwist": [(11,12,24), (12,11,23)] # Torso Twist (Shoulder-Shoulder-Hip)
}

# --- Labels for all defined angles ---
ANGLE_LABELS = {
    (11,13,15): "Left Elbow",
    (12,14,16): "Right Elbow",
    (23,25,27): "Left Knee",
    (24,26,28): "Right Knee",
    (11,23,25): "Left Hip",
    (12,24,26): "Right Hip",
    (23,11,13): "Left Shoulder Abduction", 
    (24,12,14): "Right Shoulder Abduction",
    (11,12,24): "Left Torso Twist",      
    (12,11,23): "Right Torso Twist"      
}

# --- Rep counting thresholds (high_angle, low_angle) for the *primary* joint ---
REP_THRESHOLDS = {
    # --- Legs ---
    "squat": (170, 90),         
    "deadlift": (170, 90),      
    "romaniandeadlift": (170, 95),
    "hipthrust": (170, 100),    
    "legextension": (170, 90),  
    "benchpress": (160, 70),    
    "declinebenchpress": (160, 70),
    "inclinebenchpress": (160, 70),
    "push-up": (160, 70),       
    "chestflymachine": (90, 20),
    "pullup": (170, 60),        
    "latpulldown": (170, 60),   
    "tbarrow": (160, 90),       
    "barbellbicepscurl": (160, 50), 
    "hammercurl": (160, 50),    
    "triceppushdown": (160, 60),  
    "tricepdips": (160, 70),    
    "shoulderpress": (170, 80), 
    "lateralraise": (80, 20),   
    "legraises": (160, 80),     
    "plank": (180, 180),        
    "russiantwist": (80, 20)    
}


# --- Helper Functions ---
def calculate_angle(a,b,c):
    a,b,c = np.array(a), np.array(b), np.array(c)
    ba, bc = a-b, c-b
    cosang = np.dot(ba,bc)/(np.linalg.norm(ba)*np.linalg.norm(bc)+1e-6)
    return np.degrees(np.arccos(np.clip(cosang,-1.0,1.0)))

def draw_ui_text(img, text, pos, font_scale=0.7, color=TEXT_COLOR, bg_color=UI_BG_COLOR, thickness=2, alpha=UI_ALPHA):
    (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    bg_pos1 = (pos[0] - 5, pos[1] + 5)
    bg_pos2 = (pos[0] + text_w + 5, pos[1] - text_h - 10)
    try:
        sub_img = img[bg_pos1[1]:bg_pos2[1], bg_pos1[0]:bg_pos2[0]]
        bg_rect = np.zeros(sub_img.shape, dtype=np.uint8)
        bg_rect[:] = bg_color
        res = cv2.addWeighted(sub_img, 1 - alpha, bg_rect, alpha, 0)
        img[bg_pos1[1]:bg_pos2[1], bg_pos1[0]:bg_pos2[0]] = res
    except Exception:
        pass 
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)

def get_form_quality(angle, ideal_range, margin=10):
    low, high = ideal_range
    low_with_margin = max(0, low - margin)
    high_with_margin = high + margin
    if angle < low_with_margin: return max(0, int((angle / low_with_margin) * 100))
    if angle > high_with_margin: return max(0, int((high_with_margin / angle) * 100))
    return 100

def draw_simple_skeleton(image, landmarks, connection_style):
    mp_drawing.draw_landmarks(
        image, landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=None, 
        connection_drawing_spec=connection_style
    )

def get_static_ideal_range(label):
    if "Knee" in label: return (70, 170)
    if "Hip" in label: return (80, 170)
    if "Elbow" in label: return (50, 160)
    return (70, 170) 


# ===========================================================
# 🎞️ Convert to H.264 for Browser Playback (Robust)
# ===========================================================
# Robust ffmpeg detection and conversion
def convert_to_h264(input_path):
    """
    Converts an MP4 file to browser-friendly H.264 format.
    Adds detailed console logs for debugging conversion issues.
    """
    if not os.path.exists(input_path):
        print(f"[WARN] convert_to_h264: file missing {input_path}")
        return

    print(f"🎞️ Starting H.264 conversion for: {input_path}")

    ffmpeg_path = shutil.which("ffmpeg") or r"C:\ffmpeg\bin\ffmpeg.exe"
    if not ffmpeg_path or not os.path.exists(ffmpeg_path):
        print("⚠️ ffmpeg not found. Skipping conversion.")
        return

    # Prepare output file
    output_path = input_path.replace(".mp4", "_h264.mp4")

    cmd = [
        ffmpeg_path, "-y",
        "-hide_banner", "-loglevel", "error",
        "-i", input_path,
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        output_path
    ]

    try:
        print(f"🔧 Running ffmpeg: {' '.join(cmd)}")
        process = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        if process.returncode != 0:
            print(f"❌ ffmpeg error (code {process.returncode}):\n{process.stderr.strip()}")
            return

        if not os.path.exists(output_path):
            print(f"❌ ffmpeg did not produce {output_path}")
            return

        size = os.path.getsize(output_path)
        if size < 1024:
            print(f"⚠️ Output file too small ({size} bytes). Possible conversion issue.")
        else:
            print(f"✅ Conversion successful ({size/1024:.1f} KB): {os.path.basename(output_path)}")

        # Replace original
        os.replace(output_path, input_path)
        print(f"🔁 Replaced original with H.264 version: {os.path.basename(input_path)}")

    except Exception as e:
        print(f"💥 [ERROR] ffmpeg conversion failed for {input_path}: {e}")
        traceback.print_exc()


# ===========================================================
# 🧠 AI Model Integration
# ===========================================================
class AIFormScorer:
    def __init__(self):
        print("✅ AI Form-Learner initialized.")
    def evaluate(self, video_path: str):
        try:
            result = score_user_video(video_path)
            return {
                "form_score": float(result.get("form_score", 0.0) or 0.0),
                "similarity": float(result.get("similarity", 0.0) or 0.0)
            }
        except Exception as e:
            print(f"[AI ERROR] {e}")
            return {"form_score": 0.0, "similarity": -1.0}


# ===========================================================
# 📊 [REBUILT] Analysis & Visualization Core
# ===========================================================
def analyze_multiple_videos(video_paths, exercise_name, output_folder):
    """
    This is the full, correct analysis function that draws
    skeletons, angles, reps, and quality bars.
    """
    
    safe_ex_name = exercise_name.lower().replace(" ", "")
    
    tracked_joints = EXERCISE_ANGLES.get(safe_ex_name, [])
    if not tracked_joints:
        print(f"Warning: No joints defined for '{exercise_name}'. Using defaults.")
        tracked_joints = EXERCISE_ANGLES.get("squat") 
        
    high_thresh, low_thresh = REP_THRESHOLDS.get(safe_ex_name, (160, 60))

    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=1)
    all_dfs = []

    for idx, path in enumerate(video_paths, start=1):
        base = os.path.splitext(os.path.basename(path))[0]
        vid_folder = os.path.join(output_folder, f"video_{idx}_{base}")
        os.makedirs(vid_folder, exist_ok=True)
        print(f"\n🔍 Analyzing video {idx}: {base}")

        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            print(f"Error: Could not open video {path}")
            continue
            
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 20

        out_paths = {
            "full": os.path.join(vid_folder, "annotated_full.mp4"),
            "simple": os.path.join(vid_folder, "annotated_simple.mp4"),
            "stick": os.path.join(vid_folder, "stick_figure.mp4")
        }
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_writers = {
            "full": cv2.VideoWriter(out_paths["full"], fourcc, fps, (frame_w, frame_h)),
            "simple": cv2.VideoWriter(out_paths["simple"], fourcc, fps, (frame_w, frame_h)),
            "stick": cv2.VideoWriter(out_paths["stick"], fourcc, fps, (frame_w, frame_h))
        }

        all_frame_data = []
        rep_count, direction = 0, 0
        angle_hist = deque(maxlen=60)
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_full = frame.copy()
            frame_simple = frame.copy()
            frame_stick = np.zeros_like(frame) 

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = pose.process(image_rgb)
            
            frame_data = {"Frame": frame_idx, "Reps": rep_count}
            
            if res.pose_landmarks:
                lm = res.pose_landmarks.landmark
                y_pos = 60
                form_scores_this_frame = []
                
                for triplet in tracked_joints:
                    label = ANGLE_LABELS.get(triplet, f"Joint_{triplet[1]}")
                    try:
                        a = [lm[triplet[0]].x, lm[triplet[0]].y]
                        b = [lm[triplet[1]].x, lm[triplet[1]].y]
                        c = [lm[triplet[2]].x, lm[triplet[2]].y]
                        angle = calculate_angle(a, b, c)
                        
                        frame_data[f"{label} Angle"] = angle
                        
                        ideal_range = get_static_ideal_range(label)
                        quality = get_form_quality(angle, ideal_range)
                        form_scores_this_frame.append(quality)
                        emoji = "✅" if quality >= 80 else "⚠️"
                        
                        feedback_text = f"{label}: {int(angle)}° {emoji}"
                        draw_ui_text(frame_full, feedback_text, (30, y_pos))
                        y_pos += 40

                        if triplet == tracked_joints[0]:
                            angle_hist.append(angle) 
                            if angle > high_thresh: direction = 1
                            if angle < low_thresh and direction == 1:
                                rep_count += 1
                                direction = 0
                                
                    except Exception:
                        frame_data[f"{label} Angle"] = np.nan
                        continue
                
                mp_drawing.draw_landmarks(
                    frame_full, res.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                )
                draw_simple_skeleton(frame_simple, res.pose_landmarks,
                    mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2))
                draw_simple_skeleton(frame_stick, res.pose_landmarks,
                    mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2))

                draw_ui_text(frame_full, f"Reps: {rep_count}", (30, y_pos + 10), font_scale=0.9)
                avg_quality = int(np.mean(form_scores_this_frame)) if form_scores_this_frame else 0
                frame_data["Form Quality"] = avg_quality
                
                bar_x = frame_w - 250
                bar_y = frame_h - 70
                bar_w = 200
                quality_w = int(bar_w * (avg_quality / 100))
                color = (0, 255, 0) if avg_quality >= 80 else (0, 165, 255) if avg_quality >= 50 else (0, 0, 255)
                cv2.rectangle(frame_full, (bar_x, bar_y), (bar_x + bar_w, bar_y + 20), UI_BG_COLOR, -1)
                cv2.rectangle(frame_full, (bar_x, bar_y), (bar_x + quality_w, bar_y + 20), color, -1)
                draw_ui_text(frame_full, f"Form Quality: {avg_quality}%", (bar_x, bar_y - 10), alpha=1.0)
                
            else:
                frame_data.update({f"{ANGLE_LABELS.get(t, 'Joint')} Angle": np.nan for t in tracked_joints})
                frame_data["Form Quality"] = np.nan

            out_writers["full"].write(frame_full)
            out_writers["simple"].write(frame_simple)
            out_writers["stick"].write(frame_stick)
            all_frame_data.append(frame_data)
            frame_idx += 1

        cap.release()
        for w in out_writers.values(): w.release()
        pose.close()

        for path in out_paths.values():
            convert_to_h264(path)
            
        print(f"✅ Videos generated for video {idx}.")

        df = pd.DataFrame(all_frame_data)
        df.to_csv(os.path.join(vid_folder, f"data_{exercise_name}_video{idx}.csv"), index=False)
        
        if "Left Hip Angle" in df.columns and "Left Knee Angle" in df.columns:
             df["Error_Hip_Depth"] = (df["Left Hip Angle"] > df["Left Knee Angle"]).astype(int)
        if "Left Hip Angle" in df.columns: 
             df["Error_Back_Straightness"] = (df["Left Hip Angle"] < 160).astype(int)
        if "Left Knee Angle" in df.columns: 
             df["Error_Knee_Caving"] = (df["Left Knee Angle"] < 80).astype(int)

        all_dfs.append(df)

        try:
            angle_cols = [c for c in df.columns if "Angle" in c]
            if angle_cols:
                plt.figure(figsize=(10, 5))
                for col in angle_cols:
                    plt.plot(df["Frame"], df[col], label=col)
                plt.legend(); plt.title(f"Angle Progression (Video {idx})")
                plt.xlabel("Frame"); plt.ylabel("Angle (degrees)")
                plt.savefig(os.path.join(vid_folder, f"angle_progression_video{idx}.png")); plt.close()

            if "Form Quality" in df.columns:
                plt.figure(figsize=(10, 4))
                plt.plot(df["Frame"], df["Form Quality"].rolling(window=15).mean(), color='green')
                plt.title(f"Form Quality Over Time (Video {idx})")
                plt.xlabel("Frame"); plt.ylabel("Quality (%)"); plt.ylim(0, 105)
                plt.savefig(os.path.join(vid_folder, f"quality_plot_video{idx}.png")); plt.close()

            error_cols = [c for c in df.columns if "Error_" in c]
            if error_cols:
                error_sums = df[error_cols].sum()
                plt.figure(figsize=(8, 4))
                error_sums.plot(kind='bar', color='red')
                plt.title(f"Total Error Frames (Video {idx})")
                plt.ylabel("Frame Count")
                plt.savefig(os.path.join(vid_folder, f"error_summary_video{idx}.png")); plt.close()
                
            print(f"✅ Plots generated for video {idx}.")

        except Exception as e:
            print(f"Plotting Error: {e}")
            plt.close() 

    if not all_dfs:
        print("No dataframes generated, returning empty.")
        return pd.DataFrame()
        
    combined = pd.concat(all_dfs, ignore_index=True)
    combined.to_csv(os.path.join(output_folder, f"data_{exercise_name}_combined.csv"), index=False)
    print(f"✅ Combined CSV written for {exercise_name}")
    return combined


# ===========================================================
# 🌐 Unified Entry (AI + Video Outputs)
# ===========================================================
def run_web_analysis(video_path, exercise_name, output_root):
    try:
        safe = exercise_name.replace(" ", "_").lower()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_folder = os.path.join(output_root, f"{safe}_{timestamp}")
        os.makedirs(session_folder, exist_ok=True)

        print(f"\n🏋️‍♂️ Running analysis for {exercise_name} → {session_folder}")
        
        local_video_path = os.path.join(session_folder, f"source_{safe}.mp4")
        if isinstance(video_path, str): 
             from shutil import copyfile
             copyfile(video_path, local_video_path)
        else: 
             with open(local_video_path, "wb") as f:
                   f.write(video_path.read())

        df = analyze_multiple_videos([local_video_path], exercise_name, session_folder)

        if df.empty:
            print("Analysis returned no data. Aborting.")
            return None, None, {"error": "Analysis failed to process video."}

        ai = AIFormScorer()
        ai_result = ai.evaluate(local_video_path)
        form_score, sim = ai_result["form_score"], ai_result["similarity"]

        feedback = []
        if form_score >= 85: feedback.append("Excellent technique — matches expert pattern closely.")
        elif form_score >= 70: feedback.append("Good form — minor refinements needed.")
        elif form_score >= 50: feedback.append("Moderate deviation — focus on consistency.")
        else: feedback.append("Significant deviation — revisit fundamentals.")
        
        if "Error_Back_Straightness" in df and df["Error_Back_Straightness"].sum() > 5:
            feedback.append("Back rounding detected — engage your core.")
        if "Error_Knee_Caving" in df and df["Error_Knee_Caving"].sum() > 5:
            feedback.append("Knees caving inward — push outward.")
        if "Error_Hip_Depth" in df and df["Error_Hip_Depth"].sum() > 5:
            feedback.append("Insufficient hip depth — go slightly lower.")

        total_reps = int(df["Reps"].max()) if "Reps" in df.columns else 0
        summary = {
            "exercise_name": exercise_name,
            "total_reps": total_reps,
            "ai_form_score": round(form_score, 2),
            "similarity": round(sim, 3),
            "feedback": feedback,
            "timestamp": timestamp,
            "session_id": f"{safe}_{timestamp}" 
        }
        json_path = os.path.join(session_folder, f"summary_{safe}.json")
        with open(json_path, "w") as jf:
            json.dump(summary, jf, indent=2)

        print(f"✅ Completed {exercise_name}: Score={form_score:.2f}")
        return df, session_folder, summary

    except Exception as e:
        print(f"[ERROR] run_web_analysis failed: {e}")
        traceback.print_exc()
        return None, None, {"error": str(e)}


# ===========================================================
# 🔬 Local Test
# ===========================================================
if __name__ == "__main__":
    
    TEST_VIDEO = "samples/squat_test.mp4"
    OUTPUT_DIR = "results_web"
    os.makedirs("samples", exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    try:
        ffmpeg_path = shutil.which("ffmpeg") or shutil.which("ffmpeg.exe")
        if not ffmpeg_path:
             raise FileNotFoundError("ffmpeg not in PATH")
        
        print(f"ffmpeg found at {ffmpeg_path}. Creating test video...")
        
        cmd = [
            ffmpeg_path, "-y", "-f", "lavfi", "-i", 
            "testsrc=duration=5:size=640x480:rate=20", # Corrected duration syntax
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            TEST_VIDEO
        ]
        # Use subprocess.run with check=True
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"Created dummy video: {TEST_VIDEO}")
        
    except Exception as e:
        print(f"ffmpeg not found or failed. Skipping test video creation. {e}")
        print("Please install ffmpeg and add it to your system's PATH.")
        print("You can download it from: https://www.gyan.dev/ffmpeg/builds/")
        print("Please create a 'samples/squat_test.mp4' file manually to run the test.")

    if os.path.exists(TEST_VIDEO):
        print("\n--- RUNNING LOCAL TEST ---")
        df, folder, summary = run_web_analysis(TEST_VIDEO, "squat", OUTPUT_DIR)
        print("\n📦 Summary JSON:\n", json.dumps(summary, indent=2))
        
        if folder:
            print(f"\n📁 Files generated in: {folder}")
            for root, _, files in os.walk(folder):
                for f in files:
                    print(f"  - {f}")
    else:
        print("Test video not found. Skipping main test.")