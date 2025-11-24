
# form_learner_model.py
import os, cv2, numpy as np, torch, mediapipe as mp
from torch import nn
from numpy.linalg import norm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SiameseEncoder(nn.Module):
    def __init__(self, input_dim=99, hidden_dim=256, embed_dim=128, num_layers=2, bidirectional=True):
        super().__init__()
        self.input_fc = nn.Linear(input_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True, bidirectional=bidirectional)
        self.pool_fc = nn.Linear(hidden_dim*(2 if bidirectional else 1), embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
    def forward(self, x):
        x = self.input_fc(x)
        out,_ = self.lstm(x)
        pooled = out.mean(dim=1)
        emb = self.pool_fc(pooled)
        emb = self.norm(emb)
        return nn.functional.normalize(emb, p=2, dim=1)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model_output", "form_learner_siamese_extended.pth")
EXPERT_MEAN_PATH = os.path.join(os.path.dirname(__file__), "model_output", "expert_mean_extended.npy")

model = SiameseEncoder().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
expert_mean = np.load(EXPERT_MEAN_PATH)
expert_mean /= np.linalg.norm(expert_mean) + 1e-8

mp_pose = mp.solutions.pose
def normalize_keypoints(lm):
    lm = np.array(lm, np.float32)
    c = (lm[23]+lm[24])/2; lm -= c
    h = (np.linalg.norm(lm[11]-lm[27]) + np.linalg.norm(lm[12]-lm[28]))/2
    h = max(h, 1e-6)
    lm /= h
    return lm

def extract_pose_from_video(video_path, frame_skip=3):
    cap = cv2.VideoCapture(video_path)
    pose = mp_pose.Pose(model_complexity=0)
    frames=[]; f_idx=0
    while cap.isOpened():
        ok,frame=cap.read()
        if not ok: break
        if f_idx%frame_skip==0:
            res=pose.process(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
            if res.pose_world_landmarks:
                pts=[[lm.x,lm.y,lm.z] for lm in res.pose_world_landmarks.landmark]
                frames.append(normalize_keypoints(pts))
        f_idx+=1
    cap.release(); pose.close()
    return np.array(frames,np.float32) if len(frames)>5 else None

def score_user_video(video_path):
    seq = extract_pose_from_video(video_path)
    if seq is None: return {"form_score": None, "similarity": None}
    idx = np.linspace(0, len(seq)-1, 100).astype(int)
    seq = seq[idx].reshape(1,100,-1)
    x = torch.tensor(seq, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        emb = model(x).cpu().numpy()[0]
    similarity = float(np.dot(emb, expert_mean)/(norm(emb)*norm(expert_mean)))
    score = round(((similarity + 1) / 2) * 100, 2)
    return {"form_score": score, "similarity": round(similarity, 3)}
