import torch
import os
import numpy as np
from collections import deque, Counter
from models import Encoder, Head
from feature_extractor import extract_keypoints
from consts import LABELS, TRANSLATIONS

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_DIR = "model"
BASE_MODEL_PATH = os.path.join(MODEL_DIR, "base_model.pth")
HEAD_MODEL_PATH = os.path.join(MODEL_DIR, "head_model.pth")

# نظام التصويت: بنخزن آخر 5 توقعات
history = deque(maxlen=5)

encoder = None
head = None

try:
    encoder = Encoder().to(DEVICE)
    head = Head(num_classes=len(LABELS)).to(DEVICE)
    if os.path.exists(BASE_MODEL_PATH):
        encoder.load_state_dict(torch.load(BASE_MODEL_PATH, map_location=DEVICE))
        head.load_state_dict(torch.load(HEAD_MODEL_PATH, map_location=DEVICE))
        encoder.eval(); head.eval()
        print("✅ Models loaded successfully!")
except Exception as e:
    print(f"⚠️ Error: {e}")

def predict(video_path):
    if encoder is None: return "خطأ نظام"
    
    kp = extract_keypoints(video_path)
    if kp is None or len(kp) == 0:
        history.clear() # لو مفيش إيد، امسح الذاكرة
        return "لا يوجد يد"

    X = torch.tensor(kp, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        probs = torch.softmax(head(encoder(X)), dim=1)
        conf, idx = torch.max(probs, dim=1)

    current_label_idx = idx.item()
    current_conf = conf.item()
    
    # 1. لو الثقة ضعيفة، تجاهل التوقع ده
    if current_conf < 0.60:
        return "غير واضح"

    # 2. ضيف التوقع للذاكرة
    history.append(current_label_idx)

    # 3. التصويت: لازم 3 من آخر 5 مرات يكونوا نفس الكلمة
    if len(history) == 5:
        most_common, count = Counter(history).most_common(1)[0]
        if count >= 3:
            label_en = list(LABELS.keys())[list(LABELS.values()).index(most_common)]
            return f"{TRANSLATIONS[label_en]}"
    
    return "جاري التأكد..."