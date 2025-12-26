import os
import torch
import torch.nn as nn
import numpy as np
import random
import sys
from models import Encoder, Head
from feature_extractor import extract_keypoints
from consts import LABELS  # ✅ استيراد القاموس الموحد

DEVICE = "cpu"
DATA_DIR = "data"
MODEL_DIR = "model"
BASE_MODEL_PATH = os.path.join(MODEL_DIR, "base_model.pth")
HEAD_MODEL_PATH = os.path.join(MODEL_DIR, "head_model.pth")
EPOCHS = 50
LEARNING_RATE = 0.001
os.makedirs(MODEL_DIR, exist_ok=True)
NUM_CLASSES = len(LABELS)

def apply_zoom(kp, zoom_range=(0.6, 1.4)):
    scale = random.uniform(*zoom_range)
    return kp * scale

def apply_noise(kp, factor=0.01):
    return kp + torch.randn_like(kp) * factor

def train():
    print(f"🚀 Starting Final Training...")
    dataset = []
    
    for label_name, label_id in LABELS.items():
        label_dir = os.path.join(DATA_DIR, label_name)
        if not os.path.isdir(label_dir): continue
        for video in os.listdir(label_dir):
            kp = extract_keypoints(os.path.join(label_dir, video))
            if kp is not None and len(kp) > 0:
                base = torch.tensor(kp, dtype=torch.float32).to(DEVICE)
                label = torch.tensor([label_id], dtype=torch.long).to(DEVICE)
                dataset.append((base.unsqueeze(0), label))
                # Augmentation
                for _ in range(30): 
                    aug = base.clone()
                    aug = apply_zoom(aug) 
                    if random.random() < 0.5: aug = apply_noise(aug)
                    dataset.append((aug.unsqueeze(0), label))

    if not dataset: print("❌ No data!"); return
    print(f"✅ Training on {len(dataset)} samples.")

    encoder = Encoder().to(DEVICE)
    head = Head(num_classes=NUM_CLASSES).to(DEVICE)
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(head.parameters()), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    for epoch in range(EPOCHS):
        encoder.train(); head.train()
        random.shuffle(dataset)
        correct = 0
        for X, y in dataset:
            optimizer.zero_grad()
            logits = head(encoder(X))
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            if torch.argmax(logits, dim=1) == y: correct += 1
        
        acc = (correct / len(dataset)) * 100
        if acc >= best_acc:
            best_acc = acc
            torch.save(encoder.state_dict(), BASE_MODEL_PATH)
            torch.save(head.state_dict(), HEAD_MODEL_PATH)
        
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} | Acc: {acc:.1f}%")
            
    print("🏆 Training Complete.")

if __name__ == "__main__":
    train()