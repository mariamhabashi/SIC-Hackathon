import cv2
import mediapipe as mp
import numpy as np
import os

mp_hands = mp.solutions.hands

def extract_keypoints(video_path, target_frames=45):
    if not os.path.exists(video_path): return None
        
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frames.append(frame)
    cap.release()

    if len(frames) == 0: return None

    indices = np.linspace(0, len(frames)-1, target_frames, dtype=int)
    selected_frames = [frames[i] for i in indices]

    seq = []
    with mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5) as hands:
        for frame in selected_frames:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)
            
            frame_kp = np.zeros(126)
            
            if results.multi_hand_landmarks:
                hands_data = []
                # تجميع البيانات
                if results.multi_handedness:
                    for lm, handed in zip(results.multi_hand_landmarks, results.multi_handedness):
                        hands_data.append((handed.classification[0].label, lm))
                
                # ترتيب ثابت (Right ثم Left) عشان الموديل ميحولش
                hands_data.sort(key=lambda x: x[0], reverse=True) 
                
                # لو ملقاش غير إيد واحدة، بناخدها ونكمل أصفار
                for i, (_, hand_landmarks) in enumerate(hands_data[:2]):
                    wrist = hand_landmarks.landmark[0]
                    base_idx = i * 63
                    for j, lm in enumerate(hand_landmarks.landmark):
                        # Simple Relative Coordinates (الأكثر استقراراً)
                        frame_kp[base_idx + j*3]     = lm.x - wrist.x
                        frame_kp[base_idx + j*3 + 1] = lm.y - wrist.y
                        frame_kp[base_idx + j*3 + 2] = lm.z - wrist.z
            
            seq.append(frame_kp)

    return np.array(seq, dtype=np.float32)