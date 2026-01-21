import cv2
import numpy as np
import os
import time

# --- SETUP ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DET_MODEL = os.path.join(BASE_DIR, "models", "face_detection_yunet_2023mar.onnx")
REC_MODEL = os.path.join(BASE_DIR, "models", "face_recognition_sface_2021dec.onnx")
DB_PATH = os.path.join(BASE_DIR, "data", "face_db")

detector = cv2.FaceDetectorYN.create(DET_MODEL, "", (640, 480))
recognizer = cv2.FaceRecognizerSF.create(REC_MODEL, "")

# Load Database
face_db = {f.replace(".npy", ""): np.load(os.path.join(DB_PATH, f)) 
           for f in os.listdir(DB_PATH) if f.endswith(".npy")}

cap = cv2.VideoCapture(0)
THRESHOLD = 0.363
prev_time = time.perf_counter()

# --- PERFORMANCE TRACKERS ---
frame_count = 0
total_det_time = 0
total_rec_time = 0
robustness_history = []  # Tracks number of faces detected per frame

print(f"\n{'='*60}")
print(" SYSTEM ONLINE: EVALUATING EFFICIENCY, ACCURACY, & ROBUSTNESS")
print(f"{'='*60}\n")

while True:
    current_time = time.perf_counter()
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame, 1)

    # 1. MEASURE DETECTION EFFICIENCY
    t_start_det = time.perf_counter()
    detector.setInputSize((frame.shape[1], frame.shape[0]))
    _, faces = detector.detect(frame)
    det_latency = (time.perf_counter() - t_start_det) * 1000 # Convert to ms
    total_det_time += det_latency

    # Track robustness (Face count stability)
    num_faces = len(faces) if faces is not None else 0
    robustness_history.append(num_faces)

    current_max_frame_score = 0.0 # Tracking for accuracy metric

    if faces is not None:
        for face in faces:
            # 2. MEASURE RECOGNITION EFFICIENCY
            t_start_rec = time.perf_counter()
            aligned = recognizer.alignCrop(frame, face)
            feat_live = recognizer.feature(aligned)
            
            current_best_name = "Unknown"
            current_max_score = 0.0
            
            for name, feat_known in face_db.items():
                score = recognizer.match(feat_known, feat_live, cv2.FaceRecognizerSF_FR_COSINE)
                if score > current_max_score:
                    current_max_score = score
                    if score > THRESHOLD:
                        current_best_name = name

            rec_latency = (time.perf_counter() - t_start_rec) * 1000
            total_rec_time += rec_latency
            
            # Update frame max for terminal metrics
            if current_max_score > current_max_frame_score:
                current_max_frame_score = current_max_score

            # 3. DRAWING (Confidence and Bounding Box)
            box = list(map(int, face[:4]))
            color = (0, 255, 0) if current_best_name != "Unknown" else (0, 0, 255)
            display_text = f"{current_best_name} ({current_max_score:.2f})"
            
            cv2.rectangle(frame, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), color, 2)
            cv2.putText(frame, display_text, (box[0], box[1]-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # 4. CALCULATE REAL-TIME FPS
    time_diff = current_time - prev_time
    fps = 1 / time_diff if time_diff > 0 else 0
    prev_time = current_time
    frame_count += 1
    
    cv2.putText(frame, f"FPS: {fps:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    # 5. TERMINAL PERFORMANCE EVALUATION (Every 30 frames)
    if frame_count % 30 == 0:
        avg_det = total_det_time / 30
        avg_rec = total_rec_time / (30 if total_rec_time > 0 else 1)
        # Robustness: High if face count is steady (Stability score)
        stability = 100 - (np.std(robustness_history[-30:]) * 20)
        
        print(f"--- [Frame {frame_count}] Performance Summary ---")
        print(f" EFFICIENCY | Det Latency: {avg_det:.1f}ms | Rec Latency: {avg_rec:.1f}ms | System FPS: {fps:.1f}")
        print(f" ACCURACY   | Highest Confidence: {current_max_frame_score:.4f} (Threshold: {THRESHOLD})")
        print(f" ROBUSTNESS | Signal Stability: {max(0, stability):.1f}%")
        print("-" * 50)
        
        # Reset buffers for next cycle
        total_det_time = 0
        total_rec_time = 0

    cv2.imshow('Face Recognition & Performance Monitor', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()