import cv2
import numpy as np
import os

# 1. Setup Paths (Mirroring your main.py structure)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DETECTOR_PATH = os.path.join(BASE_DIR, "models", '/Users/mayankanand/Desktop/Anand_coding/Vishveya/Face RecognitionPi/models/face_detection_yunet_2023mar.onnx')
RECOGNIZER_PATH = os.path.join(BASE_DIR, "models", '/Users/mayankanand/Desktop/Anand_coding/Vishveya/Face RecognitionPi/models/face_recognition_sface_2021dec.onnx')
DB_PATH = os.path.join(BASE_DIR, "data", "face_db")

if not os.path.exists(DB_PATH):
    os.makedirs(DB_PATH)

# 2. Initialize Models
detector = cv2.FaceDetectorYN.create(DETECTOR_PATH, "", (640, 480))
recognizer = cv2.FaceRecognizerSF.create(RECOGNIZER_PATH, "")

def enroll_via_webcam():
    name = input("Enter the name of the person to enroll: ").strip()
    if not name:
        print("Name cannot be empty!")
        return

    cap = cv2.VideoCapture(0)
    print(f"Look at the camera, {name}. Press 'SPACE' to capture or 'ESC' to cancel.")

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.flip(frame, 1) # Mirror for easier positioning
        display_frame = frame.copy()
        
        # Live detection feedback
        detector.setInputSize((frame.shape[1], frame.shape[0]))
        _, faces = detector.detect(frame)

        if faces is not None:
            for face in faces:
                box = list(map(int, face[:4]))
                cv2.rectangle(display_frame, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), (255, 0, 0), 2)

        cv2.imshow("Enrollment - Press SPACE to Capture", display_frame)
        
        key = cv2.waitKey(1)
        if key == 27: # ESC
            break
        elif key == 32: # SPACE
            if faces is not None and len(faces) > 0:
                # Use the first face detected
                aligned_face = recognizer.alignCrop(frame, faces[0])
                feature = recognizer.feature(aligned_face)
                
                # Save the feature vector
                np.save(os.path.join(DB_PATH, f"{name}.npy"), feature)
                print(f"Success! {name} has been added to the database.")
                break
            else:
                print("No face detected! Please try again.")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    enroll_via_webcam()