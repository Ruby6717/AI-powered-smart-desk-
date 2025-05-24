import cv2
import os

# === CONFIG ===
FACE_DIR = "faces"
CASCADE_PATH = "haarcascade_frontalface_default.xml"  # Make sure this file is in the same folder
CAM_INDEX = 0  # Default webcam

# === SETUP ===
os.makedirs(FACE_DIR, exist_ok=True)
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
cap = cv2.VideoCapture(CAM_INDEX)
face_id = 0

print("[INFO] Press 's' to save face | 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Failed to grab frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # Draw rectangles around faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Face Enrollment", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('s') and len(faces) > 0:
        for (x, y, w, h) in faces:
            face_img = gray[y:y + h, x:x + w]
            filename = os.path.join(FACE_DIR, f"user_{face_id}.jpg")
            cv2.imwrite(filename, face_img)
            print(f"[INFO] Saved {filename}")
            face_id += 1

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
