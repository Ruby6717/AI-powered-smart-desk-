import cv2
import os

# === Setup ===
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
cap = cv2.VideoCapture(0)
enroll_dir = "enrolled_faces"
os.makedirs(enroll_dir, exist_ok=True)

name = input("Enter name for enrollment: ")
print("[INFO] Press 's' to capture | 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    cv2.imshow("Enroll Face", frame)
    key = cv2.waitKey(1)

    if key == ord('s') and len(faces) > 0:
        for (x, y, w, h) in faces:
            face_img = frame[y:y+h, x:x+w]
            filepath = os.path.join(enroll_dir, f"{name}.jpg")
            cv2.imwrite(filepath, face_img)
            print(f"[INFO] Face saved as {filepath}")
            break
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
