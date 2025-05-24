import face_recognition
import cv2
import os
import numpy as np
from datetime import datetime
import csv

# === Setup ===
known_faces_dir = "enrolled_faces"
attendance_file = "attendance.csv"
known_encodings = []
known_names = []

# === Load known faces ===
for filename in os.listdir(known_faces_dir):
    if filename.endswith(".jpg"):
        img_path = os.path.join(known_faces_dir, filename)
        image = face_recognition.load_image_file(img_path)
        encoding = face_recognition.face_encodings(image)
        if encoding:
            known_encodings.append(encoding[0])
            known_names.append(filename.split('.')[0])

# === Initialize webcam ===
cap = cv2.VideoCapture(0)
marked_names = set()  # To avoid duplicate entries per session

def mark_attendance(name):
    if name not in marked_names:
        now = datetime.now()
        time_str = now.strftime("%H:%M:%S")
        date_str = now.strftime("%Y-%m-%d")
        with open(attendance_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([name, date_str, time_str])
        marked_names.add(name)
        print(f"[LOG] Marked attendance for {name} at {time_str}")

print("[INFO] Running Attendance System. Press 'q' to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)  # Speed boost
    rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_small)
    face_encodings = face_recognition.face_encodings(rgb_small, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.5)
        name = "Unknown"

        if True in matches:
            index = matches.index(True)
            name = known_names[index]
            mark_attendance(name)

        # Scale back up face location
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw rectangle and label
        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("Attendance System", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
