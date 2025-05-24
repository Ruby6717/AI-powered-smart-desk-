import face_recognition
import cv2
import os
import numpy as np
from datetime import datetime
import csv
import pygame

# === Setup ===
known_faces_dir = "enrolled_faces"
attendance_file = "attendance.csv"
alert_sound_path = "alert.wav"  # Your alert sound

# === Initialize pygame mixer ===
pygame.mixer.init()
alert_sound = pygame.mixer.Sound(alert_sound_path)

known_encodings = []
known_names = []

# === Load Known Faces ===
for person_name in os.listdir(known_faces_dir):
    person_folder = os.path.join(known_faces_dir, person_name)
    if os.path.isdir(person_folder):
        for filename in os.listdir(person_folder):
            if filename.endswith(".jpg"):
                img_path = os.path.join(person_folder, filename)
                image = face_recognition.load_image_file(img_path)
                encodings = face_recognition.face_encodings(image)
                if encodings:
                    known_encodings.append(encodings[0])
                    known_names.append(person_name)

print(f"[INFO] Loaded encodings for: {set(known_names)}")

# === Attendance helper ===
marked_names = set()
def mark_attendance(name):
    if name not in marked_names:
        now = datetime.now()
        time_str = now.strftime("%H:%M:%S")
        date_str = now.strftime("%Y-%m-%d")
        with open(attendance_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([name, date_str, time_str])
        marked_names.add(name)
        print(f"[LOG] Authenticated: {name} at {date_str} {time_str}")

# === Open ESP32 Camera Stream ===
url = 'http://192.168.227.169:81/stream'
cap = cv2.VideoCapture(url)

if not cap.isOpened():
    print("[ERROR] Could not open ESP32 camera stream.")
    exit()

print("[INFO] Starting face detection with motion tracking. Press 'q' to quit.")

# === Define cabin/desk area (x1, y1, x2, y2) ===
# Adjust these values based on your frame size and desk area
CABIN_AREA = (100, 100, 500, 400)  # Example: a box within which person should stay

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Frame not received.")
        break

    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_small)
    face_encodings = face_recognition.face_encodings(rgb_small, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.45)
        name = "Unknown"

        if True in matches:
            match_index = matches.index(True)
            name = known_names[match_index]
            mark_attendance(name)
        else:
            pygame.mixer.Sound.play(alert_sound)

        # Scale coordinates back to full frame
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # === Motion detection: check if face is outside cabin area ===
        face_center_x = (left + right) // 2
        face_center_y = (top + bottom) // 2
        x1, y1, x2, y2 = CABIN_AREA

        if name != "Unknown":
            if not (x1 < face_center_x < x2 and y1 < face_center_y < y2):
                print(f"[ALERT] {name} moved out of the designated area!")
                pygame.mixer.Sound.play(alert_sound)

        # === Draw rectangle and text ===
        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # Draw the cabin area box
    cv2.rectangle(frame, (CABIN_AREA[0], CABIN_AREA[1]), (CABIN_AREA[2], CABIN_AREA[3]), (255, 255, 0), 2)
    cv2.putText(frame, "Cabin Area", (CABIN_AREA[0], CABIN_AREA[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    cv2.imshow("Face Attendance + Motion Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("[INFO] Quitting system.")
        break

cap.release()
cv2.destroyAllWindows()
