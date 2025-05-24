import face_recognition
import cv2
import os

# === CONFIGURATION ===
known_faces_dir = "enrolled_faces"
url = 'http://192.168.1.104:81/stream'  # Replace with your ESP32 stream URL
save_limit = 5  # How many images to save per person

# === Create main face directory if not exist ===
if not os.path.exists(known_faces_dir):
    os.makedirs(known_faces_dir)

# === Ask for name ===
person_name = input("Enter the name of the person to enroll: ").strip()
person_folder = os.path.join(known_faces_dir, person_name)
os.makedirs(person_folder, exist_ok=True)

# === Start ESP32 stream ===
cap = cv2.VideoCapture(url)
if not cap.isOpened():
    print("[ERROR] Could not open ESP32 video stream.")
    exit()

print(f"[INFO] Enrolling faces for: {person_name}")
print("[INFO] Press 'q' to quit.")

saved_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Lost connection to the camera stream. Reconnecting...")
        cap.release()
        cap = cv2.VideoCapture(url)
        continue

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)

    for i, (top, right, bottom, left) in enumerate(face_locations):
        # Crop face
        face_image = frame[top:bottom, left:right]
        face_filename = os.path.join(person_folder, f"{person_name}_{saved_count+1}.jpg")
        cv2.imwrite(face_filename, face_image)
        print(f"[SAVED] {face_filename}")
        saved_count += 1

        # Draw rectangle around face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        if saved_count >= save_limit:
            print(f"[INFO] Saved {save_limit} face images for {person_name}.")
            break

    cv2.imshow("Enrollment - Press 'q' to Quit", frame)

    if saved_count >= save_limit:
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
