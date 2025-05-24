import face_recognition
import cv2
import os

# Directory to store enrolled faces
known_faces_dir = "enrolled_faces"

# Make sure the directory exists
if not os.path.exists(known_faces_dir):
    os.makedirs(known_faces_dir)

# === Initialize webcam or ESP32 stream ===
# Change the URL to your ESP32 camera stream URL
url = 'http://192.168.1.104:81/stream'  # ESP32 camera stream URL
cap = cv2.VideoCapture(url)

if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

def enroll_face(frame, name):
    # Detect face and encode it
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    if face_encodings:
        encoding = face_encodings[0]

        # Save the image with the name as filename
        face_image = frame[face_locations[0][0]:face_locations[0][2], face_locations[0][3]:face_locations[0][1]]
        cv2.imwrite(os.path.join(known_faces_dir, f"{name}.jpg"), face_image)
        print(f"[LOG] Enrolled face for {name}")

print("[INFO] Press 'q' to quit, 'e' to enroll a new face.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Show the frame from ESP32 camera stream
    cv2.imshow("Camera Stream", frame)

    # Enroll a new face when 'e' is pressed
    if cv2.waitKey(1) & 0xFF == ord('e'):
        name = input("Enter name for this person: ")
        enroll_face(frame, name)
        print(f"[INFO] Face enrolled for {name}.")

    # Exit the system when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
