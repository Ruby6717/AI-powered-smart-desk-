import cv2

# ESP32 Camera URL (replace with your ESP32 IP address)
url = 'http://192.168.1.104:81/stream'  # You can also try '/video' or '/mjpeg'

# Open the video stream from ESP32
cap = cv2.VideoCapture(url)

if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Display the captured frame
    cv2.imshow('ESP32 Camera Stream', frame)

    # Wait for a key press, if 'q' is pressed, exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
