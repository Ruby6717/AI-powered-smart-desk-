import cv2

# ESP32 camera stream URL (replace with your actual stream URL)
url = 'http://192.168.1.104:81/stream'

# Open the video stream
cap = cv2.VideoCapture(url)

# Check if the stream is opened successfully
if not cap.isOpened():
    print("[ERROR] Could not open ESP32 camera stream.")
else:
    print("[INFO] Successfully opened the ESP32 camera stream.")
    
    # Test if a frame can be read from the stream
    ret, frame = cap.read()
    
    if not ret:
        print("[ERROR] Could not read frame from the stream.")
    else:
        print("[INFO] Successfully read a frame from the stream.")
        # Optionally display the frame (for testing purposes)
        cv2.imshow("ESP32 Camera Stream", frame)
        cv2.waitKey(0)  # Wait for a key press to close the window
        
    cap.release()

cv2.destroyAllWindows()
