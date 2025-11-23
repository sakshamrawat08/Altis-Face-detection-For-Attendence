

import cv2
import os
import sys

NAME = "Saksham Singh Rawat"


name = NAME.strip().replace(" ", "_")
print("Registering:", name)


folder = os.path.join("dataset", name)
os.makedirs(folder, exist_ok=True)


CASCADE_FILE = "haarcascade_frontalface_default.xml"
if not os.path.exists(CASCADE_FILE):
    print(f"ERROR: '{CASCADE_FILE}' not found in current folder. Download it from OpenCV repo and place it here.")
    sys.exit(1)


cam = cv2.VideoCapture(0)
if not cam.isOpened():
    print("ERROR: Could not open camera. Check macOS Camera permissions or close other apps using camera.")
    sys.exit(1)

cascade = cv2.CascadeClassifier(CASCADE_FILE)

count = 0
MAX_IMAGES = 25  

print("\nCamera started...")
print("Press 'c' to capture face image")
print("Press 'q' to quit\n")

while True:
    ret, frame = cam.read()
    if not ret:
        print("ERROR: Failed to read frame from camera.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Capture Images - press c to capture, q to quit", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("c") and len(faces) > 0:
        (x, y, w, h) = faces[0] 
        face = gray[y:y + h, x:x + w]
        try:
            face = cv2.resize(face, (200, 200))
        except Exception:
            pass
        filepath = os.path.join(folder, f"{name}_{count}.jpg")
        cv2.imwrite(filepath, face)
        count += 1
        print("Captured:", filepath)
        if count >= MAX_IMAGES:
            print("Reached max images:", MAX_IMAGES)
            break

    if key == ord("q"):
        print("Quit pressed by user.")
        break

cam.release()
cv2.destroyAllWindows()
print("\nDone capturing images! Total saved:", count)