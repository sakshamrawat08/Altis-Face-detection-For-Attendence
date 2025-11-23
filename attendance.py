import cv2
import json
import pandas as pd
import datetime
import os

TRAINER_FILE = "trainer/trainer.yml"
LABELS_FILE = "trainer/labels.json"
CASCADE_FILE = "haarcascade_frontalface_default.xml"
ATTEND_FILE = "attendance.csv"
IMG_SIZE = (200, 200)
CONF_THRESHOLD = 60  # lower = stricter match

if not os.path.exists(TRAINER_FILE) or not os.path.exists(LABELS_FILE):
    print("ERROR: Train model first using train_model.py")
    exit(1)

# load model and labels
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(TRAINER_FILE)

with open(LABELS_FILE, "r") as f:
    labels = {int(k): v for k, v in json.load(f).items()}

# haar cascade
if not os.path.exists(CASCADE_FILE):
    print(f"ERROR: {CASCADE_FILE} not found.")
    exit(1)
cascade = cv2.CascadeClassifier(CASCADE_FILE)

# attendance file
if not os.path.exists(ATTEND_FILE):
    pd.DataFrame(columns=["name", "date", "time"]).to_csv(ATTEND_FILE, index=False)

import pandas as pd
from pandas.errors import EmptyDataError

# ensure CSV exists and has correct header
if not os.path.exists(ATTEND_FILE):
    pd.DataFrame(columns=["name","date","time"]).to_csv(ATTEND_FILE, index=False)

# now safely load it
try:
    df = pd.read_csv(ATTEND_FILE)
except EmptyDataError:
    # recreate with header if file was empty/corrupt
    df = pd.DataFrame(columns=["name","date","time"])
    df.to_csv(ATTEND_FILE, index=False)


cam = cv2.VideoCapture(0)
if not cam.isOpened():
    print("ERROR: Could not open camera.")
    exit(1)

print("Attendance started. Press 'q' to stop.")

while True:
    ret, frame = cam.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, 1.2, 5)

    for (x, y, w, h) in faces:
        face = gray[y:y + h, x:x + w]
        face = cv2.resize(face, IMG_SIZE)
        face = cv2.equalizeHist(face)

        label, conf = recognizer.predict(face)

        if conf < CONF_THRESHOLD:
            name = labels.get(label, "Unknown").replace("_", " ")
        else:
            name = "Unknown"

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f"{name} {int(conf)}", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        if name != "Unknown":
            today = datetime.date.today().isoformat()
            if not ((df["name"] == name) & (df["date"] == today)).any():
                now = datetime.datetime.now().strftime("%H:%M:%S")
                df.loc[len(df)] = [name, today, now]
                df.to_csv(ATTEND_FILE, index=False)
                print("Marked:", name)

    cv2.imshow("Attendance", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
print("Attendance stopped.")
