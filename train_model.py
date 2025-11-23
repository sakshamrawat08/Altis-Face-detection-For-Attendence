import cv2
import numpy as np
import os
import json

DATASET_DIR = "dataset"
TRAINER_DIR = "trainer"
IMG_SIZE = (200, 200)

faces = []
labels = []
label_map = {}
label_id = 0

if not os.path.exists(DATASET_DIR):
    print("Dataset folder not found. Capture images first.")
    exit(1)

for person in sorted(os.listdir(DATASET_DIR)):
    pdir = os.path.join(DATASET_DIR, person)
    if not os.path.isdir(pdir):
        continue

    print("Reading images for:", person)
    label_map[label_id] = person

    for imgname in sorted(os.listdir(pdir)):
        imgpath = os.path.join(pdir, imgname)
        im = cv2.imread(imgpath, cv2.IMREAD_GRAYSCALE)
        if im is None:
            continue

        im = cv2.resize(im, IMG_SIZE)
        im = cv2.equalizeHist(im)

        faces.append(im)
        labels.append(label_id)

    label_id += 1

if len(faces) == 0:
    print("No images found in dataset.")
    exit(1)

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.train(faces, np.array(labels))

os.makedirs(TRAINER_DIR, exist_ok=True)
recognizer.write(os.path.join(TRAINER_DIR, "trainer.yml"))

with open(os.path.join(TRAINER_DIR, "labels.json"), "w") as f:
    json.dump(label_map, f)

print("Training done. Labels:", label_map)
