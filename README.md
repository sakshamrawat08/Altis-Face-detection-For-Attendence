# Face Recognition Attendance System

A Python-based face recognition attendance system that captures face images, trains a recognition model, detects registered faces, and records attendance in CSV format.

## Features

- Capture face images using a webcam
- Train a face recognition model from collected images
- Real-time face detection and recognition
- Automatic attendance logging
- Admin panel for attendance-related operations
- CSV-based attendance records

## Tech Stack

- Python
- OpenCV
- Haar Cascade Classifier
- Face recognition / LBPH workflow
- CSV

## Project Structure

```text
.
├── admin_panel_enhanced.py
├── attendance.py
├── capture_images.py
├── train_model.py
├── attendance.csv
└── haarcascade_frontalface_default.xml
```

## How It Works

1. Run `capture_images.py` to collect face samples.
2. Run `train_model.py` to train the recognition model.
3. Run `attendance.py` to recognize faces and record attendance.
4. Use the admin panel for attendance management.

## Setup

Create a virtual environment and install the required Python packages:

```bash
python -m venv venv
source venv/bin/activate
pip install opencv-contrib-python
```

On Windows:

```bash
venv\Scripts\activate
pip install opencv-contrib-python
```

## Notes

The Haar Cascade XML file is included in the repository. Keep generated datasets, trained model files, and other local runtime artifacts out of version control when they contain personal data.

## Future Improvements

- Replace CSV storage with a database
- Add role-based authentication
- Add attendance analytics and dashboards
- Improve recognition accuracy and liveness checks
- Add Docker support and automated tests

## Author

**Saksham Singh Rawat**  
GitHub: [@sakshamrawat08](https://github.com/sakshamrawat08)
