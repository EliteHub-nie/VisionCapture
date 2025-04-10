import cv2
import face_recognition
import numpy as np
import os
import pickle
from datetime import datetime

# Config
KNOWN_FACES_DIR = "loksabha-img"
VIDEO_PATH = "videos/videoyt-lokshab.mp4"
ENCODINGS_FILE = "encodings.pkl"
ATTENDANCE_FILE = "attendance.csv"

# Load DNN face detection model (OpenCV SSD)
modelFile = "res10_300x300_ssd_iter_140000.caffemodel"
configFile = "deploy.prototxt"
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

# Load or encode known faces
known_encodings = []
known_names = []

if os.path.exists(ENCODINGS_FILE):
    with open(ENCODINGS_FILE, "rb") as f:
        known_encodings, known_names = pickle.load(f)
    print("✅ Loaded known face encodings from cache")
else:
    print("🔄 Encoding known faces...")
    for filename in os.listdir(KNOWN_FACES_DIR):
        if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
            name = os.path.splitext(filename)[0]
            path = os.path.join(KNOWN_FACES_DIR, filename)
            image = face_recognition.load_image_file(path)
            encodings = face_recognition.face_encodings(image)
            if encodings:
                known_encodings.append(encodings[0])
                known_names.append(name)
                print(f"✅ Encoded: {filename}")
            else:
                print(f"❌ No face found in {filename}, skipping.")
    with open(ENCODINGS_FILE, "wb") as f:
        pickle.dump((known_encodings, known_names), f)
    print(f"✅ Encoding completed and saved to '{ENCODINGS_FILE}'")

# Track attendance
attendance = {}

def mark_attendance(name):
    if name not in attendance:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        attendance[name] = now
        print(f"📝 Marked {name} at {now}")

# Start video
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"❌ Error: Could not open video at {VIDEO_PATH}")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
frame_skip = 3
frame_count = 0

print("🎥 Starting face detection and recognition...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % frame_skip == 0:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], False, False)
        net.setInput(blob)
        detections = net.forward()
        
        face_locations = []

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x1, y1, x2, y2) = box.astype("int")
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                face_locations.append((y1, x2, y2, x1))  # top, right, bottom, left

        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.5)
            name = "Unknown"

            if True in matches:
                match_index = matches.index(True)
                name = known_names[match_index]
                mark_attendance(name)

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow("Face Recognition Attendance", frame)

    frame_count += 1
    if cv2.waitKey(1) & 0xFF == 27:  # ESC key
        break

# Save attendance to CSV
with open(ATTENDANCE_FILE, "w") as f:
    f.write("Name,Time\n")
    for name, timestamp in attendance.items():
        f.write(f"{name},{timestamp}\n")

cap.release()
cv2.destroyAllWindows()
print("\n✅ Attendance saved to 'attendance.csv'")
