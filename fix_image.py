import cv2
import numpy as np
import time
import face_recognition
import pickle
from datetime import datetime
import os

# === Load Precomputed Face Encodings ===
print("[INFO] Loading known face encodings...")
with open("encodings.pkl", "rb") as f:
    data = pickle.load(f)

if isinstance(data, dict):
    known_encodings = data["encodings"]
    known_names = data["names"]
else:
    known_encodings, known_names = data

print(f"[INFO] Loaded {len(known_encodings)} known encodings.")

# === Attendance Tracker ===
attendance = set()

def mark_attendance(name):
    if name not in attendance:
        attendance.add(name)
        with open("attendance.csv", "a") as f:
            now = datetime.now()
            dt_string = now.strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"{name},{dt_string}\n")
            print(f"[MARKED] {name} at {dt_string}")

# === Face Detection Setup ===
print("Loading face detection models...")
try:
    modelFile = "res10_300x300_ssd_iter_140000.caffemodel"
    configFile = "deploy.prototxt"
    net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
    use_dnn = True
    print("Using DNN face detector")
except Exception as e:
    print(f"Warning: Could not load DNN model: {e}")
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    use_dnn = False

profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')

# === Video Setup ===
video_path = 'videos/videoyt-lokshab.mp4'
video_capture = cv2.VideoCapture(video_path)

if not video_capture.isOpened():
    print(f"Error: Could not open video at {video_path}")
    exit()

fps = video_capture.get(cv2.CAP_PROP_FPS)
width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

frame_skip = 1
detection_interval = 3
start_time = time.time()
frames_processed = 0
faces_detected = 0

cv2.namedWindow('Enhanced Face Detection', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Enhanced Face Detection', width, height)

def detect_faces(frame):
    faces = []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if use_dnn:
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), [104, 117, 123], False, False)
        net.setInput(blob)
        detections = net.forward()

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.6:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x1, y1, x2, y2) = box.astype("int")
                x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
                if x2 > x1 and y2 > y1 and (x2 - x1) * (y2 - y1) > 400:
                    faces.append((x1, y1, x2 - x1, y2 - y1, confidence))
    else:
        haar_faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        for (x, y, w, h) in haar_faces:
            faces.append((x, y, w, h, 0.7))

    if len(faces) < 2:
        profile_faces = profile_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        flipped = cv2.flip(gray, 1)
        flipped_profile_faces = profile_cascade.detectMultiScale(flipped, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        for (x, y, w, h) in profile_faces:
            faces.append((x, y, w, h, 0.6))
        frame_width = gray.shape[1]
        for (x, y, w, h) in flipped_profile_faces:
            new_x = frame_width - (x + w)
            faces.append((new_x, y, w, h, 0.6))

    return faces

print(f"Starting video processing: {width}x{height} at {fps} FPS")

try:
    frame_counter = 0
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        if frame_counter % frame_skip == 0:
            detected_faces = detect_faces(frame)
            faces_detected += len(detected_faces)

            rgb_frame = frame[:, :, ::-1]  # BGR to RGB

            for (x, y, w, h, conf) in detected_faces:
                name = "Unknown"

                # Slightly enlarge the face crop area to help encoding
                padding = 10
                y1 = max(0, y - padding)
                y2 = min(frame.shape[0], y + h + padding)
                x1 = max(0, x - padding)
                x2 = min(frame.shape[1], x + w + padding)

                face_crop = rgb_frame[y1:y2, x1:x2]
                encodings = face_recognition.face_encodings(face_crop)

                if encodings:
                    face_encoding = encodings[0]
                    matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.5)
                    face_distances = face_recognition.face_distance(known_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = known_names[best_match_index]
                        mark_attendance(name)

                color = (0, int(255 * conf), int(255 * (1 - conf)))
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, f"{name} {conf:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            elapsed_time = time.time() - start_time
            fps_text = f"FPS: {frames_processed / max(1, elapsed_time):.1f}"
            cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            progress = f"Frame: {frame_counter}/{total_frames} ({100*frame_counter/max(1,total_frames):.1f}%)"
            cv2.putText(frame, progress, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow('Enhanced Face Detection', frame)
            frames_processed += 1

        frame_counter += 1
        delay = max(1, int(1000 / fps))
        key = cv2.waitKey(delay)
        if key == 27:
            break
        elif key == ord('s'):
            frame_counter += int(fps * 5)
            video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_counter)

except Exception as e:
    print(f"Error during processing: {e}")
finally:
    video_capture.release()
    cv2.destroyAllWindows()
    print("\n--- Performance Summary ---")
    print(f"Total frames processed: {frames_processed}")
    print(f"Total faces detected: {faces_detected}")
    print(f"Average faces per frame: {faces_detected / max(1, frames_processed):.2f}")
    print(f"Processing time: {time.time() - start_time:.2f} seconds")
    print(f"Average FPS: {frames_processed / max(1, time.time() - start_time):.2f}")