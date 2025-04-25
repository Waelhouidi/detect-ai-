import cv2
import requests
from deepface import DeepFace
from ultralytics import YOLO
import time
import numpy as np
from datetime import datetime

# Setup Flask backend URL
API_URL = "http://127.0.0.1:5000/log_activity"

cap = cv2.VideoCapture(0)
model = YOLO('yolov8n.pt')

# Tracking variables
phone_usage_start = None
phone_usage_duration = 0
employee_present = True
employee_id = 1  # Example: Employee ID, you can set it dynamically based on your system

# Motion detection params
motion_threshold = 50000
prev_frame = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # --- Emotion Detection ---
    try:
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        emotion = result[0]['dominant_emotion']
        cv2.putText(frame, f"Emotion: {emotion}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    except:
        pass

    # --- Phone Detection ---
    results = model(frame)[0]
    phone_detected = False

    for box in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = box
        class_id = int(class_id)
        label = results.names[class_id]

        if label.lower() == 'cell phone' and score > 0.5:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            cv2.putText(frame, f"Phone Detected", (int(x1), int(y1)-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            if phone_usage_start is None:
                phone_usage_start = time.time()
            phone_detected = True

    if not phone_detected and phone_usage_start is not None:
        phone_usage_duration += time.time() - phone_usage_start
        phone_usage_start = None

        # Send phone usage data to backend
        event_data = {
            'employee_id': employee_id,
            'event_type': 'phone_usage',
            'event_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'duration': phone_usage_duration
        }
        response = requests.post(API_URL, json=event_data)
        print(response.json())

    # --- Motion Detection ---
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    if prev_frame is None:
        prev_frame = gray
        continue

    frame_diff = cv2.absdiff(prev_frame, gray)
    _, threshold = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)
    non_zero_count = np.count_nonzero(threshold)

    if non_zero_count > motion_threshold:
        if not employee_present:
            print("Employee entered the frame.")
            employee_present = True
            event_data = {
                'employee_id': employee_id,
                'event_type': 'enter_frame',
                'event_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'duration': 0
            }
            requests.post(API_URL, json=event_data)
    else:
        if employee_present:
            print("Employee left the frame.")
            employee_present = False
            event_data = {
                'employee_id': employee_id,
                'event_type': 'leave_frame',
                'event_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'duration': 0
            }
            requests.post(API_URL, json=event_data)

    prev_frame = gray

    # Show frame and logs
    cv2.putText(frame, f"Usage: {phone_usage_duration:.2f}s", (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Employee {'Present' if employee_present else 'Absent'}", (20, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Employee Monitor", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
