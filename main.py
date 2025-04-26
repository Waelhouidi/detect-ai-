# main.py
import cv2
import requests
from deepface import DeepFace
from ultralytics import YOLO
import time
import numpy as np
from datetime import datetime

# Setup Flask backend URL
API_URL = "http://127.0.0.1:5000/track_event"

cap = cv2.VideoCapture(0)
model = YOLO('yolov8n.pt')

# Tracking variables
phone_usage_start = None
phone_usage_duration = 0
phone_usage_total = 0
last_report_time = time.time()
report_interval = 60  # Report every minute

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
        
        # Report significant emotion changes (optional)
        # This could be enhanced to only report when emotion changes from previous state
        event_data = {
            'employee_id': employee_id,
            'event_type': 'emotion',
            'event_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'duration': 0,
            'details': emotion  # You might need to add a 'details' field to your database
        }
        try:
            requests.post(API_URL, json=event_data)
        except requests.exceptions.RequestException as e:
            print(f"Failed to send emotion data: {e}")
    except Exception as e:
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
        current_usage = time.time() - phone_usage_start
        phone_usage_duration += current_usage
        phone_usage_total += current_usage
        phone_usage_start = None
        
        # Report phone usage when it stops or at regular intervals
        current_time = time.time()
        if current_time - last_report_time > report_interval or current_usage > 10:
            event_data = {
                'employee_id': employee_id,
                'event_type': 'phone_usage',
                'event_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'duration': phone_usage_total,
                'details': 'Detected phone usage'
            }
            try:
                response = requests.post(API_URL, json=event_data)
                print(f"Phone usage reported: {phone_usage_total:.2f} seconds")
                phone_usage_total = 0  # Reset after reporting
                last_report_time = current_time
            except requests.exceptions.RequestException as e:
                print(f"Failed to send phone usage data: {e}")
    
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
                'duration': 0,
                'details': 'Employee entered the frame'
            }
            try:
                requests.post(API_URL, json=event_data)
            except requests.exceptions.RequestException as e:
                print(f"Failed to send enter_frame data: {e}")
    else:
        if employee_present:
            print("Employee left the frame.")
            employee_present = False
            event_data = {
                'employee_id': employee_id,
                'event_type': 'leave_frame',
                'event_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'duration': 0,
                'details': 'Employee left the frame'
            }
            try:
                requests.post(API_URL, json=event_data)
            except requests.exceptions.RequestException as e:
                print(f"Failed to send leave_frame data: {e}")
    
    prev_frame = gray
    cv2.putText(frame, f"Usage: {phone_usage_duration:.2f}s", (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Employee {'Present' if employee_present else 'Absent'}", (20, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    cv2.imshow("Employee Monitor", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        # Report final phone usage before exiting
        if phone_usage_total > 0:
            event_data = {
                'employee_id': employee_id,
                'event_type': 'phone_usage',
                'event_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'duration': phone_usage_total,
                'details': 'Final phone usage report'
            }
            try:
                requests.post(API_URL, json=event_data)
            except requests.exceptions.RequestException:
                pass
        break

cap.release()
cv2.destroyAllWindows()