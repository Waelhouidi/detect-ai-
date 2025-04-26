# app.py (Flask Backend)
from flask import Flask, jsonify, request
from flask_socketio import SocketIO, emit
import sqlite3
from datetime import datetime

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")  # Allow connections from any origin

def init_db():
    conn = sqlite3.connect('employee_activity.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS activity (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            employee_id INTEGER,
            event_type TEXT,
            event_time TEXT,
            duration REAL,
            details TEXT
        )
    ''')
    
    # Optional: Create an employees table if you want to track employee info
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS employees (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            department TEXT,
            position TEXT
        )
    ''')
    conn.commit()
    conn.close()

# Initialize database on startup
init_db()

@app.route('/track_event', methods=['POST'])
def track_event():
    data = request.json
    employee_id = data['employee_id']
    event_type = data['event_type']
    event_time = data['event_time']
    duration = data.get('duration', 0)
    details = data.get('details', '')

    conn = sqlite3.connect('employee_activity.db')
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO activity (employee_id, event_type, event_time, duration, details)
        VALUES (?, ?, ?, ?, ?)
    ''', (employee_id, event_type, event_time, duration, details))
    conn.commit()
    conn.close()

    # Emit the event data to the frontend
    socketio.emit('new_event', {
        'employee_id': employee_id,
        'event_type': event_type,
        'event_time': event_time,
        'duration': duration,
        'details': details
    })

    return jsonify({"message": "Event tracked successfully"})

# Get employee activity summary
@app.route('/activity_summary/<int:employee_id>', methods=['GET'])
def activity_summary(employee_id):
    conn = sqlite3.connect('employee_activity.db')
    cursor = conn.cursor()
    
    # Get today's date
    today = datetime.now().strftime("%Y-%m-%d")
    
    # Query for today's activity
    cursor.execute('''
        SELECT event_type, SUM(duration) 
        FROM activity 
        WHERE employee_id = ? AND event_time LIKE ?
        GROUP BY event_type
    ''', (employee_id, f"{today}%"))
    
    results = cursor.fetchall()
    summary = {}
    
    for event_type, total_duration in results:
        summary[event_type] = total_duration if total_duration else 0
    
    conn.close()
    
    return jsonify({
        "employee_id": employee_id,
        "date": today,
        "summary": summary
    })

# Get all events for a specific employee
@app.route('/events/<int:employee_id>', methods=['GET'])
def get_events(employee_id):
    conn = sqlite3.connect('employee_activity.db')
    conn.row_factory = sqlite3.Row  # This enables column access by name
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT * FROM activity 
        WHERE employee_id = ? 
        ORDER BY event_time DESC
        LIMIT 100
    ''', (employee_id,))
    
    rows = cursor.fetchall()
    events = [dict(row) for row in rows]
    
    conn.close()
    
    return jsonify({
        "employee_id": employee_id,
        "events": events
    })

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)