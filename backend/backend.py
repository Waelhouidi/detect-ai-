from flask import Flask, jsonify, request
from flask_socketio import SocketIO, emit
import sqlite3
from datetime import datetime

app = Flask(__name__)
socketio = SocketIO(app)

# Existing route to track events
@app.route('/track_event', methods=['POST'])
def track_event():
    data = request.json
    employee_id = data['employee_id']
    event_type = data['event_type']
    event_time = data['event_time']
    duration = data.get('duration')

    conn = sqlite3.connect('employee_activity.db')
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO activity (employee_id, event_type, event_time, duration)
        VALUES (?, ?, ?, ?)
    ''', (employee_id, event_type, event_time, duration))
    conn.commit()
    conn.close()

    # Emit the event data to the frontend
    socketio.emit('new_event', {
        'employee_id': employee_id,
        'event_type': event_type,
        'event_time': event_time,
        'duration': duration
    })

    return jsonify({"message": "Event tracked successfully"})

if __name__ == '__main__':
    socketio.run(app, debug=True)
