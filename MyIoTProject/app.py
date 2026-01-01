from flask import Flask, render_template, jsonify, request
import serial
import time
import numpy as np
import joblib
import threading
from datetime import datetime
import os

app = Flask(__name__)

# Global variables to store sensor data and prediction
sensor_data = {
    'rainfall': 0,
    'moisture': 0,
    'water_level': 0,
    'temperature': 0,
    'humidity': 0,
    'prediction': 0,
    'prediction_text': 'No data',
    'crop_type': 'Rice',
    'crop_code': 0,
    'last_update': 'Never',
    'is_connected': False,
    'serial_port': 'COM4'
}

# Load ML model and scaler
model = None
scaler = None
try:
    model = joblib.load('model.pkl')
    scaler = joblib.load('scaler.pkl')
    print("✓ ML model loaded successfully!")
except Exception as e:
    print(f"✗ Error loading model: {e}")
    print("Please run ml_predictor.py first to train the model.")
    print("Training model now...")
    
    # Import and run training
    from ml_predictor import train_model
    model, scaler = train_model()

# Serial reading thread
serial_thread = None
ser = None
stop_thread = False

def read_serial_data():
    """Background thread to read serial data"""
    global sensor_data, ser, stop_thread, model, scaler
    
    port = sensor_data['serial_port']
    baud_rate = 115200
    
    try:
        ser = serial.Serial(port, baud_rate, timeout=1)
        time.sleep(2)
        print(f"✓ Connected to {port}")
        sensor_data['is_connected'] = True
        
        while not stop_thread:
            if ser.in_waiting > 0:
                raw_line = ser.readline()
                try:
                    line = raw_line.decode('utf-8', errors='ignore').strip()
                except:
                    try:
                        line = raw_line.decode('latin-1', errors='ignore').strip()
                    except:
                        line = str(raw_line)
                
                # Debug: Uncomment to see raw data
                # print(f"Raw: [{line}]")
                
                if line.startswith("Data:"):
                    data = line[5:].strip()
                    parts = data.split(',')
                    
                    if len(parts) >= 5:
                        try:
                            rain = float(parts[0])
                            moisture = float(parts[1])
                            water = int(parts[2])
                            temp = float(parts[3])
                            hum = float(parts[4])
                            
                            # Update sensor data
                            sensor_data['rainfall'] = rain
                            sensor_data['moisture'] = moisture
                            sensor_data['water_level'] = water
                            sensor_data['temperature'] = temp
                            sensor_data['humidity'] = hum
                            
                            # ML Prediction
                            if model is not None and scaler is not None:
                                crop_code = sensor_data['crop_code']
                                input_data = np.array([[rain, moisture, crop_code]])
                                input_scaled = scaler.transform(input_data)
                                prediction = model.predict(input_scaled)[0]
                                
                                sensor_data['prediction'] = int(prediction)
                                sensor_data['prediction_text'] = "IRRIGATION NEEDED!" if prediction == 1 else "No Irrigation Required"
                            else:
                                sensor_data['prediction_text'] = "Model not loaded"
                            
                            sensor_data['last_update'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            
                            print(f"✓ Data updated: Rainfall={rain:.1f}mm, Moisture={moisture:.1f}%, Temp={temp:.1f}°C, Prediction={'IRRIGATE' if sensor_data['prediction'] == 1 else 'NO'}")
                            
                        except ValueError as e:
                            print(f"✗ Error parsing numbers: {e}")
                        except Exception as e:
                            print(f"✗ Error in prediction: {e}")
                    else:
                        print(f"✗ Incomplete data received: {parts}")
                elif line:  # Print other messages from Arduino
                    print(f"Arduino: {line}")
            
            time.sleep(0.5)
            
    except serial.SerialException as e:
        print(f"✗ Serial Port Error: {e}")
        sensor_data['is_connected'] = False
        sensor_data['prediction_text'] = f"Serial Error: {e}"
    except Exception as e:
        print(f"✗ Error in serial thread: {e}")
        sensor_data['is_connected'] = False

@app.route('/')
def index():
    """Render the main dashboard"""
    return render_template('index.html')

@app.route('/api/data')
def get_data():
    """API endpoint to get current sensor data"""
    return jsonify(sensor_data)

@app.route('/api/control/<command>')
def control_irrigation(command):
    """API endpoint to control irrigation"""
    global ser
    if command == 'start':
        # Send command to Arduino to start irrigation
        if ser and ser.is_open:
            ser.write(b'START\n')
            return jsonify({'status': 'success', 'message': 'Irrigation started'})
        else:
            return jsonify({'status': 'error', 'message': 'Serial not connected'})
    elif command == 'stop':
        # Send command to Arduino to stop irrigation
        if ser and ser.is_open:
            ser.write(b'STOP\n')
            return jsonify({'status': 'success', 'message': 'Irrigation stopped'})
        else:
            return jsonify({'status': 'error', 'message': 'Serial not connected'})
    elif command == 'test':
        # Test prediction with current data
        if model and scaler:
            rain = sensor_data['rainfall']
            moisture = sensor_data['moisture']
            crop_code = sensor_data['crop_code']
            input_data = np.array([[rain, moisture, crop_code]])
            input_scaled = scaler.transform(input_data)
            prediction = model.predict(input_scaled)[0]
            return jsonify({'status': 'success', 'prediction': int(prediction)})
        else:
            return jsonify({'status': 'error', 'message': 'Model not loaded'})
    
    return jsonify({'status': 'error', 'message': 'Unknown command'})

@app.route('/api/set_crop/<crop>')
def set_crop(crop):
    """Change crop type"""
    if crop == 'rice':
        sensor_data['crop_type'] = 'Rice'
        sensor_data['crop_code'] = 0
        return jsonify({'status': 'success', 'crop': 'Rice', 'code': 0})
    elif crop == 'jute':
        sensor_data['crop_type'] = 'Jute'
        sensor_data['crop_code'] = 1
        return jsonify({'status': 'success', 'crop': 'Jute', 'code': 1})
    return jsonify({'status': 'error', 'message': 'Invalid crop'})

@app.route('/api/manual_prediction', methods=['POST'])
def manual_prediction():
    """Make prediction with manual input"""
    try:
        data = request.json
        rain = float(data.get('rainfall', 0))
        moisture = float(data.get('moisture', 0))
        crop = data.get('crop', 'rice')
        crop_code = 0 if crop == 'rice' else 1
        
        if model and scaler:
            input_data = np.array([[rain, moisture, crop_code]])
            input_scaled = scaler.transform(input_data)
            prediction = model.predict(input_scaled)[0]
            
            return jsonify({
                'status': 'success',
                'prediction': int(prediction),
                'text': 'IRRIGATION NEEDED!' if prediction == 1 else 'No Irrigation Required',
                'input': {'rainfall': rain, 'moisture': moisture, 'crop': crop}
            })
        else:
            return jsonify({'status': 'error', 'message': 'Model not loaded'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

def start_serial_thread():
    """Start the serial reading thread"""
    global serial_thread, stop_thread
    stop_thread = False
    if serial_thread is None or not serial_thread.is_alive():
        serial_thread = threading.Thread(target=read_serial_data, daemon=True)
        serial_thread.start()
        print("✓ Serial thread started")

def stop_serial_thread():
    """Stop the serial reading thread"""
    global stop_thread
    stop_thread = True
    if ser and ser.is_open:
        ser.close()
        print("✓ Serial port closed")

@app.route('/api/restart_serial')
def restart_serial():
    """Restart serial connection"""
    stop_serial_thread()
    time.sleep(1)
    start_serial_thread()
    return jsonify({'status': 'success', 'message': 'Serial restarted'})

# Start serial thread when app starts
if __name__ == '__main__':
    start_serial_thread()
    print("\n" + "="*50)
    print("Smart Irrigation System Dashboard")
    print("="*50)
    print("Open your browser and go to: http://localhost:5000")
    print("Press Ctrl+C to stop the application")
    print("="*50 + "\n")
    
    try:
        app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        stop_serial_thread()