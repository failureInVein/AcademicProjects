import serial
import time
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib

# ========================= CONFIG =========================
port = 'COM4'          # আপনার পোর্ট (COM3/COM4/ইত্যাদি)
baud_rate = 115200
# =========================================================

def train_model():
    """Train ML model and save it"""
    print("Training ML model...")
    
    np.random.seed(42)
    n_samples = 3000
    
    rainfall = np.random.uniform(0, 100, n_samples)
    soil_moisture = np.random.uniform(0, 100, n_samples)
    crop_type = np.random.choice([0, 1], n_samples)  # 0: Rice, 1: Jute
    
    irrigation_needed = []
    for i in range(n_samples):
        r = rainfall[i]
        m = soil_moisture[i]
        c = crop_type[i]
        
        if c == 0:  # Rice
            irrigation_needed.append(1 if (r < 15 and m < 45) else 0)
        else:       # Jute
            irrigation_needed.append(1 if (r < 10 and m < 35) else 0)
    
    X = np.column_stack((rainfall, soil_moisture, crop_type))
    y = np.array(irrigation_needed)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = DecisionTreeClassifier(
        max_depth=5,
        min_samples_split=10,
        random_state=42,
        class_weight='balanced'
    )
    model.fit(X_train_scaled, y_train)
    
    accuracy = accuracy_score(y_test, model.predict(X_test_scaled))
    print(f"ML Model Trained! Accuracy: {accuracy * 100:.2f}%")
    
    # Save model and scaler
    joblib.dump(model, 'model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    print("Model and scaler saved as model.pkl & scaler.pkl")
    
    return model, scaler

def predict_single(rain, moisture, crop_type=0, model=None, scaler=None):
    """Make prediction for single data point"""
    if model is None or scaler is None:
        try:
            model = joblib.load('model.pkl')
            scaler = joblib.load('scaler.pkl')
        except:
            print("Error loading model. Training new model...")
            model, scaler = train_model()
    
    input_data = np.array([[rain, moisture, crop_type]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    
    return prediction

if __name__ == "__main__":
    # Train the model once
    train_model()
    
    # For real-time prediction, run the Flask app instead
    print("\nModel trained successfully!")
    print("Run 'python app.py' to start the web interface.")