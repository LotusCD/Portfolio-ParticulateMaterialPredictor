from flask import Flask, request, jsonify
import joblib
import numpy as np

# Load the pre-trained model
model = joblib.load('pm10-from-pm2_5.pkl')

app = Flask(__name__)

@app.route('/')
def home():
    return "Welcome to the prediction API!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    print("Received data:", data)  # Add this line to debug
    
    if not data or 'pm2_5' not in data:
        return jsonify({'error': 'No input data provided'}), 400
    
    pm2_5_value = data['pm2_5']
    input_data = np.array([pm2_5_value]).reshape(1, -1)
    
    try:
        prediction = model.predict(input_data)
        return jsonify({'pm10_prediction': prediction[0]})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
