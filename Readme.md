This is a Particulate Material predictor using supervised learning.

Data taken from: #Calidad_del_Aire_Municipio_de_Duitama_20240302 - https://www.datos.gov.co/Ciencia-Tecnolog-a-e-Innovaci-n/Calidad-del-Aire-Municipio-de-Duitama/aghd-ge2f/about_data


## Here is the complete Flask application:

## 1. Directory Structure
```
project/
│
├── app.py
└── pm2_5-from-pm10.pkl
└── Jupyter Notebook
```

## 2. Flask Application (app.py)
python
Copy code
from flask import Flask, request, jsonify
import joblib
import numpy as np

# Load the pre-trained model

```
model = joblib.load('pm2_5-from-pm10.pkl')

app = Flask(__name__)

@app.route('/')
def home():
    return "Welcome to the PM2.5 to PM10 prediction API!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
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

```

## 3. Running the Flask Application
Navigate to your project directory and run the Flask app:

```
python app.py
```

## 4. Testing the Endpoint
You can test the endpoint using tools like Postman or curl. Here’s an example using curl:

```
curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d '{"pm2_5": 11.0}'

```

This should return a response with the predicted PM10 value:

```
{
  "pm10_prediction": <predicted_value>
}
```

Replace <predicted_value> with the actual value returned by your model.

This completes the setup of a simple Flask application that uses a pre-trained model to predict PM10 values based on PM2.5 input. You can further enhance this application by adding more features such as input validation, logging, and more sophisticated error handling.