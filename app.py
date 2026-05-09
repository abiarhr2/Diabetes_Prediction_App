from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load model and scaler
model = pickle.load(open('diabetes_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# Home route 
@app.route('/')
def home():
    return render_template('index.html')

#prediction toute
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # Extract features from the request
        features = [
            float(data['Pregnancies']),
            float(data['Glucose']),
            float(data['BloodPressure']),
            float(data['SkinThickness']),
            float(data['Insulin']),
            float(data['BMI']),
            float(data['DiabetesPedigreeFunction']),
            float(data['Age'])
        ]

        #convert to numpy and reshape
        input_array = np.asarray(features).reshape(1,-1)

        #standardize the input
        input_scaler = scaler.transform(input_array)

        #make prediction
        prediction = model.predict(input_scaler)[0]

        #return the result
        return jsonify({
            'prediction': int(prediction),
            'status': 'success'
        })
    
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        })
if __name__ == "__main__":
    app.run(debug=False, port='0.0.0.0')