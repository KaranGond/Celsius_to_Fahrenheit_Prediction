import numpy as np
from flask import Flask, request, render_template
import tensorflow as tf

# Initialize Flask app
app = Flask(__name__)

# Assuming "model.h5" is in the same directory as your script or notebook
loaded_model = tf.keras.models.load_model("Model/model.h5")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the single input value from the form
        input_value = float(request.form['temperature'])
        
        # Convert to the form [[a]] for input to the model
        feature = np.array([[input_value]])

        # Use the loaded model for prediction
        prediction = loaded_model.predict(feature)

        # Assuming the output is a single value, extract it
        output = round(prediction[0][0], 2)

        return render_template('index.html', prediction_text='Percent with heart disease is {}'.format(output))
    except ValueError:
        return render_template('index.html', prediction_text='Invalid input. Please enter a numeric value.')

if __name__ == "__main__":
    app.run(debug=True)
