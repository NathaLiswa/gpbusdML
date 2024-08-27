from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Load the trained model
# Use __file__ (double underscores) instead of _file_ (single underscore)
main_dir = os.path.dirname(os.path.realpath(__file__))  # Corrected to __file__

# Construct the model path
model_path = os.path.join(main_dir, 'gbp_usd_model.pkl')

# Load the model
model = joblib.load(model_path)

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input data from the request
    data = request.get_json()

    # Create a DataFrame from the input data
    input_data = pd.DataFrame({
        'Previous_Close': [data['previous_close']],
        'Previous_High': [data['previous_high']],
        'Previous_Low': [data['previous_low']],
        'Previous_Open': [data['previous_open']]
    })

    # Make the prediction
    predicted_price = model.predict(input_data)[0]

    # Return the predicted price as a JSON response
    return jsonify({'predicted_price': predicted_price})

if __name__ == '__main__':
    app.run()