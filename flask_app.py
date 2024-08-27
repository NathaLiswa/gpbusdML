from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os
import yfinance as yf

app = Flask(__name__)

# Load the trained model
main_dir = os.path.dirname(os.path.realpath(__file__))  # Get the directory of the current file

# Construct the model path
model_path = os.path.join(main_dir, 'gbp_usd_model.pkl')

# Load the model
model = joblib.load(model_path)

@app.route('/predict', methods=['POST'])
def predict():
    # Fetch current data for GBP/USD
    ticker = yf.Ticker("GBPUSD=X")  # Use the appropriate ticker for GBP/USD
    current_data = ticker.history(period="1d")

    # Extract the latest OHLC data
    latest_data = current_data.iloc[-1]

    # Create a DataFrame from the latest OHLC data
    input_data = pd.DataFrame({
        'Previous_Close': [latest_data['Close']],  # Corrected 'Cose' to 'Close'
        'Previous_High': [latest_data['High']],
        'Previous_Low': [latest_data['Low']],
        'Previous_Open': [latest_data['Open']]
    })

    # Make the prediction
    predicted_price = model.predict(input_data)[0]

    # Return the predicted price as a JSON response
    return jsonify({'predicted_price': predicted_price})

if __name__ == '__main__':
    app.run()
