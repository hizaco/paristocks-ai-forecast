from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

def fetch_stock_data(ticker, days=60):
    """
    Fetch the last 'days' closing prices for a given ticker from Yahoo Finance.
    Returns list of closing prices or raises an exception if data cannot be fetched.
    """
    try:
        # Create ticker object
        stock = yf.Ticker(ticker)
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days + 10)  # Extra days to ensure we get enough data
        
        # Fetch historical data
        hist = stock.history(start=start_date, end=end_date)
        
        if hist.empty:
            raise ValueError(f"No data found for ticker '{ticker}'")
        
        # Get closing prices and convert to list
        closing_prices = hist['Close'].dropna().tail(days).tolist()
        
        if len(closing_prices) < 10:
            raise ValueError(f"Insufficient data for ticker '{ticker}'. Only {len(closing_prices)} days available.")
        
        return closing_prices
        
    except Exception as e:
        raise Exception(f"Error fetching data for ticker '{ticker}': {str(e)}")

def predict_prices(history, horizon_days):
    """
    Simple prediction using moving average for demonstration.
    Replace this with actual ML model in production.
    """
    # Use a simple moving average approach for demonstration
    if len(history) >= 5:
        # Calculate trend using last 5 days
        recent_trend = np.mean(np.diff(history[-5:]))
    else:
        recent_trend = 0
    
    last_price = history[-1]
    predictions = []
    
    for i in range(horizon_days):
        # Simple prediction: last price + trend + some noise
        predicted_price = last_price + (recent_trend * (i + 1)) + np.random.normal(0, last_price * 0.005)
        predictions.append(float(max(predicted_price, 0.01)))  # Ensure positive price
    
    return predictions

def get_forecast_summary(prices, horizons):
    summary = {}
    for label, days in horizons.items():
        summary[label] = prices[days-1] if days <= len(prices) else prices[-1]
    return summary

app = Flask(__name__)
CORS(app)

@app.route('/api/forecast', methods=['POST'])
def forecast():
    data = request.get_json()
    ticker = data.get('ticker')
    history = data.get('history')

    # If ticker is provided, fetch data from Yahoo Finance
    if ticker:
        try:
            # Fetch 60 days of historical data from Yahoo Finance
            history = fetch_stock_data(ticker.upper(), days=60)
        except Exception as e:
            return jsonify({'error': str(e)}), 400
    else:
        # If no ticker, use the provided history (existing behavior)
        if not history or len(history) < 10:
            return jsonify({'error': 'Not enough historical data. Provide at least 10 data points or a valid ticker.'}), 400

    forecast_horizons = {"1_day": 1, "1_week": 7, "1_month": 30, "1_year": 365}
    predictions = predict_prices(history, 365)
    summary = get_forecast_summary(predictions, forecast_horizons)

    detailed_analysis = {
        "trend": "bullish" if summary["1_year"] > history[-1] else "bearish",
        "volatility": float(np.std(history)),
        "predictions": summary
    }
    
    response = {
        "current_price": history[-1],
        "forecast": detailed_analysis
    }
    
    # Only include ticker in response if it was provided
    if ticker:
        response["ticker"] = ticker.upper()
    
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)