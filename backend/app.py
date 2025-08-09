from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np

# Try to import yfinance, but handle ImportError gracefully for testing
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    print("Warning: yfinance not available. Ticker-based fetching will be disabled.")

def fetch_stock_data(ticker, days=60):
    """
    Fetch stock data from Yahoo Finance for the given ticker.
    Returns the closing prices for the last 'days' days.
    """
    if not YFINANCE_AVAILABLE:
        raise Exception("yfinance library not available")
    
    try:
        # Create ticker object
        stock = yf.Ticker(ticker)
        
        # Fetch historical data for the specified period
        # Using period instead of specific dates for simplicity
        hist = stock.history(period=f"{days}d")
        
        if hist.empty:
            raise Exception(f"No data found for ticker {ticker}")
        
        # Extract closing prices and convert to list
        closing_prices = hist['Close'].tolist()
        
        if len(closing_prices) < 10:
            raise Exception(f"Not enough historical data for ticker {ticker}")
        
        return closing_prices
        
    except Exception as e:
        raise Exception(f"Failed to fetch data for {ticker}: {str(e)}")

def predict_prices(history, horizon_days):
    last_price = history[-1]
    predictions = []
    for i in range(horizon_days):
        noise = np.random.normal(0, last_price * 0.01)
        new_price = last_price * (1 + np.random.normal(0.001, 0.01)) + noise
        predictions.append(float(new_price))
        last_price = new_price
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
    data_source = "user_provided"

    # If ticker is provided, try to fetch data from Yahoo Finance
    if ticker and ticker.strip() and YFINANCE_AVAILABLE:
        try:
            # Fetch stock data using yfinance
            history = fetch_stock_data(ticker.strip(), days=60)
            data_source = "yahoo_finance"
            print(f"Successfully fetched {len(history)} data points for {ticker}")
        except Exception as e:
            # If fetching fails, fallback to provided history if available
            if not history or len(history) < 10:
                return jsonify({'error': f'Failed to fetch data for ticker {ticker}: {str(e)}'}), 400
            else:
                print(f"Failed to fetch data for {ticker}, using provided history: {str(e)}")
                data_source = "user_provided_fallback"
    
    # Validate we have enough historical data
    if not history or len(history) < 10:
        if ticker and ticker.strip() and not YFINANCE_AVAILABLE:
            return jsonify({'error': 'yfinance library not available. Please provide historical data or install yfinance to fetch data automatically.'}), 400
        else:
            return jsonify({'error': 'Not enough historical data. Please provide a valid ticker or at least 10 historical prices.'}), 400

    forecast_horizons = {"1_day": 1, "1_week": 7, "1_month": 30, "1_year": 365}
    predictions = predict_prices(history, 365)
    summary = get_forecast_summary(predictions, forecast_horizons)

    detailed_analysis = {
        "trend": "bullish" if summary["1_year"] > history[-1] else "bearish",
        "volatility": float(np.std(history)),
        "predictions": summary
    }
    
    response_data = {
        "ticker": ticker,
        "current_price": history[-1],
        "forecast": detailed_analysis,
        "data_source": data_source
    }
    
    # Add metadata about data points if fetched from Yahoo Finance
    if data_source == "yahoo_finance":
        response_data["data_points"] = len(history)
    
    return jsonify(response_data)

if __name__ == '__main__':
    app.run(debug=True)