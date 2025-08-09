from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np

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

    if not history or len(history) < 10:
        return jsonify({'error': 'Not enough historical data.'}), 400

    forecast_horizons = {"1_day": 1, "1_week": 7, "1_month": 30, "1_year": 365}
    predictions = predict_prices(history, 365)
    summary = get_forecast_summary(predictions, forecast_horizons)

    detailed_analysis = {
        "trend": "bullish" if summary["1_year"] > history[-1] else "bearish",
        "volatility": float(np.std(history)),
        "predictions": summary
    }
    return jsonify({
        "ticker": ticker,
        "current_price": history[-1],
        "forecast": detailed_analysis
    })

if __name__ == '__main__':
    app.run(debug=True)