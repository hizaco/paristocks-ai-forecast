import React, { useState } from "react";

function App() {
  const [ticker, setTicker] = useState("BNP.PA");
  const [history, setHistory] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleForecast = async () => {
    setLoading(true);
    setResult(null);
    const historyArray = history.split(",").map(Number).filter(x => !isNaN(x));
    const res = await fetch("http://localhost:5000/api/forecast", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ticker, history: historyArray }),
    });
    const data = await res.json();
    setResult(data);
    setLoading(false);
  };

  return (
    <div style={{ maxWidth: 600, margin: "40px auto", fontFamily: "sans-serif" }}>
      <h1>Paris Stocks AI Forecast</h1>
      <p>Entrez un ticker boursier et une série de valeurs (ex: 66.4,66.7,66.2,...)</p>
      <input
        placeholder="Ticker (ex: BNP.PA)"
        value={ticker}
        onChange={e => setTicker(e.target.value)}
      />
      <br />
      <textarea
        placeholder="Historique des prix, séparés par des virgules"
        value={history}
        onChange={e => setHistory(e.target.value)}
        rows={3}
        style={{ width: "100%", marginTop: 8 }}
      />
      <br />
      <button onClick={handleForecast} disabled={loading}>
        {loading ? "Analyse en cours..." : "Analyser"}
      </button>
      {result && (
        <div style={{ marginTop: 20 }}>
          <h2>Résultats pour {result.ticker}</h2>
          <p>Prix actuel : {result.current_price} €</p>
          <ul>
            <li>Prévision 1 jour : {result.forecast.predictions["1_day"].toFixed(2)} €</li>
            <li>Prévision 1 semaine : {result.forecast.predictions["1_week"].toFixed(2)} €</li>
            <li>Prévision 1 mois : {result.forecast.predictions["1_month"].toFixed(2)} €</li>
            <li>Prévision 1 an : {result.forecast.predictions["1_year"].toFixed(2)} €</li>
          </ul>
          <p>
            Tendance : <b>{result.forecast.trend === "bullish" ? "Haussière 📈" : "Baissière 📉"}</b>
            <br />
            Volatilité historique : {result.forecast.volatility.toFixed(2)}
          </p>
        </div>
      )}
    </div>
  );
}

export default App;