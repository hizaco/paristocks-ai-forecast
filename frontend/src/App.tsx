import React, { useState } from "react";
import axios from "axios";
import { Container, TextField, Button, Typography, Box } from "@mui/material";

function App() {
  const [ticker, setTicker] = useState("");
  const [history, setHistory] = useState("");
  const [result, setResult] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async () => {
    setError(null);
    setResult(null);
    try {
      const prices = history.split(",").map((val) => parseFloat(val.trim())).filter(x => !isNaN(x));
      const res = await axios.post("http://localhost:5000/api/forecast", {
        ticker,
        history: prices,
      });
      setResult(res.data);
    } catch (err: any) {
      setError(err?.response?.data?.error || "Erreur API");
    }
  };

  return (
    <Container maxWidth="sm" sx={{ mt: 4 }}>
      <Typography variant="h4" gutterBottom>
        Paris Stocks AI Forecast
      </Typography>
      <TextField
        label="Ticker"
        value={ticker}
        onChange={e => setTicker(e.target.value)}
        fullWidth
        sx={{ mb: 2 }}
      />
      <TextField
        label="Historique des prix (séparés par des virgules)"
        value={history}
        onChange={e => setHistory(e.target.value)}
        fullWidth
        sx={{ mb: 2 }}
      />
      <Button variant="contained" onClick={handleSubmit}>
        Prédire
      </Button>
      {error && <Typography color="error" sx={{ mt: 2 }}>{error}</Typography>}
      {result && (
        <Box sx={{ mt: 3 }}>
          <Typography variant="h6">Résultat :</Typography>
          <pre>{JSON.stringify(result, null, 2)}</pre>
        </Box>
      )}
    </Container>
  );
}

export default App;