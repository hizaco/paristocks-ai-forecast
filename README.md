# ParisStocks AI Forecast

Ce projet fournit une API Flask pour la prédiction de cours boursiers et une interface web React pour visualiser les résultats.

## Lancer le backend

```bash
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python app.py
```

## Lancer le frontend

```bash
cd frontend
npm install
npm start
```

L'interface est accessible sur `http://localhost:3000`, le backend sur `http://localhost:5000`.

## Utilisation de l'API

### Option 1: Avec un ticker Yahoo Finance (nouveau)

```bash
curl -X POST http://localhost:5000/api/forecast \
  -H "Content-Type: application/json" \
  -d '{"ticker": "AAPL"}'
```

L'API télécharge automatiquement les 60 derniers cours de clôture quotidiens via Yahoo Finance.

### Option 2: Avec des données historiques manuelles (existant)

```bash
curl -X POST http://localhost:5000/api/forecast \
  -H "Content-Type: application/json" \
  -d '{"history": [13.5, 13.6, 13.8, 14.0, 14.2, 14.5, 14.8, 15.0, 15.3, 15.5, 15.7, 15.8]}'
```

### Réponse de l'API

```json
{
  "ticker": "AAPL",
  "current_price": 150.25,
  "forecast": {
    "trend": "bullish",
    "volatility": 2.45,
    "predictions": {
      "1_day": 151.20,
      "1_week": 153.80,
      "1_month": 158.90,
      "1_year": 195.30
    }
  }
}
```

## Exemples d'utilisation

### Dans l'interface web
Renseignez un ticker (ex : ACA.PA, AAPL, GOOGL) pour une prédiction automatique.

### Avec des données manuelles
Renseignez une série d'historiques de prix (ex : `13.5, 13.6, 13.8, 14.0, 14.2, 14.5, 14.8, 15.0, 15.3, 15.5, 15.7, 15.8`).

---

**Fonctionnalités :**
- ✅ Intégration Yahoo Finance via yfinance
- ✅ Téléchargement automatique de 60 jours de données
- ✅ Gestion d'erreurs pour tickers invalides
- ✅ Rétrocompatibilité avec les données manuelles
- ✅ Prédictions basées sur une moyenne mobile (démo)

**À personnaliser :**
- Remplacer la fonction `predict_prices` par un vrai modèle ML.
- Améliorer l'UI pour plus de détails graphiques.