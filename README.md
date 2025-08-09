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

L’interface est accessible sur `http://localhost:3000`, le backend sur `http://localhost:5000`.

## Exemple d’utilisation

Dans l’interface, renseignez un ticker (ex : ACA.PA) et une série d’historiques de prix (ex : `13.5, 13.6, 13.8, 14.0, 14.2, 14.5, 14.8, 15.0, 15.3, 15.5, 15.7, 15.8`).

---

**À personnaliser :**
- Remplacer la fonction `predict_prices` par un vrai modèle ML.
- Améliorer l’UI pour plus de détails graphiques.