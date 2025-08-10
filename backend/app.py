from __future__ import annotations

import os
import time
import math
import ssl
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

DEFAULT_LOOKBACK_DAYS = int(os.getenv("LOOKBACK_DAYS", "540"))
MAX_FORECAST_DAYS = 252
REQUEST_TIMEOUT = 10
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
RISK_FREE = float(os.getenv("RISK_FREE", "0.0"))

YAHOO_CHART = "https://query1.finance.yahoo.com/v8/finance/chart"
YAHOO_SEARCH = "https://query2.finance.yahoo.com/v1/finance/search"

DEFAULT_BENCH = "^GSPC"
EU_BENCH = "^FCHI"

ssl._create_default_https_context = ssl._create_unverified_context  # fallback

class TTLCache:
    def __init__(self, ttl_seconds: int = 120):
        self.ttl = ttl_seconds
        self._data: Dict[str, Tuple[float, Any]] = {}

    def get(self, key: str):
        v = self._data.get(key)
        if not v:
            return None
        ts, val = v
        if time.time() - ts > self.ttl:
            self._data.pop(key, None)
            return None
        return val

    def set(self, key: str, value: Any):
        self._data[key] = (time.time(), value)

cache = TTLCache(ttl_seconds=120)

def safe_get(url: str, params: Optional[dict] = None, timeout: int = REQUEST_TIMEOUT, allow_insecure_fallback: bool = True):
    headers = {"User-Agent": USER_AGENT}
    try:
        r = requests.get(url, params=params, headers=headers, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as e1:
        if allow_insecure_fallback:
            try:
                r = requests.get(url, params=params, headers=headers, timeout=timeout, verify=False)
                r.raise_for_status()
                return r.json()
            except Exception as e2:
                print(f"[safe_get] Fallback failed {url}: {e1} // {e2}")
                return None
        print(f"[safe_get] Failed {url}: {e1}")
        return None

def try_import_yf():
    try:
        import yfinance as yf  # type: ignore
        return yf
    except Exception as e:
        print(f"[yfinance] not available: {e}")
        return None

def normalize_symbol(user_symbol: str) -> List[str]:
    s = user_symbol.strip()
    su = s.upper()
    variations = [su]
    if "." not in su:
        variations += [f"{su}.PA", f"{su}.DE", f"{su}.L"]
    special_map = {
        "BNP": "BNP.PA",
        "BNPP": "BNP.PA",
        "TOTAL": "TTE.PA",
        "TOTALENERGIES": "TTE.PA",
        "LVMH": "MC.PA",
        "LOREAL": "OR.PA",
        "SANOFI": "SAN.PA",
        "AIRBUS": "AIR.PA",
        "DANONE": "BN.PA",
        "ORANGE": "ORA.PA",
        "SOCIETE GENERALE": "GLE.PA",
        "CREDIT AGRICOLE": "ACA.PA",
        "CARREFOUR": "CA.PA",
        "RENAULT": "RNO.PA",
        "PEUGEOT": "UG.PA",
    }
    if su in special_map:
        variations.insert(0, special_map[su])
    out, seen = [], set()
    for v in variations:
        if v not in seen:
            out.append(v); seen.add(v)
    return out

def pick_benchmark(symbol: str) -> str:
    return EU_BENCH if (symbol.endswith(".PA") or symbol.endswith(".FR")) else DEFAULT_BENCH

def pandas_from_yahoo_http(symbol: str, days: int) -> Optional[pd.DataFrame]:
    end = int(datetime.now().timestamp())
    start = int((datetime.now() - timedelta(days=days)).timestamp())
    url = f"{YAHOO_CHART}/{symbol}"
    params = {"period1": start, "period2": end, "interval": "1d", "includePrePost": "false", "events": "div%2Csplits"}
    data = safe_get(url, params)
    try:
        if not data or "chart" not in data or not data["chart"]["result"]:
            return None
        result = data["chart"]["result"][0]
        ts = result.get("timestamp", [])
        q = result.get("indicators", {}).get("quote", [{}])[0]
        if not ts or not q:
            return None
        df = pd.DataFrame(q)
        df["Date"] = pd.to_datetime(ts, unit="s")
        df = df.rename(columns={"open":"Open","high":"High","low":"Low","close":"Close","volume":"Volume"})
        df = df[["Date","Open","High","Low","Close","Volume"]].dropna(subset=["Open","Close"]).reset_index(drop=True)
        return df
    except Exception as e:
        print(f"[yahoo_http] {symbol} failed: {e}")
        return None

def pandas_from_yfinance(symbol: str, days: int) -> Optional[pd.DataFrame]:
    yf = try_import_yf()
    if yf is None:
        return None
    try:
        period_days = min(max(days, 30), 1825)
        df = yf.download(symbol, period=f"{period_days}d", interval="1d", auto_adjust=False, progress=False, threads=False)
        if df is None or df.empty:
            return None
        df = df.reset_index().rename(columns={"Open":"Open","High":"High","Low":"Low","Close":"Close","Volume":"Volume"})
        df = df[["Date","Open","High","Low","Close","Volume"]].dropna(subset=["Open","Close"])
        return df
    except Exception as e:
        print(f"[yfinance] {symbol} failed: {e}")
        return None

def get_price_df(symbol: str, days: int = DEFAULT_LOOKBACK_DAYS):
    ck = f"px:{symbol}:{days}"
    cached = cache.get(ck)
    if cached is not None:
        return cached
    df = pandas_from_yfinance(symbol, days)
    meta_name, curr = None, None
    if df is not None and not df.empty:
        yf = try_import_yf()
        if yf:
            try:
                t = yf.Ticker(symbol)
                meta_name = getattr(t, "fast_info", {}).get("shortName") or getattr(t, "info", {}).get("longName") or getattr(t, "info", {}).get("shortName")
                curr = getattr(t, "fast_info", {}).get("currency")
            except Exception:
                pass
        cache.set(ck, (df, meta_name, curr))
        return df, meta_name, curr
    df = pandas_from_yahoo_http(symbol, days)
    if df is not None and not df.empty:
        cache.set(ck, (df, meta_name, curr))
        return df, meta_name, curr
    cache.set(ck, (None, None, None))
    return None, None, None

def ema(series: np.ndarray, span: int) -> np.ndarray:
    alpha = 2 / (span + 1.0)
    out = np.zeros_like(series, dtype=float)
    out[0] = series[0]
    for i in range(1, len(series)):
        out[i] = alpha * series[i] + (1 - alpha) * out[i - 1]
    return out

def compute_indicators(df: pd.DataFrame) -> Dict[str, Any]:
    closes = df["Close"].values.astype(float)
    highs = df["High"].values.astype(float)
    lows = df["Low"].values.astype(float)
    n = len(df)
    out: Dict[str, Any] = {}

    def sma(arr, w):
        return float(pd.Series(arr).rolling(w).mean().iloc[-1]) if len(arr) >= w else None
    out["sma_20"] = sma(closes, 20)
    out["sma_50"] = sma(closes, 50)
    out["sma_200"] = sma(closes, 200)

    if n >= 35:
        ema12 = ema(closes, 12)
        ema26 = ema(closes, 26)
        macd = ema12 - ema26
        signal = ema(macd, 9)
        hist = macd - signal
        out["ema_12"] = float(ema12[-1]); out["ema_26"] = float(ema26[-1])
        out["macd"] = float(macd[-1]); out["macd_signal"] = float(signal[-1]); out["macd_hist"] = float(hist[-1])

    if n >= 15:
        deltas = np.diff(closes)
        gains = np.clip(deltas, 0, None)
        losses = -np.clip(deltas, None, 0)
        avg_gain = pd.Series(gains).rolling(14).mean().iloc[-1] if len(gains) >= 14 else gains.mean()
        avg_loss = pd.Series(losses).rolling(14).mean().iloc[-1] if len(losses) >= 14 else losses.mean()
        rs = (avg_gain / avg_loss) if avg_loss > 1e-12 else np.inf
        rsi = 100 - (100 / (1 + rs))
        out["rsi"] = float(np.nan_to_num(rsi, nan=50.0))

    if n >= 20:
        s = pd.Series(closes).rolling(20)
        mid = s.mean().iloc[-1]
        std = s.std(ddof=0).iloc[-1]
        out["bb_middle"] = float(mid); out["bb_upper"] = float(mid + 2 * std); out["bb_lower"] = float(mid - 2 * std)

    if n >= 15:
        prev_close = np.concatenate([[closes[0]], closes[:-1]])
        tr = np.maximum.reduce([highs - lows, np.abs(highs - prev_close), np.abs(lows - prev_close)])
        atr = pd.Series(tr).rolling(14).mean().iloc[-1]
        out["atr"] = float(atr)

    if n >= 30:
        ret = pd.Series(closes).pct_change().dropna()
        mu_d = float(ret.mean()); sigma_d = float(ret.std(ddof=0))
        out["mu_daily"] = mu_d; out["sigma_daily"] = sigma_d; out["volatility_annual"] = float(sigma_d * math.sqrt(252))
        neg = ret[ret < 0]; downside = float(neg.std(ddof=0)) if len(neg) > 0 else 0.0
        sharpe = ((mu_d * 252) - RISK_FREE) / (sigma_d * math.sqrt(252)) if sigma_d > 1e-9 else 0.0
        sortino = ((mu_d * 252) - RISK_FREE) / (downside * math.sqrt(252)) if downside > 1e-9 else 0.0
        cum = (1 + ret).cumprod(); peak = cum.cummax(); dd = (cum / peak) - 1.0; max_dd = float(dd.min())
        out["sharpe"] = float(sharpe); out["sortino"] = float(sortino); out["max_drawdown"] = max_dd
    return out

def compute_beta(df: pd.DataFrame, bench_df: pd.DataFrame) -> Optional[float]:
    try:
        a = pd.Series(df["Close"]).pct_change().dropna()
        b = pd.Series(bench_df["Close"]).pct_change().dropna()
        joined = pd.concat([a, b], axis=1).dropna()
        if len(joined) < 30:
            return None
        cov = np.cov(joined.iloc[:, 0], joined.iloc[:, 1])[0, 1]
        var_b = np.var(joined.iloc[:, 1])
        if var_b <= 1e-12:
            return None
        return float(cov / var_b)
    except Exception:
        return None

def monte_carlo(closes: np.ndarray, horizon_days: List[int], sims: int = 2000):
    ret = pd.Series(closes).pct_change().dropna()
    mu = float(ret.mean()); sigma = float(ret.std(ddof=0))
    last = float(closes[-1])
    out: Dict[int, Dict[str, float]] = {}
    if sigma < 1e-9:
        for h in horizon_days:
            out[h] = {"p5": last, "p50": last, "p95": last}
        return out
    for h in horizon_days:
        steps = max(1, min(h, MAX_FORECAST_DAYS))
        dt = 1.0
        shocks = np.random.normal(loc=(mu - 0.5 * sigma * sigma) * dt, scale=sigma * math.sqrt(dt), size=(sims, steps))
        paths = last * np.exp(np.cumsum(shocks, axis=1))
        terminal = paths[:, -1]
        out[h] = {"p5": float(np.percentile(terminal, 5)), "p50": float(np.percentile(terminal, 50)), "p95": float(np.percentile(terminal, 95))}
    return out

def build_scores(ind: Dict[str, Any], price: float):
    scores: Dict[str, float] = {}
    rsi = ind.get("rsi")
    if rsi is not None:
        scores["rsi"] = 85 if rsi < 30 else 65 if rsi < 50 else 45 if rsi < 70 else 20
    macd, sig, mh = ind.get("macd"), ind.get("macd_signal"), ind.get("macd_hist", 0.0)
    if macd is not None and sig is not None:
        scores["macd"] = 75 if macd > sig else 30
    s20, s50 = ind.get("sma_20"), ind.get("sma_50")
    if s20 and s50:
        if price > s20 > s50: scores["trend"] = 80
        elif price > s20: scores["trend"] = 60
        elif price < s20 and s50 and s20 < s50: scores["trend"] = 25
        else: scores["trend"] = 45
    elif s20:
        scores["trend"] = 55 if price > s20 else 40
    if macd is not None and sig is not None:
        if macd > sig and mh > 0: scores["momentum"] = 75
        elif macd > sig: scores["momentum"] = 60
        elif mh > 0: scores["momentum"] = 55
        else: scores["momentum"] = 35
    vol_ann = ind.get("volatility_annual")
    if vol_ann is not None:
        scores["risk"] = 75 if vol_ann < 0.15 else 60 if vol_ann < 0.25 else 45 if vol_ann < 0.35 else 30
    weights = {"rsi":0.25, "macd":0.2, "trend":0.25, "momentum":0.15, "risk":0.15}
    total = sum(scores[k]*w for k,w in weights.items() if k in scores)
    wsum = sum(w for k,w in weights.items() if k in scores)
    final_score = round((total/wsum) if wsum>0 else 50.0, 1)
    if final_score >= 70: rec, action = "ACHAT FORT", "buy_strong"
    elif final_score >= 60: rec, action = "ACHAT", "buy"
    elif final_score >= 50: rec, action = "NEUTRE", "hold"
    elif final_score >= 40: rec, action = "VENTE", "sell"
    else: rec, action = "VENTE FORTE", "sell_strong"
    return scores, final_score, rec, action

def analysis_summary(ind: Dict[str, Any], score: float, current: float) -> List[str]:
    lines = []
    rsi = ind.get("rsi")
    if rsi is not None:
        lines.append("RSI en zone de survente: potentiel de rebond." if rsi < 30 else ("RSI en zone de surachat: risque de repli accru." if rsi > 70 else "RSI neutre."))
    macd, sig, hist = ind.get("macd"), ind.get("macd_signal"), ind.get("macd_hist")
    if macd is not None and sig is not None:
        lines.append("MACD haussier." if macd > sig else "MACD baissier.")
    if hist is not None:
        lines.append("Momentum positif (histogramme > 0)." if hist > 0 else "Momentum négatif (histogramme < 0).")
    s200 = ind.get("sma_200")
    if s200:
        lines.append("Cours au-dessus de la SMA200 (tendance LT positive)." if current > s200 else "Cours sous la SMA200 (tendance LT fragile).")
    vol = ind.get("volatility_annual")
    if vol is not None:
        lines.append(f"Volatilité annualisée: {vol:.2%}.")
    sharpe = ind.get("sharpe")
    if sharpe is not None:
        lines.append(f"Sharpe: {sharpe:.2f}.")
    mdd = ind.get("max_drawdown")
    if mdd is not None:
        lines.append(f"Max drawdown: {mdd:.1%}.")
    return lines

def df_to_series_payload(df: pd.DataFrame, tail: int = 360):
    dft = df.tail(tail)
    return [
        {
            "date": d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d),
            "open": float(o),
            "high": float(h),
            "low": float(l),
            "close": float(c),
            "volume": int(v) if not pd.isna(v) else 0,
        }
        for d, o, h, l, c, v in zip(dft["Date"], dft["Open"], dft["High"], dft["Low"], dft["Close"], dft["Volume"])
    ]

@app.route("/api/forecast", methods=["POST"])
def forecast():
    body = request.get_json(force=True, silent=True) or {}
    ticker = (body.get("ticker") or "").strip()
    days = int(body.get("days") or DEFAULT_LOOKBACK_DAYS)
    if not ticker:
        return jsonify({"error": "Veuillez fournir un ticker."}), 400

    candidates = normalize_symbol(ticker)
    last_error = None
    df = None; used_symbol = None; name = None; currency = None
    for sym in candidates:
        df, name, currency = get_price_df(sym, days=days)
        if df is not None and not df.empty:
            used_symbol = sym; break
        last_error = f"Aucune donnée pour {sym}"

    if df is None or df.empty or used_symbol is None:
        return jsonify({"error": f"Aucune donnée valide pour '{ticker}'. Dernier essai: {last_error}"}), 400

    df = df.sort_values("Date").reset_index(drop=True)
    history_payload = df_to_series_payload(df)

    bench_symbol = pick_benchmark(used_symbol)
    bench_df, _, _ = get_price_df(bench_symbol, days=days)
    beta = compute_beta(df, bench_df) if bench_df is not None else None

    ind = compute_indicators(df)
    if beta is not None:
        ind["beta"] = beta

    current_price = float(df["Close"].iloc[-1])
    prev_close = float(df["Close"].iloc[-2]) if len(df) >= 2 else current_price
    change = current_price - prev_close
    change_pct = (change / prev_close) if prev_close else 0.0

    detailed_scores, final_score, reco, action = build_scores(ind, current_price)

    horizons = [1, 5, 21, 63, 252]
    mc = monte_carlo(df["Close"].values, horizons, sims=2000)
    preds = {
        "1_day": round(mc[1]["p50"], 2),
        "1_week": round(mc[5]["p50"], 2),
        "1_month": round(mc[21]["p50"], 2),
        "3_month": round(mc[63]["p50"], 2),
        "1_year": round(mc[252]["p50"], 2),
    }
    bands = {
        "1_day": {"p5": round(mc[1]["p5"], 2), "p50": round(mc[1]["p50"], 2), "p95": round(mc[1]["p95"], 2)},
        "1_week": {"p5": round(mc[5]["p5"], 2), "p50": round(mc[5]["p50"], 2), "p95": round(mc[5]["p95"], 2)},
        "1_month": {"p5": round(mc[21]["p5"], 2), "p50": round(mc[21]["p50"], 2), "p95": round(mc[21]["p95"], 2)},
        "3_month": {"p5": round(mc[63]["p5"], 2), "p50": round(mc[63]["p50"], 2), "p95": round(mc[63]["p95"], 2)},
        "1_year": {"p5": round(mc[252]["p5"], 2), "p50": round(mc[252]["p50"], 2), "p95": round(mc[252]["p95"], 2)},
    }

    if len(df) >= 20:
        recent_high = float(df["High"].tail(20).max())
        recent_low = float(df["Low"].tail(20).min())
    else:
        recent_high, recent_low = current_price * 1.05, current_price * 0.95

    summary = analysis_summary(ind, final_score, current_price)

    response = {
        "ticker": used_symbol,
        "name": name or used_symbol,
        "currency": currency or ("EUR" if used_symbol.endswith(".PA") else "USD"),
        "current_price": round(current_price, 4),
        "previous_close": round(prev_close, 4),
        "change": round(change, 4),
        "change_pct": round(change_pct, 4),
        "history": history_payload,
        "forecast": {
            "predictions": preds,
            "bands": bands,
            "trend": "bullish" if final_score >= 60 else ("neutral" if final_score >= 50 else "bearish"),
        },
        "ai_analysis": {
            "recommendation": reco,
            "action": action,
            "confidence": "Haute" if final_score >= 70 else ("Moyenne-Haute" if final_score >= 60 else ("Moyenne" if final_score >= 50 else "Moyenne-Basse")),
            "score": final_score,
            "detailed_scores": detailed_scores,
            "price_targets": {
                "upside": round(recent_high * 1.02, 2),
                "downside": round(recent_low * 0.98, 2),
                "current": round(current_price, 2),
            },
            "key_levels": {"resistance": round(recent_high, 2), "support": round(recent_low, 2)},
            "analysis_summary": summary,
        },
        "technical_indicators": {k: (round(v, 4) if isinstance(v, (int, float)) else v) for k, v in ind.items()},
        "benchmark": bench_symbol,
        "data_points": len(df),
        "generated_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    return jsonify(response)

@app.route("/api/symbols", methods=["GET"])
def symbols():
    q = (request.args.get("q") or "").strip()
    if not q:
        return jsonify({"items": []})
    params = {"q": q, "lang": "fr-FR", "region": "FR", "quotesCount": 10}
    data = safe_get(YAHOO_SEARCH, params=params)
    items = []
    if data and "quotes" in data:
        for it in data["quotes"][:10]:
            if not it.get("symbol"): continue
            items.append({"symbol": it.get("symbol"), "name": it.get("shortname") or it.get("longname") or it.get("symbol"), "exch": it.get("exchDisp"), "type": it.get("typeDisp")})
    return jsonify({"items": items})

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "time": datetime.utcnow().isoformat() + "Z"})

@app.route("/api/test", methods=["GET"])
def test_connection():
    sample = request.args.get("symbol", "AAPL")
    df, _, _ = get_price_df(sample, days=60)
    return jsonify({"symbol": sample, "status": "OK" if df is not None and not df.empty else "FAILED"})

if __name__ == "__main__":
    print("Starting Enhanced Stock Analysis API on :5001")
    app.run(host="0.0.0.0", port=5001, debug=True)
