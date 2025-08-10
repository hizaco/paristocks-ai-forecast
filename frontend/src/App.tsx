import React, { useEffect, useMemo, useState } from "react";
import {
  TrendingUp,
  TrendingDown,
  Target,
  AlertCircle,
  BarChart3,
  Activity,
  ShieldCheck,
  LineChart as LineChartIcon,
  Search,
  Eye,
  EyeOff,
} from "lucide-react";
import {
  ResponsiveContainer,
  ComposedChart,
  Line,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  Brush,
  ReferenceLine,
} from "recharts";

type Suggestion = { symbol: string; name: string; exch?: string };
type Bands = { p5: number; p50: number; p95: number };
type Forecast = {
  bands?: Record<"1_day" | "1_week" | "1_month" | "3_month" | "1_year", Bands>;
  predictions?: Record<string, number>;
  trend?: string;
};
type AIAnalysis = {
  recommendation?: string;
  action?: "buy_strong" | "buy" | "hold" | "sell" | "sell_strong";
  confidence?: string;
  score?: number;
  detailed_scores?: Record<string, number>;
  price_targets?: { upside?: number; downside?: number; current?: number };
  key_levels?: { resistance?: number; support?: number };
  analysis_summary?: string[];
};
type APIResult = {
  name?: string;
  ticker?: string;
  currency?: string;
  current_price?: number;
  change?: number;
  change_pct?: number;
  benchmark?: string;
  data_points?: number;
  history?: { date: string; close: number }[];
  forecast?: Forecast;
  ai_analysis?: AIAnalysis;
  technical_indicators?: Record<string, number>;
};

const API_BASE =
  (import.meta as any)?.env?.VITE_API_BASE ||
  (import.meta as any)?.env?.REACT_APP_API_BASE ||
  "http://localhost:5001";

function formatPct(p?: number | null) {
  if (p === null || p === undefined || isNaN(p)) return "—";
  return (p * 100).toFixed(2) + "%";
}
function formatNum(n?: number | null, digits = 2) {
  if (n === null || n === undefined || isNaN(n)) return "—";
  return Number(n).toFixed(digits);
}
const toTs = (dateStr: string) => new Date(dateStr + "T00:00:00").getTime();
const addDaysTs = (ts: number, n: number) => {
  const d = new Date(ts);
  d.setDate(d.getDate() + n);
  return d.getTime();
};

function App() {
  const [ticker, setTicker] = useState("BNP.PA");
  const [suggestions, setSuggestions] = useState<Suggestion[]>([]);
  const [showSug, setShowSug] = useState(false);
  const [result, setResult] = useState<APIResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState("");

  const [showHistory, setShowHistory] = useState(true);
  const [showForecast, setShowForecast] = useState(true);
  const [showBands, setShowBands] = useState(true);

  useEffect(() => {
    const handler = setTimeout(async () => {
      if (!ticker || ticker.length < 2) {
        setSuggestions([]);
        return;
      }
      try {
        const res = await fetch(`${API_BASE}/api/symbols?q=${encodeURIComponent(ticker)}`);
        const data = await res.json();
        setSuggestions(data.items || []);
      } catch {
        setSuggestions([]);
      }
    }, 250);
    return () => clearTimeout(handler);
  }, [ticker]);

  const handlePick = (s: Suggestion) => {
    setTicker(s.symbol);
    setShowSug(false);
  };

  const handleAnalyze = async () => {
    setLoading(true);
    setErr("");
    setResult(null);
    try {
      const res = await fetch(`${API_BASE}/api/forecast`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ticker }),
      });
      const data = (await res.json()) as APIResult | { error?: string };
      if (!res.ok) {
        setErr((data as any).error || "Erreur lors de l'analyse.");
      } else {
        setResult(data as APIResult);
      }
    } catch {
      setErr("Erreur de connexion à l'API");
    } finally {
      setLoading(false);
    }
  };

  const getRecommendationColor = (action?: string) => {
    switch (action) {
      case "buy_strong":
        return "bg-green-600";
      case "buy":
        return "bg-green-500";
      case "hold":
        return "bg-yellow-500";
      case "sell":
        return "bg-red-500";
      case "sell_strong":
        return "bg-red-600";
      default:
        return "bg-gray-500";
    }
  };

  const getRecommendationIcon = (action?: string) => {
    if (action?.includes("buy")) return <TrendingUp className="w-6 h-6" />;
    if (action?.includes("sell")) return <TrendingDown className="w-6 h-6" />;
    return <Activity className="w-6 h-6" />;
  };

  const currency = result?.currency || (result?.ticker?.endsWith(".PA") ? "EUR" : "USD");

  // Données pour le graphe (time scale + padding à droite)
  const ChartData = useMemo(() => {
    if (!result?.history) return [] as { ts: number; close?: number; forecast_p50?: number; forecast_p5?: number; forecast_p95?: number }[];
    const hist = result.history || [];
    const histPts = hist.map((d) => ({ ts: toTs(d.date), close: Number(d.close) }));

    const lastTs = histPts.length ? histPts[histPts.length - 1].ts : Date.now();
    const bands = result.forecast?.bands || {};
    const steps: { days: number; key: keyof NonNullable<Forecast["bands"]> }[] = [
      { days: 1, key: "1_day" },
      { days: 5, key: "1_week" },
      { days: 21, key: "1_month" },
      { days: 63, key: "3_month" },
      { days: 252, key: "1_year" },
    ].filter((p) => bands && (bands as any)[p.key]) as any;

    const fpts = steps.map((p) => {
      const ts = addDaysTs(lastTs, p.days);
      const b = (bands as any)[p.key] as Bands;
      return { ts, forecast_p50: b.p50, forecast_p5: b.p5, forecast_p95: b.p95 };
    });

    return [...histPts, ...fpts];
  }, [result]);

  const todayTs = useMemo(() => {
    const d = new Date();
    d.setHours(0, 0, 0, 0);
    return d.getTime();
  }, []);

  const xDomain = useMemo<[number | "dataMin", number | "dataMax"]>(() => {
    if (!ChartData.length) return ["dataMin", "dataMax"];
    const dataMin = Math.min(...ChartData.map((d) => d.ts));
    const dataMax = Math.max(...ChartData.map((d) => d.ts));
    const has1y = Boolean(result?.forecast?.bands?.["1_year"]);
    const padDays = has1y ? Math.round(252 * 0.2) : 45;
    const padMs = padDays * 24 * 3600 * 1000;
    return [dataMin, dataMax + padMs];
  }, [ChartData, result]);

  return (
    <div className="min-h-screen bg-gradient-to-b from-slate-50 to-white text-slate-900">
      <div className="max-w-7xl mx-auto p-6">
        <div className="flex items-center justify-between mb-4">
          <h1 className="text-2xl font-bold flex items-center gap-2">
            <Target className="w-6 h-6 text-blue-600" />
            Analyse IA des Actions
          </h1>
          <div className="text-gray-600 flex items-center gap-2">
            <ShieldCheck className="w-5 h-5 text-emerald-600" />
            Fiabilité renforcée (Monte Carlo, indicateurs avancés)
          </div>
        </div>

        <div className="flex gap-2 relative">
          <div className="flex-1 relative">
            <div className="flex items-center border rounded px-2 bg-white shadow-sm">
              <Search className="w-4 h-4 text-gray-500" />
              <input
                value={ticker}
                onChange={(e) => {
                  setTicker(e.target.value);
                  setShowSug(true);
                }}
                onFocus={() => setShowSug(true)}
                className="px-2 py-2 flex-1 outline-none"
                placeholder="Ex: AAPL, TSLA, BNP.PA, MC.PA"
              />
            </div>
            {showSug && suggestions.length > 0 && (
              <div className="absolute z-10 bg-white border rounded w-full mt-1 max-h-64 overflow-auto shadow">
                {suggestions.map((s, idx) => (
                  <div
                    key={idx}
                    onMouseDown={() => handlePick(s)}
                    className="px-3 py-2 hover:bg-gray-100 cursor-pointer flex justify-between"
                  >
                    <span className="font-mono">{s.symbol}</span>
                    <span className="text-gray-600">{s.name}</span>
                    <span className="text-gray-400">{s.exch}</span>
                  </div>
                ))}
              </div>
            )}
          </div>
          <button
            onClick={handleAnalyze}
            disabled={loading || !ticker}
            className="px-4 py-2 rounded bg-blue-600 text-white disabled:opacity-50 shadow hover:bg-blue-700"
          >
            {loading ? "Analyse..." : "Analyser"}
          </button>
        </div>

        {err && (
          <div className="mt-4 p-3 rounded bg-red-50 text-red-700 flex items-center">
            <AlertCircle className="w-5 h-5 mr-2" />
            {err}
          </div>
        )}

        {loading && (
          <div className="mt-6 text-gray-700 flex items-center gap-2">
            <Activity className="w-5 h-5 animate-spin" />
            Analyse en cours...
          </div>
        )}

        {result && !err && (
          <div className="mt-6 grid gap-6 lg:grid-cols-3">
            {/* Colonne gauche */}
            <div className="lg:col-span-1 space-y-4">
              <div className="p-4 rounded-lg border bg-white shadow-sm">
                <div className="flex items-center justify-between">
                  <div>
                    <h3 className="text-xl font-semibold">{result.name}</h3>
                    <p className="text-sm text-gray-600">
                      {result.ticker} • {result.currency || currency}
                    </p>
                  </div>
                  <div
                    className={`px-3 py-2 rounded text-white flex items-center gap-2 ${getRecommendationColor(
                      result.ai_analysis?.action
                    )}`}
                    title={result.ai_analysis?.recommendation}
                  >
                    {getRecommendationIcon(result.ai_analysis?.action)}
                    <span className="font-semibold">{result.ai_analysis?.recommendation || "—"}</span>
                  </div>
                </div>

                <div className="mt-4">
                  <div className="text-3xl font-bold">
                    {formatNum(result.current_price)} {result.currency || currency}
                  </div>
                  <div className={`text-sm ${Number(result.change) >= 0 ? "text-green-600" : "text-red-600"}`}>
                    {Number(result.change) >= 0 ? "+" : ""}
                    {formatNum(result.change)} ({formatPct(result.change_pct)}) depuis la veille
                  </div>
                  <div className="text-xs text-gray-500">
                    Bench: {result.benchmark} • Points de données: {result.data_points}
                  </div>
                </div>

                <div className="mt-4 grid grid-cols-2 gap-2">
                  <Metric label="Sharpe" value={formatNum(result.technical_indicators?.sharpe)} />
                  <Metric label="Sortino" value={formatNum(result.technical_indicators?.sortino)} />
                  <Metric label="Vol annualisée" value={formatPct(result.technical_indicators?.volatility_annual)} />
                  <Metric label="Max Drawdown" value={formatPct(result.technical_indicators?.max_drawdown)} />
                  <Metric label="Beta" value={formatNum(result.technical_indicators?.beta)} />
                </div>

                <div className="mt-4">
                  <ScoreBar label="RSI" score={result.ai_analysis?.detailed_scores?.rsi ?? 0} />
                  <ScoreBar label="MACD" score={result.ai_analysis?.detailed_scores?.macd ?? 0} />
                  <ScoreBar label="Tendance" score={result.ai_analysis?.detailed_scores?.trend ?? 0} />
                  <ScoreBar label="Momentum" score={result.ai_analysis?.detailed_scores?.momentum ?? 0} />
                  <ScoreBar label="Risque" score={result.ai_analysis?.detailed_scores?.risk ?? 0} />
                </div>
              </div>

              {Array.isArray(result.ai_analysis?.analysis_summary) && result.ai_analysis.analysis_summary.length > 0 && (
                <div className="p-4 rounded-lg border bg-white shadow-sm">
                  <div className="font-semibold mb-2 flex items-center gap-2">
                    <LineChartIcon className="w-4 h-4" /> Résumé d'analyse
                  </div>
                  <ul className="list-disc pl-5 space-y-1 text-sm text-gray-700">
                    {result.ai_analysis.analysis_summary.map((l, i) => (
                      <li key={i}>{l}</li>
                    ))}
                  </ul>
                </div>
              )}
            </div>

            {/* Colonne droite */}
            <div className="lg:col-span-2 space-y-4">
              <div className="p-4 rounded-lg border bg-white shadow-sm">
                <div className="flex items-center justify-between mb-2">
                  <div className="font-semibold flex items-center gap-2">
                    <BarChart3 className="w-4 h-4" />
                    Historique & Prévisions (bandes de confiance)
                  </div>
                  <div className="flex items-center gap-2 text-sm">
                    <Toggle on={showHistory} onClick={() => setShowHistory((v) => !v)} label="Historique" />
                    <Toggle on={showForecast} onClick={() => setShowForecast((v) => !v)} label="Prévision" />
                    <Toggle on={showBands} onClick={() => setShowBands((v) => !v)} label="Bandes" />
                  </div>
                </div>

                <div style={{ width: "100%", height: 460 }}>
                  <ResponsiveContainer>
                    <ComposedChart data={ChartData} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
                      <defs>
                        <linearGradient id="bandFill" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="0%" stopColor="#93c5fd" stopOpacity={0.35} />
                          <stop offset="100%" stopColor="#93c5fd" stopOpacity={0.05} />
                        </linearGradient>
                      </defs>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis
                        type="number"
                        dataKey="ts"
                        scale="time"
                        domain={xDomain as any}
                        tickFormatter={(v) => new Date(v).toLocaleDateString("fr-FR")}
                        minTickGap={30}
                        allowDataOverflow
                      />
                      <YAxis />
                      <Tooltip
                        labelFormatter={(v) => new Date(v).toLocaleDateString("fr-FR")}
                        formatter={(value: any, name: any) => [formatNum(Number(value)), name]}
                      />
                      <Legend />
                      <ReferenceLine
                        x={todayTs}
                        stroke="#9ca3af"
                        strokeDasharray="4 2"
                        label={{ value: "Aujourd'hui", position: "top", fill: "#6b7280" } as any}
                      />

                      {showBands && (
                        <>
                          <Area
                            type="monotone"
                            dataKey="forecast_p95"
                            stroke="none"
                            fill="url(#bandFill)"
                            name="Prévision P95"
                            isAnimationActive={false}
                          />
                          <Area
                            type="monotone"
                            dataKey="forecast_p5"
                            stroke="none"
                            fill="#ffffff"
                            fillOpacity={1}
                            name="Prévision P5"
                            isAnimationActive={false}
                          />
                        </>
                      )}

                      {showHistory && (
                        <Line type="monotone" dataKey="close" stroke="#111827" dot={false} name="Historique" />
                      )}

                      {showForecast && (
                        <Line
                          type="monotone"
                          dataKey="forecast_p50"
                          stroke="#2563eb"
                          strokeWidth={2}
                          strokeDasharray="4 4"
                          name="Prévision médiane"
                          dot={{ r: 2 }}
                        />
                      )}

                      <Brush dataKey="ts" height={24} travellerWidth={12} />
                    </ComposedChart>
                  </ResponsiveContainer>
                </div>
              </div>

              <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                <MetricCard label="Objectif haussier" color="green" value={result.ai_analysis?.price_targets?.upside} currency={result.currency || currency} />
                <MetricCard label="Objectif baissier" color="red" value={result.ai_analysis?.price_targets?.downside} currency={result.currency || currency} />
                <MetricCard label="Support" color="gray" value={result.ai_analysis?.key_levels?.support} currency={result.currency || currency} />
                <MetricCard label="Résistance" color="gray" value={result.ai_analysis?.key_levels?.resistance} currency={result.currency || currency} />
                <MetricCard label="Tendance" color="blue" value={result.forecast?.trend} />
                <MetricCard label="Score IA" color="purple" value={result.ai_analysis?.score} />
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

function Toggle({ on, onClick, label }: { on: boolean; onClick: () => void; label: string }) {
  return (
    <button onClick={onClick} className="px-2 py-1 rounded border hover:bg-gray-50 flex items-center gap-1">
      {on ? <Eye className="w-4 h-4" /> : <EyeOff className="w-4 h-4" />}
      {label}
    </button>
  );
}

function ScoreBar({ score = 0, label = "" }: { score?: number; label?: string }) {
  return (
    <div className="mb-2">
      <div className="flex justify-between text-sm mb-1">
        <span className="font-medium">{label}</span>
        <span>{Math.round(score)}/100</span>
      </div>
      <div className="w-full bg-gray-200 rounded-full h-2">
        <div
          className={`h-2 rounded-full transition-all duration-500 ${
            score >= 70 ? "bg-green-500" : score >= 50 ? "bg-yellow-500" : "bg-red-500"
          }`}
          style={{ width: `${Math.max(0, Math.min(100, score))}%` }}
        ></div>
      </div>
    </div>
  );
}

function Metric({ label, value }: { label: string; value?: string }) {
  return (
    <div className="p-3 rounded bg-gray-50">
      <div className="text-xs text-gray-600">{label}</div>
      <div className="text-lg font-semibold">{value ?? "—"}</div>
    </div>
  );
}

function MetricCard({
  label,
  value,
  currency,
  color = "gray",
}: {
  label: string;
  value?: number | string;
  currency?: string;
  color?: "green" | "red" | "blue" | "gray" | "purple";
}) {
  const colorMap: Record<string, string> = {
    green: "bg-green-50 text-green-700",
    red: "bg-red-50 text-red-700",
    blue: "bg-blue-50 text-blue-700",
    gray: "bg-gray-50 text-gray-700",
    purple: "bg-purple-50 text-purple-700",
  };
  return (
    <div className={`p-3 rounded ${colorMap[color]}`}>
      <div className="text-xs opacity-80">{label}</div>
      <div className="text-lg font-semibold">
        {value === undefined || value === null ? "—" : value} {currency || ""}
      </div>
    </div>
  );
}

export default App;
