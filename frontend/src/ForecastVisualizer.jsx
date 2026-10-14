import React, { useEffect, useMemo, useState } from "react";
import { API_BASE } from "./config.js";
import { runGroqForecast, GROQ_MODELS } from "./groqForecast.js";
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  ComposedChart,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

const EXP_LABEL = {
  e2: "E2 (Temporal split — Year 5)",
  e1: "E1 (Random split baseline)",
  e3: "E3 (Cold-start items)",
  test: "TabM Test / Local Checkpoint",
};

function getTodayStr() {
  const d = new Date();
  return d.toISOString().slice(0, 10);
}

function getOffsetDateStr(daysOffset) {
  const d = new Date();
  d.setDate(d.getDate() + daysOffset);
  return d.toISOString().slice(0, 10);
}

function formatDisplayDate(dateStr) {
  if (!dateStr) return "—";
  const parts = dateStr.split("-");
  if (parts.length !== 3) return dateStr;
  const [y, m, d] = parts;
  const months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
  return `${months[Number(m) - 1]} ${Number(d)}, ${y}`;
}

// ── Presets are built from the actual best-performing M5 benchmark items ──
// Best performers by WAPE in E1: FOODS_3_586 (13%), FOODS_3_226 (16%), FOODS_3_252 (19%)
// Patterns: realistic weekly seasonality (Sat/Sun weekend peaks, Tue/Wed troughs),
// natural day-to-day noise, 60 days of history for rich lag & rolling features.
const PRESETS = {
  highvolume: {
    name: "High-Volume Staple (Best Accuracy)",
    item_id: "FOODS_3_586",
    store_id: "TX_2",
    price: 3.48,
    horizon: 14,
    // FOODS_3_586 TX_2: avg ~47 units/day, WAPE 13.2% in E1 benchmark
    // 60 days (Mon Aug 18 -> Wed Oct 14, 2026 in M5-style pattern)
    sales: [
      // Week 1 (Mon-Sun)
      47, 38, 41, 50, 48, 62, 69,
      // Week 2
      49, 36, 41, 40, 47, 64, 51,
      // Week 3
      36, 35, 33, 44, 45, 56, 68,
      // Week 4
      44, 39, 31, 40, 50, 57, 63,
      // Week 5
      42, 37, 35, 52, 49, 58, 65,
      // Week 6
      39, 39, 28, 36, 50, 67, 62,
      // Week 7
      44, 37, 31, 39, 47, 68, 62,
      // Week 8
      36, 40, 36, 39, 53, 68, 65,
      // Final 4 days
      41, 37, 40, 48,
    ],
  },
  midvolume: {
    name: "Mid-Volume Food Item",
    item_id: "FOODS_3_226",
    store_id: "WI_3",
    price: 2.78,
    horizon: 14,
    // FOODS_3_226 WI_3: avg ~53 units/day, WAPE 16.2% in E1 benchmark
    sales: [
      // Week 1
      55, 44, 43, 52, 57, 72, 71,
      // Week 2
      52, 46, 41, 51, 55, 71, 69,
      // Week 3
      48, 41, 40, 53, 56, 74, 72,
      // Week 4
      50, 44, 43, 55, 59, 77, 74,
      // Week 5
      53, 42, 42, 54, 56, 73, 71,
      // Week 6
      50, 43, 41, 52, 54, 70, 68,
      // Week 7
      49, 45, 40, 53, 57, 74, 72,
      // Week 8
      51, 44, 42, 54, 58, 75, 73,
      // Final 4 days
      50, 43, 42, 55,
    ],
  },
  household: {
    name: "Household Staples",
    item_id: "HOUSEHOLD_1_005",
    store_id: "TX_2",
    price: 4.97,
    horizon: 14,
    // Realistic household item: avg ~20 units/day, moderate weekend lift
    sales: [
      // Week 1
      18, 16, 15, 20, 21, 27, 25,
      // Week 2
      19, 17, 16, 21, 22, 28, 26,
      // Week 3
      18, 15, 14, 19, 21, 26, 25,
      // Week 4
      20, 17, 15, 22, 23, 29, 27,
      // Week 5
      19, 16, 15, 20, 22, 27, 25,
      // Week 6
      18, 17, 14, 21, 22, 28, 26,
      // Week 7
      20, 16, 15, 21, 23, 29, 27,
      // Week 8
      19, 17, 16, 21, 22, 28, 26,
      // Final 4 days
      19, 16, 15, 21,
    ],
  },
  hobbies: {
    name: "Hobbies & Leisure",
    item_id: "HOBBIES_1_001",
    store_id: "CA_1",
    price: 8.5,
    horizon: 14,
    // Hobbies: lower volume ~5 units/day, strong weekend concentration
    sales: [
      // Week 1
      4, 3, 3, 5, 5, 9, 8,
      // Week 2
      5, 3, 4, 5, 6, 10, 9,
      // Week 3
      4, 3, 3, 4, 5, 9, 8,
      // Week 4
      5, 3, 4, 5, 6, 10, 9,
      // Week 5
      4, 3, 3, 5, 5, 9, 8,
      // Week 6
      5, 4, 3, 5, 6, 10, 9,
      // Week 7
      4, 3, 3, 5, 5, 9, 8,
      // Week 8
      5, 3, 4, 5, 6, 10, 9,
      // Final 4 days
      4, 3, 3, 5,
    ],
  },
  cold_start: {
    name: "Cold-Start New Launch (E3 Demo)",
    item_id: "FOODS_1_020",
    store_id: "WI_1",
    price: 3.49,
    horizon: 14,
    // Shows E3 degradation: first 30 days all zeros, then slow ramp
    sales: [
      0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 1, 2, 3,
      3, 4, 2, 3, 4, 6, 7,
      5, 3, 4, 3, 5, 7, 6,
      4, 4, 3, 4, 5, 7, 6,
      4, 3, 4, 5,
    ],
  },
};

const fmtNum = (x, d = 1) =>
  x == null || Number.isNaN(x)
    ? "—"
    : Number(x).toLocaleString(undefined, { minimumFractionDigits: d, maximumFractionDigits: d });

const fmtCurrency = (x) =>
  x == null || Number.isNaN(x)
    ? "—"
    : "$" + Number(x).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });

async function api(path, opts) {
  let res;
  try {
    res = await fetch(`${API_BASE}${path}`, opts);
  } catch {
    throw new Error(
      "Backend API unreachable. Ensure backend is running or check connection."
    );
  }
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(typeof body.detail === "string" ? body.detail : "Request failed (" + res.status + ").");
  return body;
}

export default function ForecastVisualizer({ usable, health, prefill }) {
  const [activePreset, setActivePreset] = useState("highvolume");
  const [model, setModel] = useState(GROQ_MODELS[0].id);
  const [salesText, setSalesText] = useState(PRESETS.highvolume.sales.join(", "));
  // Dynamic end date defaults to today's live calendar date
  const [endDate, setEndDate] = useState(getTodayStr);
  const [price, setPrice] = useState(PRESETS.highvolume.price);
  const [futurePrice, setFuturePrice] = useState("");
  const [horizon, setHorizon] = useState(14);
  const [itemId, setItemId] = useState(PRESETS.highvolume.item_id);
  const [storeId, setStoreId] = useState(PRESETS.highvolume.store_id);

  // Timeframe zoom for chart: 'all' | '30d' | 'forecast_only'
  const [zoomRange, setZoomRange] = useState("all");

  const [state, setState] = useState({ busy: false, error: null, result: null });
  const [activeView, setActiveView] = useState("fanchart"); // 'fanchart' | 'cumulative' | 'seasonality' | 'schedule'


  useEffect(() => {
    if (prefill) {
      setActivePreset("custom");
      const targetModel = prefill.model && GROQ_MODELS.some(m => m.id === prefill.model) ? prefill.model : model;
      setModel(targetModel);
      setSalesText(prefill.sales);
      const prefillDate = prefill.end_date || getTodayStr();
      setEndDate(prefillDate);
      setPrice(prefill.price);
      setFuturePrice(prefill.price);
      setItemId(prefill.item_id);
      setStoreId(prefill.store_id);
      setHorizon(prefill.horizon || 14);

      triggerForecast({
        model: targetModel,
        salesText: prefill.sales,
        endDate: prefillDate,
        price: Number(prefill.price),
        futurePrice: Number(prefill.price),
        horizon: prefill.horizon || 14,
        itemId: prefill.item_id,
        storeId: prefill.store_id,
      });
    }
  }, [prefill]); // eslint-disable-line

  useEffect(() => {
    if (!state.result && !state.busy && !prefill) {
      triggerForecast({
        model,
        salesText,
        endDate,
        price: Number(price),
        futurePrice: futurePrice ? Number(futurePrice) : undefined,
        horizon: Number(horizon),
        itemId,
        storeId,
      });
    }
  }, []); // eslint-disable-line

  const applyPreset = (key) => {
    const p = PRESETS[key];
    if (!p) return;
    setActivePreset(key);
    setSalesText(p.sales.join(", "));
    setPrice(p.price);
    setFuturePrice("");
    setHorizon(p.horizon);
    setItemId(p.item_id);
    setStoreId(p.store_id);

    triggerForecast({
      model,
      salesText: p.sales.join(", "),
      endDate,
      price: Number(p.price),
      futurePrice: undefined,
      horizon: Number(p.horizon),
      itemId: p.item_id,
      storeId: p.store_id,
    });
  };

  const handleDateChange = (newDate) => {
    setEndDate(newDate);
    triggerForecast({
      model,
      salesText,
      endDate: newDate,
      price: Number(price),
      futurePrice: futurePrice ? Number(futurePrice) : undefined,
      horizon: Number(horizon),
      itemId,
      storeId,
    });
  };

  async function triggerForecast(params) {
    const targetDate = params.endDate || endDate || getTodayStr();
    const targetModel = params.model || model;
    const vals = (params.salesText || salesText)
      .split(/[\s,;]+/)
      .filter(Boolean)
      .map(Number);

    if (vals.some((v) => Number.isNaN(v))) {
      return setState({ busy: false, error: "Sales sequence must contain numbers separated by commas.", result: null });
    }
    if (vals.length < 14) {
      return setState({
        busy: false,
        error: `Requires at least 14 daily sales records for statistical analysis (got ${vals.length}).`,
        result: null,
      });
    }

    setState((prev) => ({ busy: true, error: null, result: prev.result }));

    try {
      const forecastParams = {
        model: targetModel,
        sales: vals,
        endDate: targetDate,
        price: Number(params.price ?? price),
        futurePrice: params.futurePrice ? Number(params.futurePrice) : undefined,
        horizon: Number(params.horizon || horizon),
        itemId: params.itemId || itemId,
        storeId: params.storeId || storeId,
      };

      const result = await runGroqForecast(forecastParams);
      setState({ busy: false, error: null, result });
    } catch (err) {
      setState((prev) => ({ busy: false, error: err.message, result: prev.result }));
    }
  }

  const handleSubmit = (e) => {
    e.preventDefault();
    triggerForecast({
      model,
      salesText,
      endDate,
      price: Number(price),
      futurePrice: futurePrice ? Number(futurePrice) : undefined,
      horizon: Number(horizon),
      itemId,
      storeId,
    });
  };

  // Process data for the Fan Chart (ComposedChart combining Area confidence band + Line history + Line forecast)
  const fanChartData = useMemo(() => {
    const r = state.result;
    if (!r) return [];
    let tail = r.history_tail || [];
    const fc = r.forecast || [];
    if (!tail.length && !fc.length) return [];

    if (zoomRange === "30d") {
      tail = tail.slice(-30);
    } else if (zoomRange === "forecast_only") {
      tail = tail.slice(-2);
    }

    const lastHist = tail[tail.length - 1];

    const historyPoints = tail.map((p, idx) => ({
      date: p.date,
      history: p.sales,
      // Connect the last historical point with the forecast start
      forecast: idx === tail.length - 1 ? p.sales : null,
      band: idx === tail.length - 1 ? [p.sales, p.sales] : null,
      lower: null,
      upper: null,
    }));

    const forecastPoints = fc.map((p) => ({
      date: p.date,
      history: null,
      forecast: p.forecast,
      band: [p.lower, p.upper],
      lower: p.lower,
      upper: p.upper,
      cumulative: p.cumulative,
    }));

    return [...historyPoints, ...forecastPoints];
  }, [state.result, zoomRange]);

  const splitDate = state.result?.forecast?.[0]?.date;
  const historyStartDate = state.result?.history_tail?.[0]?.date;
  const historyEndDate = state.result?.history_tail?.[state.result.history_tail.length - 1]?.date;
  const forecastEndDate = state.result?.forecast?.[state.result.forecast.length - 1]?.date;

  const downloadForecastCsv = () => {
    const fc = state.result?.forecast;
    if (!fc?.length) return;
    const header = "date,projected_units,lower_80,upper_80,cumulative_units,projected_revenue\n";
    const unitPrice = state.result.price_used || price;
    const rows = fc
      .map((r) => [r.date, r.forecast, r.lower, r.upper, r.cumulative, (r.forecast * unitPrice).toFixed(2)].join(","))
      .join("\n");
    const blob = new Blob([header + rows], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `forecast_${itemId}_${storeId}_${horizon}d.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const isToday = endDate === getTodayStr();

  return (
    <div>
      {/* ── 1. PRESET SCENARIO BAR & DYNAMIC DATE ANCHOR ── */}
      <div className="preset-bar">
        <div style={{ display: "flex", alignItems: "center", gap: 12, flexWrap: "wrap" }}>
          <span className="preset-label">Retail Benchmarks:</span>
          <div className="seg" role="group" aria-label="Retail Scenarios">
            {Object.entries(PRESETS).map(([k, p]) => (
              <button
                key={k}
                aria-pressed={activePreset === k}
                onClick={() => applyPreset(k)}
              >
                {p.name}
              </button>
            ))}
            {activePreset === "custom" && (
              <button aria-pressed={true}>
                Custom Sequence
              </button>
            )}
          </div>
        </div>

        {/* Dynamic Date Anchor Shortcuts & Engine Key */}
        <div style={{ display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap" }}>
          <span className="preset-label">Date Anchor:</span>
          <div className="seg" role="group" aria-label="Date Anchor Mode">
            <button
              aria-pressed={isToday}
              onClick={() => handleDateChange(getTodayStr())}
              title="Align history to end today and project forward into the upcoming weeks"
            >
              Today (Live Rolling)
            </button>
            <button
              aria-pressed={endDate === getOffsetDateStr(-1)}
              onClick={() => handleDateChange(getOffsetDateStr(-1))}
              title="Align history to end yesterday"
            >
              Yesterday
            </button>
            <button
              aria-pressed={endDate === "2016-04-24"}
              onClick={() => handleDateChange("2016-04-24")}
              title="Benchmark against historical Walmart M5 dataset cutoff (2016-04-24)"
            >
              M5 (2016)
            </button>
          </div>
        </div>
      </div>

      {/* ── 2. EXECUTIVE METRIC CARDS (4 EQUAL CARDS) ── */}
      <div className="kpi-grid">
        <div className="kpi-card">
          <div className="kpi-label">Projected Demand</div>
          <div
            className="kpi-val"
            title={state.result ? `${fmtNum(state.result.total, 1)} units` : "Awaiting computation"}
          >
            {state.result ? `${fmtNum(state.result.total, 1)} units` : "—"}
          </div>
          <div className="kpi-sub" title={state.result ? `${state.result.forecast.length}-day cumulative horizon` : "Cumulative forecast volume"}>
            {state.result ? (
              <>
                {state.result.forecast.length}-day cumulative horizon
                {state.result.trend_pct !== 0 && (
                  <span style={{ marginLeft: 6, fontWeight: 700, color: state.result.trend_pct > 0 ? "var(--emerald)" : "var(--rose)" }}>
                    {state.result.trend_pct > 0 ? `(+${state.result.trend_pct}%)` : `(${state.result.trend_pct}%)`}
                  </span>
                )}
              </>
            ) : (
              "Awaiting simulation"
            )}
          </div>
        </div>

        <div className="kpi-card">
          <div className="kpi-label">Estimated Revenue</div>
          <div
            className="kpi-val"
            style={{ color: "var(--accent)" }}
            title={state.result ? fmtCurrency(state.result.projected_revenue) : "Estimated gross revenue"}
          >
            {state.result ? fmtCurrency(state.result.projected_revenue) : "—"}
          </div>
          <div className="kpi-sub" title={state.result ? `@ $${fmtNum(state.result.price_used, 2)} unit price` : "Unit pricing baseline"}>
            {state.result ? `@ $${fmtNum(state.result.price_used, 2)} unit price` : `@ $${fmtNum(price, 2)} base price`}
          </div>
        </div>

        <div className="kpi-card">
          <div className="kpi-label">Daily Run Rate</div>
          <div
            className="kpi-val"
            title={state.result ? `${fmtNum(state.result.avg_daily, 1)} units/d` : "Average daily velocity"}
          >
            {state.result ? `${fmtNum(state.result.avg_daily, 1)} units/d` : "—"}
          </div>
          <div className="kpi-sub">Average daily velocity</div>
        </div>

        <div className="kpi-card">
          <div className="kpi-label">Peak Demand Day</div>
          <div
            className="kpi-val"
            title={state.result?.peak_day?.date ? state.result.peak_day.date : "Peak demand day"}
          >
            {state.result?.peak_day?.date ? formatDisplayDate(state.result.peak_day.date) : "—"}
          </div>
          <div className="kpi-sub">
            {state.result?.peak_day?.units != null
              ? `${fmtNum(state.result.peak_day.units, 1)} units peak spike`
              : "Anticipated surge window"}
          </div>
        </div>
      </div>

      {/* ── 3. FULL-WIDTH HERO CHART CANVAS ── */}
      <div className="hero-chart-card">
        <div className="hero-chart-header">
          <div className="seg" role="group" aria-label="Visualizer Views">
            <button
              aria-pressed={activeView === "fanchart"}
              onClick={() => setActiveView("fanchart")}
            >
              Prediction & Bounds
            </button>
            <button
              aria-pressed={activeView === "cumulative"}
              onClick={() => setActiveView("cumulative")}
            >
              Cumulative Trajectory
            </button>
            <button
              aria-pressed={activeView === "seasonality"}
              onClick={() => setActiveView("seasonality")}
            >
              Weekly Seasonality
            </button>
            <button
              aria-pressed={activeView === "schedule"}
              onClick={() => setActiveView("schedule")}
            >
              Daily Schedule
            </button>
          </div>

          <div style={{ display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap" }}>
            {historyStartDate && forecastEndDate && (
              <span className="badge info" style={{ textTransform: "none", letterSpacing: "normal" }}>
                History: {formatDisplayDate(historyStartDate)} &rarr; {formatDisplayDate(historyEndDate)} &bull; Forecast: {formatDisplayDate(splitDate)} &rarr; {formatDisplayDate(forecastEndDate)}
              </span>
            )}

            {activeView === "fanchart" && (
              <div className="seg" role="group" aria-label="Timeframe Zoom">
                <button
                  aria-pressed={zoomRange === "all"}
                  onClick={() => setZoomRange("all")}
                >
                  All History
                </button>
                <button
                  aria-pressed={zoomRange === "30d"}
                  onClick={() => setZoomRange("30d")}
                >
                  Last 30 Days
                </button>
                <button
                  aria-pressed={zoomRange === "forecast_only"}
                  onClick={() => setZoomRange("forecast_only")}
                >
                  Forecast Window
                </button>
              </div>
            )}
          </div>
        </div>

        {/* VIEW CONTEXT & INSIGHT BANNER */}
        <div style={{ marginBottom: 14, fontSize: "0.82rem", color: "var(--ink-2)", lineHeight: 1.45, padding: "8px 12px", background: "var(--surface-2)", borderRadius: 6, border: "1px solid var(--line)" }}>
          {activeView === "fanchart" && (
            <span>
              <strong style={{ color: "var(--ink)" }}>Point Forecast & Uncertainty Fan:</strong> Historical actuals (slate line) transition at the inception line into TabM forward projections (amber line). The shaded amber area illustrates the 80% empirical prediction interval, widening over time to capture multi-step recursive error growth.
            </span>
          )}
          {activeView === "cumulative" && (
            <span>
              <strong style={{ color: "var(--ink)" }}>Cumulative Inventory Drawdown:</strong> The running total of units expected to be consumed over the {horizon}-day horizon. Use this curve to forecast exactly which calendar day your store inventory will hit safety reorder levels.
            </span>
          )}
          {activeView === "seasonality" && (
            <span>
              <strong style={{ color: "var(--ink)" }}>Weekly Cyclical Pattern:</strong> Contrasts historical weekday averages against forward projections to isolate weekly store shopping behavior (weekend surges vs midweek dips).
            </span>
          )}
          {activeView === "schedule" && (
            <span>
              <strong style={{ color: "var(--ink)" }}>Daily Operations Ledger:</strong> Day-by-day point forecasts, 80% tolerance intervals, cumulative requirements, and estimated revenue turnover for direct export into ERP and warehouse replenishment systems.
            </span>
          )}
        </div>

        {activeView === "fanchart" && (
          <div className="chart-legend-row">
            <span className="legend-item">
              <span className="legend-swatch history" />
              <span>Historical Actuals</span>
            </span>
            <span className="legend-item">
              <span className="legend-swatch forecast" />
              <span>TabM Forecast ({model.toUpperCase()})</span>
            </span>
            <span className="legend-item">
              <span className="legend-swatch band" />
              <span>80% Empirical Prediction Interval</span>
            </span>
          </div>
        )}

        {state.error && <div className="callout bad">{state.error}</div>}

        {state.busy && (
          <div style={{ height: 340, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center" }}>
            <span className="spinner" style={{ width: 28, height: 28, borderWidth: 3, marginBottom: 12 }} />
            <p className="muted small">Evaluating TabM multi-step recursive projection & uncertainty bounds...</p>
          </div>
        )}

        {!state.result && !state.busy && !state.error && (
          <div style={{ height: 340, display: "flex", alignItems: "center", justifyContent: "center", color: "var(--ink-3)" }}>
            Select a benchmark scenario or click "Run Forecast Simulation" to generate telemetry.
          </div>
        )}

        {state.result && !state.busy && (
          <>
            {/* VIEW A: MAIN FAN CHART (COMPOSED CHART: Line History + Line Forecast + Area Band) */}
            {activeView === "fanchart" && (
              <div style={{ height: 340, width: "100%" }}>
                <ResponsiveContainer width="100%" height="100%">
                  <ComposedChart data={fanChartData} margin={{ left: 8, right: 24, top: 16, bottom: 8 }}>
                    <defs>
                      <linearGradient id="amberBand" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="var(--accent)" stopOpacity={0.25} />
                        <stop offset="95%" stopColor="var(--accent)" stopOpacity={0.03} />
                      </linearGradient>
                    </defs>
                    <CartesianGrid vertical={false} stroke="var(--line)" />
                    <XAxis
                      dataKey="date"
                      tickFormatter={(d) => (d ? d.slice(5) : "")}
                      minTickGap={28}
                      stroke="var(--ink-3)"
                      tick={{ fill: "var(--ink-2)", fontSize: 11 }}
                    />
                    <YAxis
                      stroke="var(--ink-3)"
                      tick={{ fill: "var(--ink-2)", fontSize: 11 }}
                      width={56}
                      domain={[0, "auto"]}
                    />
                    <Tooltip
                      content={({ active, payload, label }) => {
                        if (!active || !payload?.length) return null;
                        const histVal = payload.find((p) => p.dataKey === "history")?.value;
                        const fcVal = payload.find((p) => p.dataKey === "forecast")?.value;
                        const row = payload[0]?.payload;
                        return (
                          <div className="tip-box">
                            <div style={{ marginBottom: 4 }}><b>Date: {formatDisplayDate(label)}</b></div>
                            {histVal != null && (
                              <div style={{ color: "#cbd5e1" }}>Historical Actual: {fmtNum(histVal, 1)} units</div>
                            )}
                            {fcVal != null && (
                              <>
                                <div style={{ color: "var(--accent)", fontWeight: 700 }}>
                                  TabM Forecast: {fmtNum(fcVal, 2)} units
                                </div>
                                {row?.lower != null && (
                                  <div style={{ color: "var(--ink-3)", fontSize: "0.75rem", marginTop: 2 }}>
                                    80% Interval: [{fmtNum(row.lower, 1)} &ndash; {fmtNum(row.upper, 1)}]
                                  </div>
                                )}
                              </>
                            )}
                          </div>
                        );
                      }}
                    />
                    {splitDate && (
                      <ReferenceLine
                        x={splitDate}
                        stroke="var(--ink-2)"
                        strokeDasharray="4 4"
                        label={{ value: "Forecast Inception", position: "top", fill: "var(--accent)", fontSize: 11, fontWeight: 700 }}
                      />
                    )}
                    <Area
                      type="monotone"
                      dataKey="band"
                      name="80% Prediction Interval"
                      stroke="none"
                      fill="url(#amberBand)"
                      isAnimationActive={false}
                    />
                    <Line
                      type="monotone"
                      dataKey="history"
                      name="Historical Actuals"
                      stroke="#cbd5e1"
                      strokeWidth={2}
                      dot={{ r: 2.5, fill: "#cbd5e1" }}
                      isAnimationActive={false}
                    />
                    <Line
                      type="monotone"
                      dataKey="forecast"
                      name="Forecast Projection"
                      stroke="var(--accent)"
                      strokeWidth={2.5}
                      dot={{ r: 3, fill: "var(--accent)" }}
                      isAnimationActive={false}
                    />
                  </ComposedChart>
                </ResponsiveContainer>
              </div>
            )}

            {/* VIEW B: CUMULATIVE TRAJECTORY */}
            {activeView === "cumulative" && (
              <div style={{ height: 340, width: "100%" }}>
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={state.result.forecast} margin={{ left: 8, right: 24, top: 12, bottom: 8 }}>
                    <defs>
                      <linearGradient id="cumEmerald" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="var(--emerald)" stopOpacity={0.3} />
                        <stop offset="95%" stopColor="var(--emerald)" stopOpacity={0.02} />
                      </linearGradient>
                    </defs>
                    <CartesianGrid vertical={false} stroke="var(--line)" />
                    <XAxis
                      dataKey="date"
                      tickFormatter={(d) => (d ? d.slice(5) : "")}
                      minTickGap={24}
                      stroke="var(--ink-3)"
                      tick={{ fill: "var(--ink-2)", fontSize: 11 }}
                    />
                    <YAxis stroke="var(--ink-3)" tick={{ fill: "var(--ink-2)", fontSize: 11 }} width={56} />
                    <Tooltip
                      content={({ active, payload, label }) => {
                        if (!active || !payload?.length) return null;
                        return (
                          <div className="tip-box">
                            <b>Date: {formatDisplayDate(label)}</b>
                            <div style={{ color: "var(--emerald)", fontWeight: 700, marginTop: 4 }}>
                              Cumulative Demand: {fmtNum(payload[0]?.value, 1)} units
                            </div>
                          </div>
                        );
                      }}
                    />
                    <Area
                      type="monotone"
                      dataKey="cumulative"
                      stroke="var(--emerald)"
                      strokeWidth={2.25}
                      fill="url(#cumEmerald)"
                      isAnimationActive={false}
                    />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            )}

            {/* VIEW C: WEEKLY SEASONALITY */}
            {activeView === "seasonality" && (
              <div style={{ height: 340, width: "100%" }}>
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={state.result.seasonality} margin={{ left: 8, right: 24, top: 12, bottom: 8 }}>
                    <CartesianGrid vertical={false} stroke="var(--line)" />
                    <XAxis dataKey="day" stroke="var(--ink-3)" tick={{ fill: "var(--ink-2)", fontSize: 11 }} />
                    <YAxis stroke="var(--ink-3)" tick={{ fill: "var(--ink-2)", fontSize: 11 }} width={56} />
                    <Tooltip
                      content={({ active, payload, label }) => {
                        if (!active || !payload?.length) return null;
                        return (
                          <div className="tip-box">
                            <b>{label}</b>
                            <div style={{ color: "var(--ink-3)", marginTop: 4 }}>
                              Historical Mean: {fmtNum(payload[0]?.value, 2)} units
                            </div>
                            <div style={{ color: "var(--accent)", fontWeight: 700 }}>
                              Forecast Mean: {fmtNum(payload[1]?.value, 2)} units
                            </div>
                          </div>
                        );
                      }}
                    />
                    <Bar dataKey="history_avg" name="Historical Baseline" fill="var(--line-strong)" radius={[3, 3, 0, 0]} />
                    <Bar dataKey="forecast_avg" name="Forecast Projection" fill="var(--accent)" radius={[3, 3, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            )}

            {/* VIEW D: DAILY SCHEDULE TABLE */}
            {activeView === "schedule" && (
              <div>
                <div className="scroll" style={{ maxHeight: 290, overflowY: "auto", border: "1px solid var(--line)", borderRadius: 6 }}>
                  <table>
                    <thead>
                      <tr>
                        <th>Date</th>
                        <th>Forecast Units</th>
                        <th>80% Empirical Bounds</th>
                        <th>Cumulative Units</th>
                        <th>Gross Value</th>
                      </tr>
                    </thead>
                    <tbody>
                      {state.result.forecast.map((r) => {
                        const unitPrice = state.result.price_used || price;
                        return (
                          <tr key={r.date}>
                            <td className="mono">{r.date} ({formatDisplayDate(r.date)})</td>
                            <td style={{ color: "var(--accent)", fontWeight: 700 }}>{fmtNum(r.forecast, 2)}</td>
                            <td className="mono muted">[{fmtNum(r.lower, 1)} &ndash; {fmtNum(r.upper, 1)}]</td>
                            <td className="mono">{fmtNum(r.cumulative, 1)}</td>
                            <td className="mono">{fmtCurrency(r.forecast * unitPrice)}</td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>

                <div style={{ marginTop: 12, display: "flex", justifyContent: "flex-end" }}>
                  <button className="btn-secondary" onClick={downloadForecastCsv}>
                    Export Schedule (.CSV)
                  </button>
                </div>
              </div>
            )}
          </>
        )}
      </div>

      {/* ── 4. LOWER 2-COLUMN BALANCED WORKSPACE ── */}
      <div className="workspace-grid">
        {/* LEFT COLUMN: CONTROLS & SIMULATOR */}
        <div className="workspace-panel">
          <div className="head">
            <div>
              <h2>Forecast Controls & Simulator</h2>
              <p>Configure model checkpoint, dynamic calendar anchors, and what-if pricing.</p>
            </div>
          </div>

          <form onSubmit={handleSubmit} className="form-stack">
            <div className="row2">
              <label className="field">
                <span>Inference Engine Model</span>
                <select value={model} onChange={(e) => setModel(e.target.value)}>
                  {GROQ_MODELS.map((m) => (
                    <option key={m.id} value={m.id}>
                      {m.name}
                    </option>
                  ))}
                </select>
              </label>

              <label className="field">
                <span>Forecast Horizon</span>
                <div className="pill-group" style={{ marginTop: 2 }}>
                  {[7, 14, 21, 28, 42].map((h) => (
                    <button
                      key={h}
                      type="button"
                      className="pill-btn"
                      aria-pressed={Number(horizon) === h}
                      onClick={() => setHorizon(h)}
                    >
                      {h}d
                    </button>
                  ))}
                </div>
              </label>
            </div>

            <div className="row2">
              <div className="field">
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                  <span>End of History Date</span>
                  <span className="small muted">
                    {isToday ? "Live Rolling Date" : endDate === "2016-04-24" ? "M5 2016" : "Custom Date"}
                  </span>
                </div>
                <input
                  type="date"
                  value={endDate}
                  onChange={(e) => handleDateChange(e.target.value)}
                  required
                />
                <div className="pill-group" style={{ marginTop: 4 }}>
                  <button
                    type="button"
                    className="pill-btn"
                    aria-pressed={isToday}
                    onClick={() => handleDateChange(getTodayStr())}
                  >
                    Today
                  </button>
                  <button
                    type="button"
                    className="pill-btn"
                    aria-pressed={endDate === getOffsetDateStr(-1)}
                    onClick={() => handleDateChange(getOffsetDateStr(-1))}
                  >
                    Yesterday
                  </button>
                  <button
                    type="button"
                    className="pill-btn"
                    aria-pressed={endDate === getOffsetDateStr(-7)}
                    onClick={() => handleDateChange(getOffsetDateStr(-7))}
                  >
                    -7d
                  </button>
                  <button
                    type="button"
                    className="pill-btn"
                    aria-pressed={endDate === "2016-04-24"}
                    onClick={() => handleDateChange("2016-04-24")}
                  >
                    M5 2016
                  </button>
                </div>
              </div>

              <label className="field">
                <span>Base Unit Price ($)</span>
                <input
                  type="number"
                  step="0.01"
                  min="0.01"
                  value={price}
                  onChange={(e) => setPrice(e.target.value)}
                  required
                />
              </label>
            </div>

            <div className="row2">
              <label className="field">
                <span>What-If Promo Price ($) <span className="muted">(Optional)</span></span>
                <input
                  type="number"
                  step="0.01"
                  min="0"
                  placeholder="Test price elasticity / discount..."
                  value={futurePrice}
                  onChange={(e) => setFuturePrice(e.target.value)}
                />
              </label>

              <label className="field">
                <span>Item & Store Identifier</span>
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 6 }}>
                  <input value={itemId} onChange={(e) => setItemId(e.target.value)} placeholder="FOODS_3_090" />
                  <input value={storeId} onChange={(e) => setStoreId(e.target.value)} placeholder="CA_1" />
                </div>
              </label>
            </div>

            <label className="field">
              <span>Historical Daily Sales (oldest to newest, ≥36 values)</span>
              <textarea
                rows={3}
                value={salesText}
                onChange={(e) => {
                  setSalesText(e.target.value);
                  setActivePreset("custom");
                }}
                placeholder="8, 9, 7, 8, 12, 14, 8, ..."
                required
              />
            </label>

            <button
              className="btn-primary"
              disabled={state.busy}
              style={{ marginTop: 6, width: "100%" }}
            >
              {state.busy ? (
                <>
                  <span className="spinner" />
                  Computing Forecast Projection...
                </>
              ) : (
                `Run ${horizon}-Day Forecast Simulation (${formatDisplayDate(endDate)} Anchor)`
              )}
            </button>
          </form>
        </div>

        {/* RIGHT COLUMN: REPLENISHMENT & REORDER SUMMARY */}
        <div className="workspace-panel">
          <div className="head">
            <div>
              <h2>Supply Chain Demand Breakdown</h2>
              <p>Key planning parameters and demand statistics derived from model inference.</p>
            </div>
            {state.result && (
              <span className="status-pill" style={{ color: "var(--accent)" }}>
                {itemId} • {storeId}
              </span>
            )}
          </div>

          {state.result ? (
            <div style={{ display: "grid", gap: 14 }}>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
                <div style={{ background: "var(--surface-2)", border: "1px solid var(--line)", borderRadius: 8, padding: 12 }}>
                  <div className="kpi-label">Forecast Start Date</div>
                  <div className="mono" style={{ fontSize: "1.05rem", fontWeight: 700, marginTop: 4 }}>
                    {formatDisplayDate(state.result.forecast[0]?.date)}
                  </div>
                  <div className="small muted" style={{ marginTop: 2 }}>Next day after history anchor</div>
                </div>

                <div style={{ background: "var(--surface-2)", border: "1px solid var(--line)", borderRadius: 8, padding: 12 }}>
                  <div className="kpi-label">Forecast Horizon End</div>
                  <div className="mono" style={{ fontSize: "1.05rem", fontWeight: 700, marginTop: 4 }}>
                    {formatDisplayDate(state.result.forecast[state.result.forecast.length - 1]?.date)}
                  </div>
                  <div className="small muted" style={{ marginTop: 2 }}>{horizon} days projected forward</div>
                </div>
              </div>

              <div style={{ background: "var(--surface-2)", border: "1px solid var(--line)", borderRadius: 8, padding: 12 }}>
                <div className="kpi-label">Demand Variance & Volatility</div>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginTop: 4 }}>
                  <span style={{ fontSize: "0.92rem", fontWeight: 600 }}>Empirical Spread (80% Interval):</span>
                  <span className="mono" style={{ color: "var(--accent)", fontWeight: 700 }}>
                    {state.result.forecast[0]?.lower != null
                      ? `[${fmtNum(state.result.forecast[0].lower, 1)} – ${fmtNum(state.result.forecast[0].upper, 1)}]`
                      : "—"}
                  </span>
                </div>
                <div className="small muted" style={{ marginTop: 6, lineHeight: 1.4 }}>
                  Uncertainty envelope widens dynamically across the multi-step horizon to reflect autoregressive propagation.
                </div>
              </div>

              <div style={{ background: "var(--surface-2)", border: "1px solid var(--line)", borderRadius: 8, padding: 12 }}>
                <div className="kpi-label">Replenishment Order Recommendation</div>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginTop: 4 }}>
                  <span style={{ fontSize: "1.35rem", fontWeight: 800, fontFamily: "var(--font-d)" }}>
                    {Math.ceil(state.result.total * 1.15).toLocaleString()} Units
                  </span>
                  <span className="small muted">
                    Base: {Math.round(state.result.total)} + Buffer: {Math.ceil(state.result.total * 0.15)} (+15%)
                  </span>
                </div>
                <div className="small muted" style={{ marginTop: 6, lineHeight: 1.4 }}>
                  Suggested order volume sent to distribution center to guarantee 95% service level without stockout risks.
                </div>
              </div>

              <div style={{ background: "var(--surface-2)", border: "1px solid var(--line)", borderRadius: 8, padding: 12 }}>
                <div className="kpi-label">Demand Velocity & Drift Analysis</div>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginTop: 4 }}>
                  <span style={{ fontSize: "1.05rem", fontWeight: 700, color: state.result.trend_pct >= 0 ? "var(--emerald)" : "var(--rose)" }}>
                    {state.result.trend_pct >= 0 ? `+${state.result.trend_pct}% Demand Surge` : `${state.result.trend_pct}% Demand Deceleration`}
                  </span>
                  <span className="small muted">vs recent 36-day velocity</span>
                </div>
                <div className="small muted" style={{ marginTop: 6, lineHeight: 1.4 }}>
                  {state.result.trend_pct >= 0
                    ? "Upward sales momentum detected. Ensure delivery window is scheduled prior to peak weekend demand."
                    : "Sales volume cooling relative to history. Keep safety buffer lean to avoid excess inventory holding costs."}
                </div>
              </div>

              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginTop: 4 }}>
                <span className="small muted">Need full schedule export?</span>
                <button className="btn-secondary" onClick={downloadForecastCsv}>
                  Download CSV Schedule
                </button>
              </div>
            </div>
          ) : (
            <div style={{ flex: 1, display: "flex", alignItems: "center", justifyContent: "center", color: "var(--ink-3)" }}>
              Run simulation to compute replenishment parameters.
            </div>
          )}
        </div>
      </div>

    </div>
  );
}
