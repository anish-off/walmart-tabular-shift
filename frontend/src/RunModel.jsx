import React, { useEffect, useMemo, useRef, useState } from "react";
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
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
  test: "Test / Local Dev Checkpoint",
};

const REQUIRED_COLS = ["date", "item_id", "store_id", "sales", "sell_price"];
const OPTIONAL_COLS = ["snap", "event_type_1", "dept_id", "cat_id", "state_id"];

const pct = (x) => (x == null ? "n/a" : (x * 100).toFixed(1) + "%");
const fmtNum = (x, d = 2) =>
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
    res = await fetch(path, opts);
  } catch {
    throw new Error(
      "Cannot reach model API. Start it with: .venv\\Scripts\\python -m uvicorn app:app --app-dir backend --port 8000"
    );
  }
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(typeof body.detail === "string" ? body.detail : "The request failed (" + res.status + ").");
  return body;
}

function parseSales(text) {
  const vals = text.split(/[\s,;]+/).filter(Boolean).map(Number);
  if (vals.some((v) => Number.isNaN(v))) return { error: "Sales must be numbers separated by commas or new lines." };
  return { vals };
}

function exampleSales() {
  let seed = 7;
  const rnd = () => ((seed = (seed * 16807) % 2147483647) / 2147483647);
  return Array.from({ length: 70 }, (_, i) =>
    Math.max(0, Math.round(5 + (i % 7 > 4 ? 4 : 0) + (rnd() - 0.5) * 4))
  ).join(", ");
}

function downloadCsvTemplate() {
  const template =
    "date,item_id,store_id,sales,sell_price,dept_id,cat_id,state_id,event_type_1,snap\n" +
    "2016-01-01,FOODS_3_090,CA_1,8,2.98,FOODS_3,FOODS,CA,,1\n" +
    "2016-01-02,FOODS_3_090,CA_1,11,2.98,FOODS_3,FOODS,CA,,1\n" +
    "2016-01-03,FOODS_3_090,CA_1,9,2.98,FOODS_3,FOODS,CA,,1\n";
  const blob = new Blob([template], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "retail_sales_template.csv";
  a.click();
  URL.revokeObjectURL(url);
}

function parseCsvClientSide(text) {
  const lines = text.trim().split(/\r?\n/).filter(Boolean);
  if (!lines.length) return { error: "CSV file is empty." };

  const rawHeaders = lines[0].split(",").map((c) => c.trim().replace(/^["']|["']$/g, ""));
  const headers = rawHeaders.map((c) => c.toLowerCase());

  const missing = REQUIRED_COLS.filter((c) => !headers.includes(c));
  const optionalFound = OPTIONAL_COLS.filter((c) => headers.includes(c));

  const previewRows = [];
  const seriesSet = new Set();
  const dateSet = [];

  const dateIdx = headers.indexOf("date");
  const itemIdx = headers.indexOf("item_id");
  const storeIdx = headers.indexOf("store_id");

  for (let i = 1; i < lines.length; i++) {
    const cols = lines[i].split(",").map((c) => c.trim().replace(/^["']|["']$/g, ""));
    if (cols.length < headers.length) continue;

    if (previewRows.length < 5) {
      const rowObj = {};
      rawHeaders.forEach((h, idx) => {
        rowObj[h] = cols[idx] ?? "";
      });
      previewRows.push(rowObj);
    }

    if (itemIdx >= 0 && storeIdx >= 0) {
      seriesSet.add(`${cols[itemIdx]}_${cols[storeIdx]}`);
    }
    if (dateIdx >= 0 && cols[dateIdx]) {
      dateSet.push(cols[dateIdx]);
    }
  }

  dateSet.sort();
  const minDate = dateSet[0] || "n/a";
  const maxDate = dateSet[dateSet.length - 1] || "n/a";

  return {
    headers: rawHeaders,
    missing,
    optionalFound,
    previewRows,
    rowCount: lines.length - 1,
    seriesCount: seriesSet.size,
    dateRange: `${minDate} → ${maxDate}`,
    isValid: missing.length === 0,
  };
}

function Status({ health }) {
  if (!health) return null;
  if (health.unreachable) {
    return (
      <div className="callout bad">
        <b>Model inference API unreachable.</b> Start the backend in terminal:
        <pre>.venv\Scripts\python -m uvicorn app:app --app-dir backend --port 8000</pre>
      </div>
    );
  }
  const bad = Object.entries(health).filter(([, v]) => !v.ok);
  if (!bad.length) return <div className="callout ok">All registered TabM checkpoints ready.</div>;
  return (
    <div className="callout bad">
      <b>{bad.length === Object.keys(health).length ? "No checkpoint could be loaded." : "Some checkpoints are incomplete."}</b>
      {bad.map(([k, v]) => (
        <div key={k} style={{ marginTop: 4 }}>
          <code>{k}</code>: {v.error}
        </div>
      ))}
    </div>
  );
}

function ModelSelect({ value, onChange, usable, health }) {
  const allModels = ["e2", "e1", "e3", "test"];
  return (
    <label className="field">
      <span>Target Checkpoint</span>
      <select value={value} onChange={(e) => onChange(e.target.value)}>
        {allModels.map((k) => {
          const isOk = usable.includes(k);
          const isConfigured = health && health[k];
          return (
            <option key={k} value={k} disabled={!isOk}>
              TabM ({EXP_LABEL[k] || k}) {isOk ? "— [Ready]" : isConfigured ? "— [Incomplete]" : "— [Not present]"}
            </option>
          );
        })}
      </select>
    </label>
  );
}

export default function RunModel() {
  const [health, setHealth] = useState(null);
  const [mode, setMode] = useState("forecast"); // Default to forecast visualization

  // State to bridge CSV series directly into Forecast view
  const [forecastPrefill, setForecastPrefill] = useState(null);

  useEffect(() => {
    api("/api/health")
      .then(setHealth)
      .catch(() => setHealth({ unreachable: true }));
  }, []);

  const usable = health && !health.unreachable ? Object.entries(health).filter(([, v]) => v.ok).map(([k]) => k) : [];

  const handleForwardProjectFromCsv = (seriesItem, targetModel) => {
    setForecastPrefill({
      model: targetModel,
      sales: seriesItem.sales_history.join(", "),
      end_date: seriesItem.last_date,
      price: seriesItem.last_price || 3.5,
      item_id: seriesItem.item_id || "ITEM_1",
      store_id: seriesItem.store_id || "STORE_1",
      horizon: 14,
      autoRun: true,
    });
    setMode("forecast");
  };

  return (
    <>
      <header className="top">
        <h1>Retail Forecast & Demand Telemetry</h1>
        <p>
          Generate forward recursive projections with confidence intervals, or ingest historical multi-series CSV actuals to
          benchmark TabM against rolling seasonal baselines.
        </p>
        <div className="controls">
          <div className="seg" role="group" aria-label="Task">
            <button aria-pressed={mode === "forecast"} onClick={() => setMode("forecast")}>
              Forward Forecast Visualization
            </button>
            <button aria-pressed={mode === "score"} onClick={() => setMode("score")}>
              Score a CSV Dataset
            </button>
          </div>
        </div>
      </header>

      <Status health={health} />

      {mode === "forecast" ? (
        <Forecast usable={usable} health={health} prefill={forecastPrefill} onClearPrefill={() => setForecastPrefill(null)} />
      ) : (
        <Score usable={usable} health={health} onForwardProject={handleForwardProjectFromCsv} />
      )}
    </>
  );
}

/* ─────────────────────────────────────────────────────────────────────────────
   FORWARD FORECAST VISUALIZATION (ENGINEERING-GRADE / ANTI-AI SLOP)
   ───────────────────────────────────────────────────────────────────────────── */
function Forecast({ usable, health, prefill, onClearPrefill }) {
  const [f, setF] = useState({
    model: "test",
    sales: "",
    end_date: new Date().toISOString().slice(0, 10),
    price: "3.50",
    future_price: "",
    horizon: 14,
    item_id: "FOODS_3_090",
    store_id: "CA_1",
  });

  const [state, setState] = useState({ busy: false, error: null, result: null });
  const [activeChartTab, setActiveChartTab] = useState("trajectory"); // 'trajectory' | 'cumulative' | 'seasonality' | 'table'

  const setField = (k) => (e) => setF((prev) => ({ ...prev, [k]: e.target.value }));

  // Handle prefill from CSV scoring
  useEffect(() => {
    if (prefill) {
      setF((prev) => ({
        ...prev,
        model: prefill.model || prev.model,
        sales: prefill.sales,
        end_date: prefill.end_date,
        price: String(prefill.price),
        future_price: String(prefill.price),
        item_id: prefill.item_id,
        store_id: prefill.store_id,
        horizon: prefill.horizon || 14,
      }));
      if (prefill.autoRun) {
        executeForecast({
          model: prefill.model || f.model,
          salesText: prefill.sales,
          end_date: prefill.end_date,
          price: Number(prefill.price),
          future_price: Number(prefill.price),
          horizon: prefill.horizon || 14,
          item_id: prefill.item_id,
          store_id: prefill.store_id,
        });
      }
      onClearPrefill();
    }
  }, [prefill]); // eslint-disable-line

  // Auto-select valid model
  useEffect(() => {
    if (usable.length && !usable.includes(f.model)) {
      setF((p) => ({ ...p, model: usable[0] }));
    }
  }, [usable.join()]); // eslint-disable-line

  async function executeForecast(params) {
    const { vals, error } = parseSales(params.salesText || f.sales);
    if (error) return setState({ busy: false, error, result: null });

    setState({ busy: true, error: null, result: null });
    try {
      const payload = {
        model: params.model || f.model,
        sales: vals,
        end_date: params.end_date || f.end_date,
        price: Number(params.price ?? f.price),
        future_price: f.future_price ? Number(f.future_price) : undefined,
        horizon: Number(params.horizon || f.horizon),
        item_id: params.item_id || f.item_id,
        store_id: params.store_id || f.store_id,
      };

      const result = await api("/api/forecast", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      setState({ busy: false, error: null, result });
    } catch (err) {
      setState({ busy: false, error: err.message, result: null });
    }
  }

  function handleSubmit(e) {
    e.preventDefault();
    executeForecast({});
  }

  // Combined time series for Fan Chart
  const trajectoryData = useMemo(() => {
    const r = state.result;
    if (!r) return [];
    const tail = r.history_tail || [];
    const fc = r.forecast || [];
    if (!tail.length && !fc.length) return [];

    const lastHist = tail[tail.length - 1];

    const historyPoints = tail.map((p) => ({
      date: p.date,
      history: p.sales,
      forecast: null,
      band: null,
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

    // Connect the history line to the first forecast point
    if (historyPoints.length && forecastPoints.length) {
      historyPoints[historyPoints.length - 1].forecast = lastHist.sales;
      historyPoints[historyPoints.length - 1].band = [lastHist.sales, lastHist.sales];
    }

    return [...historyPoints, ...forecastPoints];
  }, [state.result]);

  const splitDate = state.result?.forecast?.[0]?.date;

  return (
    <div className="grid2">
      {/* LEFT FORM PANEL */}
      <section className="panel">
        <div className="head">
          <div>
            <h2>Forecast Configuration</h2>
            <p>Provide at least 36 consecutive daily unit sales for one retail series, oldest first.</p>
          </div>
        </div>

        <form onSubmit={handleSubmit} className="form">
          <ModelSelect value={f.model} onChange={(v) => setF({ ...f, model: v })} usable={usable} health={health} />

          <label className="field">
            <span>Daily Sales Sequence (comma separated, oldest to newest)</span>
            <textarea
              rows={4}
              value={f.sales}
              onChange={setField("sales")}
              placeholder="5, 6, 8, 12, 10, 4, 5, 7, ..."
              required
            />
          </label>

          <button
            type="button"
            className="link"
            onClick={() => setF((prev) => ({ ...prev, sales: exampleSales() }))}
          >
            Populate with synthetic example series (70 days)
          </button>

          <div className="row2">
            <label className="field">
              <span>Date of Last Historical Point</span>
              <input type="date" value={f.end_date} onChange={setField("end_date")} required />
            </label>
            <label className="field">
              <span>Historical Unit Price ($)</span>
              <input type="number" step="0.01" min="0" value={f.price} onChange={setField("price")} required />
            </label>
          </div>

          <div className="row2">
            <label className="field">
              <span>Future Price Scenario ($) <span className="muted">(Optional what-if)</span></span>
              <input
                type="number"
                step="0.01"
                min="0"
                placeholder={f.price || "Same as history"}
                value={f.future_price}
                onChange={setField("future_price")}
              />
            </label>
            <label className="field">
              <span>Forecast Horizon</span>
              <div className="quick-horizon-bar" style={{ marginTop: 2 }}>
                {[7, 14, 21, 28].map((h) => (
                  <button
                    key={h}
                    type="button"
                    aria-pressed={Number(f.horizon) === h}
                    onClick={() => setF((prev) => ({ ...prev, horizon: h }))}
                  >
                    {h}d
                  </button>
                ))}
              </div>
            </label>
          </div>

          <div className="row2">
            <label className="field">
              <span>Item ID</span>
              <input value={f.item_id} onChange={setField("item_id")} placeholder="FOODS_3_090" />
            </label>
            <label className="field">
              <span>Store ID</span>
              <input value={f.store_id} onChange={setField("store_id")} placeholder="CA_1" />
            </label>
          </div>

          <button className="primary" disabled={state.busy || !usable.length} style={{ marginTop: 8 }}>
            {state.busy ? (
              <>
                <span className="spinner" style={{ marginRight: 8 }} />
                Generating Projections...
              </>
            ) : (
              `Generate ${f.horizon}-Day Forecast`
            )}
          </button>
        </form>
      </section>

      {/* RIGHT VISUALIZATION PANEL */}
      <section className="panel">
        <div className="head">
          <div>
            <h2>Forecast Visualization & Analytics</h2>
            <p>Recursive step projections with statistical uncertainty bounds and day-of-week seasonality decomposition.</p>
          </div>
        </div>

        {state.error && <div className="callout bad">{state.error}</div>}

        {!state.result && !state.error && !state.busy && (
          <div style={{ padding: "60px 10px", textAlign: "center", color: "var(--ink-2)" }}>
            <p>Configure historical sales parameters and execute a projection to generate the forecast visualization.</p>
          </div>
        )}

        {state.busy && (
          <div style={{ padding: "60px 10px", textAlign: "center" }}>
            <div className="spinner" style={{ width: 32, height: 32, borderWidth: 3, margin: "0 auto 16px" }} />
            <p className="muted">Evaluating TabM multi-step recursive projection & prediction intervals...</p>
          </div>
        )}

        {state.result && (
          <>
            {/* KPI METRIC CARDS */}
            <div className="metric-row">
              <div className="metric-tile">
                <div className="label">Total Projected Volume</div>
                <div className="val">{fmtNum(state.result.total, 1)}</div>
                <div className="sub">Units over {state.result.forecast.length} days</div>
              </div>
              <div className="metric-tile">
                <div className="label">Projected Revenue</div>
                <div className="val">{fmtCurrency(state.result.projected_revenue)}</div>
                <div className="sub">@ ${fmtNum(state.result.price_used)} / unit</div>
              </div>
              <div className="metric-tile">
                <div className="label">Daily Run Rate</div>
                <div className="val">{fmtNum(state.result.avg_daily, 1)}</div>
                <div className="sub">Units / day average</div>
              </div>
              <div className="metric-tile">
                <div className="label">Peak Demand Day</div>
                <div className="val" style={{ fontSize: "1.35rem" }}>
                  {state.result.peak_day?.date ? state.result.peak_day.date.slice(5) : "—"}
                </div>
                <div className="sub">{fmtNum(state.result.peak_day?.units, 1)} units peak</div>
              </div>
            </div>

            {/* VISUALIZATION VIEW SELECTOR TABS */}
            <div className="viz-nav">
              <div className="seg" role="group" aria-label="Forecast Views">
                <button
                  aria-pressed={activeChartTab === "trajectory"}
                  onClick={() => setActiveChartTab("trajectory")}
                >
                  Fan Chart & Interval
                </button>
                <button
                  aria-pressed={activeChartTab === "cumulative"}
                  onClick={() => setActiveChartTab("cumulative")}
                >
                  Cumulative Volume
                </button>
                <button
                  aria-pressed={activeChartTab === "seasonality"}
                  onClick={() => setActiveChartTab("seasonality")}
                >
                  Weekly Seasonality
                </button>
                <button
                  aria-pressed={activeChartTab === "table"}
                  onClick={() => setActiveChartTab("table")}
                >
                  Data Table
                </button>
              </div>

              {activeChartTab === "trajectory" && (
                <div className="legend-band">
                  <span className="legend-band-box" />
                  <span>80% Prediction Interval</span>
                </div>
              )}
            </div>

            {/* VIEW 1: TRAJECTORY & FAN CHART */}
            {activeChartTab === "trajectory" && (
              <div className="chart-box">
                <div className="chart-box-header">
                  <span className="chart-box-title">
                    Historical Demand &rarr; TabM Projection Horizon
                  </span>
                  <span className="badge info">Model: {state.result.model}</span>
                </div>

                <div style={{ height: 280, width: "100%" }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={trajectoryData} margin={{ left: 0, right: 12, top: 10, bottom: 0 }}>
                      <defs>
                        <linearGradient id="bandGrad" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="5%" stopColor="var(--m-tabm)" stopOpacity={0.22} />
                          <stop offset="95%" stopColor="var(--m-tabm)" stopOpacity={0.04} />
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
                      <YAxis stroke="var(--ink-3)" tick={{ fill: "var(--ink-2)" }} width={42} />
                      <Tooltip
                        content={({ active, payload, label }) => {
                          if (!active || !payload?.length) return null;
                          const pHistory = payload.find((p) => p.dataKey === "history")?.value;
                          const pForecast = payload.find((p) => p.dataKey === "forecast")?.value;
                          const row = payload[0]?.payload;
                          return (
                            <div className="tip">
                              <div>
                                <b>Date: {label}</b>
                              </div>
                              {pHistory != null && (
                                <div style={{ color: "var(--ink-2)" }}>History: {fmtNum(pHistory, 1)} units</div>
                              )}
                              {pForecast != null && (
                                <>
                                  <div style={{ color: "var(--m-tabm)", fontWeight: 700 }}>
                                    TabM Projected: {fmtNum(pForecast, 2)} units
                                  </div>
                                  {row?.lower != null && (
                                    <div style={{ color: "var(--ink-3)", fontSize: "0.78rem" }}>
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
                          stroke="var(--warn)"
                          strokeDasharray="4 4"
                          label={{ value: "Forecast Start", position: "top", fill: "var(--warn)", fontSize: 11 }}
                        />
                      )}
                      {/* Prediction uncertainty band */}
                      <Area
                        type="monotone"
                        dataKey="band"
                        stroke="none"
                        fill="url(#bandGrad)"
                        isAnimationActive={false}
                        connectNulls
                      />
                      {/* History line */}
                      <Line
                        type="monotone"
                        dataKey="history"
                        name="Historical Actuals"
                        stroke="var(--ink-2)"
                        strokeWidth={2}
                        dot={false}
                        isAnimationActive={false}
                        connectNulls
                      />
                      {/* Forecast line */}
                      <Line
                        type="monotone"
                        dataKey="forecast"
                        name="TabM Projected"
                        stroke="var(--m-tabm)"
                        strokeWidth={2.75}
                        dot={{ r: 3, fill: "var(--m-tabm)" }}
                        isAnimationActive={false}
                        connectNulls
                      />
                    </AreaChart>
                  </ResponsiveContainer>
                </div>
              </div>
            )}

            {/* VIEW 2: CUMULATIVE DEMAND CURVE */}
            {activeChartTab === "cumulative" && (
              <div className="chart-box">
                <div className="chart-box-header">
                  <span className="chart-box-title">Cumulative Projected Demand Curve</span>
                  <span className="small muted">Total units needed over forecast horizon</span>
                </div>
                <div style={{ height: 280, width: "100%" }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={state.result.forecast} margin={{ left: 0, right: 12, top: 10, bottom: 0 }}>
                      <defs>
                        <linearGradient id="cumGrad" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="5%" stopColor="var(--m-tabpfn)" stopOpacity={0.3} />
                          <stop offset="95%" stopColor="var(--m-tabpfn)" stopOpacity={0.02} />
                        </linearGradient>
                      </defs>
                      <CartesianGrid vertical={false} stroke="var(--line)" />
                      <XAxis
                        dataKey="date"
                        tickFormatter={(d) => (d ? d.slice(5) : "")}
                        minTickGap={20}
                        stroke="var(--ink-3)"
                        tick={{ fill: "var(--ink-2)", fontSize: 11 }}
                      />
                      <YAxis stroke="var(--ink-3)" tick={{ fill: "var(--ink-2)" }} width={48} />
                      <Tooltip
                        content={({ active, payload, label }) => {
                          if (!active || !payload?.length) return null;
                          const val = payload[0]?.value;
                          return (
                            <div className="tip">
                              <b>Date: {label}</b>
                              <div style={{ color: "var(--m-tabpfn)", fontWeight: 700 }}>
                                Cumulative Demand: {fmtNum(val, 1)} units
                              </div>
                            </div>
                          );
                        }}
                      />
                      <Area
                        type="monotone"
                        dataKey="cumulative"
                        name="Cumulative Units"
                        stroke="var(--m-tabpfn)"
                        strokeWidth={2.5}
                        fill="url(#cumGrad)"
                        isAnimationActive={false}
                      />
                    </AreaChart>
                  </ResponsiveContainer>
                </div>
              </div>
            )}

            {/* VIEW 3: DAY-OF-WEEK SEASONALITY BAR CHART */}
            {activeChartTab === "seasonality" && (
              <div className="chart-box">
                <div className="chart-box-header">
                  <span className="chart-box-title">Weekly Day-of-Week Seasonality Decomposition</span>
                  <span className="small muted">Historical Avg vs TabM Forecast Avg</span>
                </div>
                <div style={{ height: 280, width: "100%" }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={state.result.seasonality} margin={{ left: 0, right: 12, top: 10, bottom: 0 }}>
                      <CartesianGrid vertical={false} stroke="var(--line)" />
                      <XAxis dataKey="day" stroke="var(--ink-3)" tick={{ fill: "var(--ink-2)", fontSize: 11 }} />
                      <YAxis stroke="var(--ink-3)" tick={{ fill: "var(--ink-2)" }} width={40} />
                      <Tooltip
                        content={({ active, payload, label }) => {
                          if (!active || !payload?.length) return null;
                          return (
                            <div className="tip">
                              <b>{label}</b>
                              <div style={{ color: "var(--ink-2)" }}>
                                Historical Avg: {fmtNum(payload[0]?.value, 2)} units
                              </div>
                              <div style={{ color: "var(--m-tabm)", fontWeight: 700 }}>
                                Projected Avg: {fmtNum(payload[1]?.value, 2)} units
                              </div>
                            </div>
                          );
                        }}
                      />
                      <Legend />
                      <Bar dataKey="history_avg" name="Historical Average" fill="var(--ink-3)" radius={[4, 4, 0, 0]} />
                      <Bar dataKey="forecast_avg" name="Forecast Average" fill="var(--m-tabm)" radius={[4, 4, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>
            )}

            {/* VIEW 4: DATA TABLE */}
            {activeChartTab === "table" && (
              <div className="scroll" style={{ maxHeight: 310, overflowY: "auto", border: "1px solid var(--line)", borderRadius: 8 }}>
                <table>
                  <thead>
                    <tr>
                      <th>Date</th>
                      <th>Projected Units</th>
                      <th>80% Confidence Interval</th>
                      <th>Cumulative Volume</th>
                    </tr>
                  </thead>
                  <tbody>
                    {state.result.forecast.map((r) => (
                      <tr key={r.date}>
                        <td className="mono">{r.date}</td>
                        <td style={{ color: "var(--m-tabm)", fontWeight: 700 }}>{fmtNum(r.forecast, 2)}</td>
                        <td className="mono muted">
                          [{fmtNum(r.lower, 1)} &ndash; {fmtNum(r.upper, 1)}]
                        </td>
                        <td className="mono">{fmtNum(r.cumulative, 1)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </>
        )}
      </section>
    </div>
  );
}

/* ─────────────────────────────────────────────────────────────────────────────
   SCORE A CSV COMPONENT (WITH FORWARD PROJECTION HOOK)
   ───────────────────────────────────────────────────────────────────────────── */
function Score({ usable, health, onForwardProject }) {
  const [model, setModel] = useState("test");
  const [file, setFile] = useState(null);
  const [inspect, setInspect] = useState(null);
  const [isDragging, setIsDragging] = useState(false);
  const [loadingSample, setLoadingSample] = useState(false);
  const [state, setState] = useState({ busy: false, error: null, result: null });
  const [selectedSeriesIdx, setSelectedSeriesIdx] = useState(0);
  const [searchDate, setSearchDate] = useState("");
  const fileInputRef = useRef(null);

  useEffect(() => {
    if (usable.length && !usable.includes(model)) {
      setModel(usable[0]);
    }
  }, [usable.join()]); // eslint-disable-line

  const handleFileProcess = (chosenFile) => {
    if (!chosenFile) return;
    setFile(chosenFile);
    setState({ busy: false, error: null, result: null });

    const reader = new FileReader();
    reader.onload = (e) => {
      const text = e.target.result;
      const parsed = parseCsvClientSide(text);
      setInspect(parsed);
    };
    reader.readAsText(chosenFile.slice(0, 150000));
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileProcess(e.dataTransfer.files[0]);
    }
  };

  const loadBuiltinSample = async () => {
    setLoadingSample(true);
    try {
      const res = await fetch("/sample_sales.csv");
      if (!res.ok) throw new Error("Could not fetch /sample_sales.csv");
      const blob = await res.blob();
      const sampleFile = new File([blob], "sample_sales.csv", { type: "text/csv" });
      handleFileProcess(sampleFile);
    } catch (err) {
      setState((prev) => ({ ...prev, error: `Failed to load sample: ${err.message}` }));
    } finally {
      setLoadingSample(false);
    }
  };

  const clearFile = () => {
    setFile(null);
    setInspect(null);
    setState({ busy: false, error: null, result: null });
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  async function runScoring(e) {
    e.preventDefault();
    if (!file) return setState({ busy: false, error: "Select or drop a CSV file first.", result: null });
    if (inspect && !inspect.isValid) {
      return setState({ busy: false, error: `Missing required columns: ${inspect.missing.join(", ")}`, result: null });
    }

    setState({ busy: true, error: null, result: null });
    const body = new FormData();
    body.append("file", file);
    body.append("model", model);

    try {
      const res = await api("/api/score", { method: "POST", body });
      setState({ busy: false, error: null, result: res });
      setSelectedSeriesIdx(0);
    } catch (err) {
      setState({ busy: false, error: err.message, result: null });
    }
  }

  function downloadResultsCsv() {
    const r = state.result;
    if (!r?.rows) return;
    const header = "id,date,actual,pred,abs_error\n";
    const body = r.rows
      .map((x) => [x.id, x.date, x.actual, x.pred, Math.abs(x.actual - x.pred).toFixed(3)].join(","))
      .join("\n");
    const a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob([header + body], { type: "text/csv" }));
    a.download = `tabm_${model}_predictions.csv`;
    a.click();
    URL.revokeObjectURL(a.href);
  }

  const result = state.result;
  const activeSeries = result?.series?.[selectedSeriesIdx] || result?.series?.[0];

  const filteredRows = useMemo(() => {
    if (!result?.rows) return [];
    let list = result.rows;
    if (activeSeries?.id) {
      list = list.filter((r) => r.id === activeSeries.id);
    }
    if (searchDate.trim()) {
      list = list.filter((r) => r.date.includes(searchDate.trim()));
    }
    return list;
  }, [result, activeSeries, searchDate]);

  return (
    <div className="grid2">
      {/* LEFT PANEL */}
      <section className="panel">
        <div className="head">
          <div>
            <h2>Dataset Ingestion</h2>
            <p>One row per item/store/day. Continuous daily timestamps required (minimum 36 continuous days for lag_35).</p>
          </div>
        </div>

        <form onSubmit={runScoring} className="form">
          <ModelSelect value={model} onChange={setModel} usable={usable} health={health} />

          <div
            className={`drop-zone ${isDragging ? "dragging" : ""} ${file ? "has-file" : ""}`}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            onClick={() => !file && fileInputRef.current?.click()}
          >
            <input
              type="file"
              ref={fileInputRef}
              accept=".csv,text/csv"
              style={{ display: "none" }}
              onChange={(e) => handleFileProcess(e.target.files?.[0])}
            />

            {!file ? (
              <>
                <svg className="drop-zone-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.75">
                  <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                  <polyline points="17 8 12 3 7 8" />
                  <line x1="12" y1="3" x2="12" y2="15" />
                </svg>
                <div className="drop-zone-title">Drop CSV file here, or click to browse</div>
                <div className="drop-zone-subtitle">Required columns: date, item_id, store_id, sales, sell_price</div>
                <div className="toolbar-actions" onClick={(e) => e.stopPropagation()}>
                  <button type="button" className="secondary" onClick={() => fileInputRef.current?.click()}>
                    Browse Local File
                  </button>
                  <button type="button" className="secondary" onClick={loadBuiltinSample} disabled={loadingSample}>
                    {loadingSample ? "Loading..." : "Load Built-in Sample (sample_sales.csv)"}
                  </button>
                </div>
              </>
            ) : (
              <div className="file-meta-card" onClick={(e) => e.stopPropagation()}>
                <div className="file-meta-info">
                  <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
                    <polyline points="14 2 14 8 20 8" />
                  </svg>
                  <div>
                    <div className="file-meta-name">{file.name}</div>
                    <div className="file-meta-size">
                      {(file.size / 1024).toFixed(1)} KB &bull; {inspect?.rowCount ? `${inspect.rowCount.toLocaleString()} rows` : "Parsing..."}
                    </div>
                  </div>
                </div>
                <button type="button" className="secondary" onClick={clearFile} style={{ padding: "6px 10px" }}>
                  Replace
                </button>
              </div>
            )}
          </div>

          {inspect && (
            <div className="preflight-card">
              <div className="preflight-header">
                <span className="preflight-title">Pre-Flight Schema Validation</span>
                <span className={`badge ${inspect.isValid ? "ok" : "err"}`}>
                  {inspect.isValid ? "Valid Schema" : `Missing ${inspect.missing.length} Column(s)`}
                </span>
              </div>

              <div>
                <span className="small muted">Required columns check:</span>
                <div className="badge-group">
                  {REQUIRED_COLS.map((col) => {
                    const found = !inspect.missing.includes(col);
                    return (
                      <span key={col} className={`badge ${found ? "ok" : "err"}`}>
                        <span style={{ fontWeight: 800 }}>{found ? "PASS" : "MISSING"}</span> {col}
                      </span>
                    );
                  })}
                </div>
              </div>

              {inspect.optionalFound.length > 0 && (
                <div style={{ marginTop: 10 }}>
                  <span className="small muted">Optional features detected:</span>
                  <div className="badge-group">
                    {inspect.optionalFound.map((col) => (
                      <span key={col} className="badge info">
                        + {col}
                      </span>
                    ))}
                  </div>
                </div>
              )}

              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8, marginTop: 12 }}>
                <div className="stat" style={{ paddingLeft: 8 }}>
                  <div className="v" style={{ fontSize: "1.2rem" }}>
                    {inspect.seriesCount}
                  </div>
                  <div className="l">Series detected</div>
                </div>
                <div className="stat" style={{ paddingLeft: 8 }}>
                  <div className="v" style={{ fontSize: "1rem", fontFamily: "ui-monospace, monospace" }}>
                    {inspect.dateRange}
                  </div>
                  <div className="l">Date coverage</div>
                </div>
              </div>

              {inspect.previewRows?.length > 0 && (
                <div className="preview-box">
                  <div className="preview-box-header">
                    <span>First 5 parsed records</span>
                    <span className="muted">{inspect.headers.join(", ")}</span>
                  </div>
                  <div className="scroll">
                    <table>
                      <thead>
                        <tr>
                          {inspect.headers.map((h) => (
                            <th key={h}>{h}</th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {inspect.previewRows.map((r, idx) => (
                          <tr key={idx}>
                            {inspect.headers.map((h) => (
                              <td key={h}>{r[h] ?? ""}</td>
                            ))}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}
            </div>
          )}

          <div className="toolbar-actions">
            <button className="primary" disabled={state.busy || !file || (inspect && !inspect.isValid) || !usable.length}>
              {state.busy ? (
                <>
                  <span className="spinner" style={{ marginRight: 8 }} />
                  Extracting Features & Scoring...
                </>
              ) : (
                "Score Dataset with TabM"
              )}
            </button>
            <button type="button" className="secondary" onClick={downloadCsvTemplate}>
              Download CSV Template
            </button>
          </div>
        </form>
      </section>

      {/* RIGHT RESULTS PANEL */}
      <section className="panel">
        <div className="head">
          <div>
            <h2>Model Performance & Scoring</h2>
            <p>Weighted Absolute Percentage Error (WAPE) compared directly against a repeat-last-week (lag_7) seasonal baseline.</p>
          </div>
        </div>

        {state.error && <div className="callout bad">{state.error}</div>}

        {!result && !state.error && !state.busy && (
          <div style={{ padding: "40px 10px", textAlign: "center", color: "var(--ink-2)" }}>
            <p>Upload a CSV dataset and run scoring to view per-series forecasting telemetry and residual analysis.</p>
          </div>
        )}

        {state.busy && (
          <div style={{ padding: "50px 10px", textAlign: "center" }}>
            <div className="spinner" style={{ width: 28, height: 28, borderWidth: 3, margin: "0 auto 16px" }} />
            <p className="muted">Running feature transformation pipeline and evaluating PyTorch checkpoint...</p>
          </div>
        )}

        {result && (
          <>
            <div className="metric-row">
              <div className="metric-tile">
                <div className="label">TabM WAPE</div>
                <div className="val">{pct(result.wape)}</div>
                <div className="sub">Weighted percentage error</div>
              </div>
              <div className="metric-tile">
                <div className="label">Lag-7 Baseline</div>
                <div className="val">{pct(result.wape_lag7_baseline)}</div>
                <div className="sub">Repeat-last-week error</div>
              </div>
              <div className="metric-tile">
                <div className="label">Relative Delta</div>
                <div
                  className="val"
                  style={{ color: (result.wape || 0) <= (result.wape_lag7_baseline || 0) ? "var(--m-tabpfn)" : "var(--warn)" }}
                >
                  {result.wape_lag7_baseline
                    ? `${(((result.wape - result.wape_lag7_baseline) / result.wape_lag7_baseline) * 100).toFixed(1)}%`
                    : "—"}
                </div>
                <div className="sub">Error delta vs baseline</div>
              </div>
              <div className="metric-tile">
                <div className="label">Scored Horizon</div>
                <div className="val">{result.n_rows.toLocaleString()}</div>
                <div className="sub">{result.n_series} unique series</div>
              </div>
            </div>

            {/* SERIES SWITCHER & FORWARD PROJECT ACTION */}
            {result.series?.length > 0 && (
              <>
                <div style={{ margin: "18px 0 8px", display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: 8 }}>
                  <span className="small muted">Evaluated Series:</span>
                  {activeSeries?.sales_history && (
                    <button
                      type="button"
                      className="secondary"
                      style={{ padding: "4px 10px", fontSize: "0.82rem", borderColor: "var(--m-tabm)", color: "var(--m-tabm)" }}
                      onClick={() => onForwardProject(activeSeries, model)}
                    >
                      Project Future Forecast for {activeSeries.id} &rarr;
                    </button>
                  )}
                </div>
                <div className="seg" role="group" aria-label="Series selector">
                  {result.series.map((s, i) => (
                    <button key={s.id} aria-pressed={selectedSeriesIdx === i} onClick={() => setSelectedSeriesIdx(i)}>
                      {s.id} &bull; {pct(s.wape)}
                    </button>
                  ))}
                </div>
              </>
            )}

            {activeSeries?.points && (
              <div style={{ height: 260, marginTop: 16 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={activeSeries.points} margin={{ left: 0, right: 12, top: 8, bottom: 0 }}>
                    <CartesianGrid vertical={false} stroke="var(--line)" />
                    <XAxis
                      dataKey="date"
                      tickFormatter={(d) => d.slice(5)}
                      minTickGap={20}
                      stroke="var(--ink-3)"
                      tick={{ fill: "var(--ink-2)", fontSize: 11 }}
                    />
                    <YAxis stroke="var(--ink-3)" tick={{ fill: "var(--ink-2)" }} width={42} />
                    <Tooltip
                      content={({ active, payload, label }) => {
                        if (!active || !payload?.length) return null;
                        const actualVal = payload.find((p) => p.dataKey === "actual")?.value;
                        const predVal = payload.find((p) => p.dataKey === "pred")?.value;
                        const res = actualVal != null && predVal != null ? Math.abs(actualVal - predVal).toFixed(2) : "—";
                        return (
                          <div className="tip">
                            <div>
                              <b>Date: {label}</b>
                            </div>
                            <div style={{ color: "var(--ink-2)" }}>Actual: {actualVal}</div>
                            <div style={{ color: "var(--m-tabm)" }}>TabM Forecast: {predVal}</div>
                            <div style={{ color: "var(--warn)" }}>Absolute Error: {res}</div>
                          </div>
                        );
                      }}
                    />
                    <Legend />
                    <Line isAnimationActive={false} dataKey="actual" name="Actual Units" stroke="var(--ink-2)" strokeWidth={2} dot={false} />
                    <Line isAnimationActive={false} dataKey="pred" name="TabM Predicted" stroke="var(--m-tabm)" strokeWidth={2.5} dot={{ r: 2.5 }} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            )}

            <div style={{ marginTop: 20 }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
                <span className="small muted">Record Explorer ({filteredRows.length} rows)</span>
                <input
                  type="search"
                  placeholder="Filter by date (YYYY-MM-DD)..."
                  value={searchDate}
                  onChange={(e) => setSearchDate(e.target.value)}
                  style={{ minWidth: 200, padding: "4px 8px", fontSize: "0.82rem" }}
                />
              </div>

              <div className="scroll" style={{ maxHeight: 200, overflowY: "auto", border: "1px solid var(--line)", borderRadius: 8 }}>
                <table>
                  <thead>
                    <tr>
                      <th>Series</th>
                      <th>Date</th>
                      <th>Actual</th>
                      <th>TabM Predicted</th>
                      <th>Residual (|A - P|)</th>
                    </tr>
                  </thead>
                  <tbody>
                    {filteredRows.slice(0, 100).map((r, i) => {
                      const diff = Math.abs(r.actual - r.pred);
                      return (
                        <tr key={i}>
                          <td className="mono">{r.id}</td>
                          <td className="mono">{r.date}</td>
                          <td>{fmtNum(r.actual, 1)}</td>
                          <td style={{ color: "var(--m-tabm)", fontWeight: 600 }}>{fmtNum(r.pred, 2)}</td>
                          <td className="muted">{fmtNum(diff, 2)}</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>

            <div className="toolbar-actions" style={{ marginTop: 14 }}>
              <button className="secondary" onClick={downloadResultsCsv}>
                Export Predictions CSV
              </button>
              {result.truncated && <span className="small muted">Dataset truncated to first 20,000 scored rows for memory efficiency.</span>}
            </div>
          </>
        )}
      </section>
    </div>
  );
}
