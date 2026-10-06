import React, { useEffect, useMemo, useState } from "react";
import { CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

const EXP_LABEL = { e1: "E1 (random split)", e2: "E2 (temporal split)", e3: "E3 (cold-start items)" };
const pct = (x) => (x == null ? "n/a" : (x * 100).toFixed(1) + "%");

async function api(path, opts) {
  let res;
  try {
    res = await fetch(path, opts);
  } catch {
    throw new Error("Cannot reach the model API. Start it with the command shown above.");
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

// Synthetic demo only: weekly pattern plus noise. Not real Walmart data.
function exampleSales() {
  let seed = 7;
  const rnd = () => ((seed = (seed * 16807) % 2147483647) / 2147483647);
  return Array.from({ length: 70 }, (_, i) => Math.max(0, Math.round(5 + (i % 7 > 4 ? 3 : 0) + (rnd() - 0.5) * 4))).join(", ");
}

function Status({ health }) {
  if (!health) return null;
  if (health.unreachable) {
    return (
      <div className="callout bad">
        <b>The model API is not running.</b> In a terminal at the repo root run:
        <pre>.venv\Scripts\python -m uvicorn app:app --app-dir backend --port 8000</pre>
      </div>
    );
  }
  const bad = Object.entries(health).filter(([, v]) => !v.ok);
  if (!bad.length) return <div className="callout ok">All three TabM checkpoints loaded.</div>;
  return (
    <div className="callout bad">
      <b>{bad.length === 3 ? "No checkpoint can be loaded." : "Some checkpoints cannot be loaded."}</b>
      {bad.map(([k, v]) => <div key={k}>{EXP_LABEL[k]}: {v.error}</div>)}
    </div>
  );
}

export default function RunModel() {
  const [health, setHealth] = useState(null);
  const [mode, setMode] = useState("forecast");
  useEffect(() => {
    api("/api/health").then(setHealth).catch(() => setHealth({ unreachable: true }));
  }, []);
  const usable = health && !health.unreachable ? Object.entries(health).filter(([, v]) => v.ok).map(([k]) => k) : [];
  return (
    <>
      <header className="top">
        <h1>Run the model on your own sales.</h1>
        <p>
          Enter a sales history and TabM forecasts the next days, or upload a CSV and see how the model scores against
          your actuals. The model runs on the local API, not in the browser.
        </p>
        <div className="controls">
          <div className="seg" role="group" aria-label="Task">
            <button aria-pressed={mode === "forecast"} onClick={() => setMode("forecast")}>Forecast one item</button>
            <button aria-pressed={mode === "score"} onClick={() => setMode("score")}>Score a CSV</button>
          </div>
        </div>
      </header>
      <Status health={health} />
      {mode === "forecast" ? <Forecast usable={usable} /> : <Score usable={usable} />}
    </>
  );
}

function ModelSelect({ value, onChange, usable }) {
  return (
    <label className="field">
      <span>Model</span>
      <select value={value} onChange={(e) => onChange(e.target.value)}>
        {["e1", "e2", "e3"].map((e) => (
          <option key={e} value={e} disabled={!usable.includes(e)}>TabM, {EXP_LABEL[e]}{usable.includes(e) ? "" : " (unavailable)"}</option>
        ))}
      </select>
    </label>
  );
}

function Forecast({ usable }) {
  const [f, setF] = useState({
    model: "e2", sales: "", end_date: new Date().toISOString().slice(0, 10), price: "3.50", horizon: 14,
    item_id: "FOODS_3_090", store_id: "CA_1",
  });
  const [state, setState] = useState({ busy: false, error: null, result: null });
  const set = (k) => (e) => setF({ ...f, [k]: e.target.value });
  useEffect(() => {
    if (usable.length && !usable.includes(f.model)) setF((p) => ({ ...p, model: usable[0] }));
  }, [usable.join()]); // eslint-disable-line

  async function run(e) {
    e.preventDefault();
    const { vals, error } = parseSales(f.sales);
    if (error) return setState({ busy: false, error, result: null });
    setState({ busy: true, error: null, result: null });
    try {
      const result = await api("/api/forecast", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model: f.model, sales: vals, end_date: f.end_date, price: Number(f.price),
          horizon: Number(f.horizon), item_id: f.item_id, store_id: f.store_id,
        }),
      });
      setState({ busy: false, error: null, result });
    } catch (err) {
      setState({ busy: false, error: err.message, result: null });
    }
  }

  const chart = useMemo(() => {
    const r = state.result;
    if (!r) return [];
    const last = r.history_tail[r.history_tail.length - 1];
    return [
      ...r.history_tail.map((p) => ({ date: p.date, history: p.sales })),
      ...r.forecast.map((p, i) => ({ date: p.date, forecast: p.forecast, ...(i === 0 ? {} : {}) })),
    ].map((p, i, a) => (p.date === r.forecast[0].date ? { ...p, history: last.sales } : p));
  }, [state.result]);

  return (
    <div className="grid2">
      <section className="panel">
        <div className="head"><div><h2>Inputs</h2><p>Give at least 36 days of daily unit sales for one item in one store, oldest first.</p></div></div>
        <form onSubmit={run} className="form">
          <ModelSelect value={f.model} onChange={(v) => setF({ ...f, model: v })} usable={usable} />
          <label className="field">
            <span>Daily sales (oldest to newest)</span>
            <textarea rows={5} value={f.sales} onChange={set("sales")} placeholder="4, 6, 5, 9, 11, 5, 4, ..." required />
          </label>
          <button type="button" className="link" onClick={() => setF({ ...f, sales: exampleSales() })}>Fill with synthetic example data</button>
          <div className="row2">
            <label className="field"><span>Date of last value</span><input type="date" value={f.end_date} onChange={set("end_date")} required /></label>
            <label className="field"><span>Price</span><input type="number" step="0.01" min="0" value={f.price} onChange={set("price")} required /></label>
          </div>
          <div className="row2">
            <label className="field"><span>Item ID</span><input value={f.item_id} onChange={set("item_id")} /></label>
            <label className="field"><span>Store ID</span><input value={f.store_id} onChange={set("store_id")} /></label>
          </div>
          <label className="field"><span>Days to forecast: {f.horizon}</span>
            <input type="range" min="1" max="28" value={f.horizon} onChange={set("horizon")} /></label>
          <button className="primary" disabled={state.busy || !usable.length}>{state.busy ? "Running model…" : "Run forecast"}</button>
          <p className="muted small">Item and store names that follow M5 patterns (FOODS_3_090, CA_1) set the category, department, state and store inputs. Other names fall back to defaults.</p>
        </form>
      </section>

      <section className="panel">
        <div className="head"><div><h2>Forecast</h2><p>Each day is predicted from the earlier days, including earlier forecasts.</p></div></div>
        {state.error && <div className="callout bad">{state.error}</div>}
        {!state.result && !state.error && <p className="muted">Run a forecast to see the next days here.</p>}
        {state.result && (
          <>
            <div className="stats"><div className="stat"><div className="v">{state.result.total}</div><div className="l">Units forecast over {state.result.forecast.length} days</div></div></div>
            <div style={{ height: 260, marginTop: 12 }}>
              <ResponsiveContainer>
                <LineChart data={chart} margin={{ left: 0, right: 12 }}>
                  <CartesianGrid vertical={false} stroke="var(--line)" />
                  <XAxis dataKey="date" tickFormatter={(d) => d.slice(5)} minTickGap={24} stroke="var(--ink-3)" tick={{ fill: "var(--ink-2)", fontSize: 11 }} />
                  <YAxis stroke="var(--ink-3)" tick={{ fill: "var(--ink-2)" }} width={40} />
                  <Tooltip contentStyle={{ background: "var(--surface)", border: "1px solid var(--line)", borderRadius: 8 }} />
                  <Legend />
                  <Line isAnimationActive={false} dataKey="history" name="History" stroke="var(--ink-2)" strokeWidth={2} dot={false} connectNulls />
                  <Line isAnimationActive={false} dataKey="forecast" name="Forecast" stroke="var(--m-tabm)" strokeWidth={2.5} dot={{ r: 3 }} />
                </LineChart>
              </ResponsiveContainer>
            </div>
            <div className="scroll" style={{ maxHeight: 220, overflowY: "auto", marginTop: 8 }}>
              <table><thead><tr><th>Date</th><th>Forecast units</th></tr></thead>
                <tbody>{state.result.forecast.map((r) => <tr key={r.date}><td>{r.date}</td><td>{r.forecast.toFixed(2)}</td></tr>)}</tbody></table>
            </div>
          </>
        )}
      </section>
    </div>
  );
}

function Score({ usable }) {
  const [model, setModel] = useState("e2");
  const [file, setFile] = useState(null);
  const [state, setState] = useState({ busy: false, error: null, result: null });
  const [pick, setPick] = useState(0);
  useEffect(() => {
    if (usable.length && !usable.includes(model)) setModel(usable[0]);
  }, [usable.join()]); // eslint-disable-line

  async function run(e) {
    e.preventDefault();
    if (!file) return setState({ busy: false, error: "Choose a CSV file first.", result: null });
    setState({ busy: true, error: null, result: null });
    const body = new FormData();
    body.append("file", file);
    body.append("model", model);
    try {
      setState({ busy: false, error: null, result: await api("/api/score", { method: "POST", body }) });
      setPick(0);
    } catch (err) {
      setState({ busy: false, error: err.message, result: null });
    }
  }
  function download() {
    const r = state.result;
    const csv = "id,date,actual,pred\n" + r.rows.map((x) => [x.id, x.date, x.actual, x.pred].join(",")).join("\n");
    const a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob([csv], { type: "text/csv" }));
    a.download = "tabm_predictions.csv";
    a.click();
    URL.revokeObjectURL(a.href);
  }
  const r = state.result;
  const s = r?.series[pick];
  return (
    <div className="grid2">
      <section className="panel">
        <div className="head"><div><h2>Upload sales</h2><p>One row per item, store and day. Columns: date, item_id, store_id, sales, sell_price. Optional: snap, event_type_1, dept_id, cat_id, state_id.</p></div></div>
        <form onSubmit={run} className="form">
          <ModelSelect value={model} onChange={setModel} usable={usable} />
          <label className="field"><span>CSV file</span>
            <input type="file" accept=".csv,text/csv" onChange={(e) => setFile(e.target.files[0] || null)} /></label>
          <button className="primary" disabled={state.busy || !usable.length}>{state.busy ? "Running model…" : "Score file"}</button>
          <p className="muted small">Each series needs at least 36 consecutive days. The model predicts every day after that from its own history, and the result is compared with your actual sales.</p>
        </form>
      </section>
      <section className="panel">
        <div className="head"><div><h2>Results</h2><p>WAPE compares the model with a simple baseline that repeats last week's sales.</p></div></div>
        {state.error && <div className="callout bad">{state.error}</div>}
        {!r && !state.error && <p className="muted">Upload a file to see how the model does on it.</p>}
        {r && (
          <>
            <div className="stats">
              <div className="stat"><div className="v">{pct(r.wape)}</div><div className="l">TabM WAPE</div></div>
              <div className="stat"><div className="v">{pct(r.wape_lag7_baseline)}</div><div className="l">Repeat-last-week WAPE</div></div>
              <div className="stat"><div className="v">{r.n_series.toLocaleString()}</div><div className="l">Series, {r.n_rows.toLocaleString()} scored days</div></div>
            </div>
            <div className="seg" style={{ margin: "14px 0 8px" }} role="group" aria-label="Series">
              {r.series.map((x, i) => <button key={x.id} aria-pressed={pick === i} onClick={() => setPick(i)}>{x.id.replace(/_evaluation$/, "")}</button>)}
            </div>
            <p className="muted small">WAPE for this series: {pct(s.wape)}. Showing the last {s.points.length} days.</p>
            <div style={{ height: 240 }}>
              <ResponsiveContainer>
                <LineChart data={s.points} margin={{ left: 0, right: 12 }}>
                  <CartesianGrid vertical={false} stroke="var(--line)" />
                  <XAxis dataKey="date" tickFormatter={(d) => d.slice(5)} minTickGap={24} stroke="var(--ink-3)" tick={{ fill: "var(--ink-2)", fontSize: 11 }} />
                  <YAxis stroke="var(--ink-3)" tick={{ fill: "var(--ink-2)" }} width={40} />
                  <Tooltip contentStyle={{ background: "var(--surface)", border: "1px solid var(--line)", borderRadius: 8 }} />
                  <Legend />
                  <Line isAnimationActive={false} dataKey="actual" name="Actual" stroke="var(--ink-2)" strokeWidth={2} dot={false} />
                  <Line isAnimationActive={false} dataKey="pred" name="TabM" stroke="var(--m-tabm)" strokeWidth={2.5} dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>
            <button className="primary" onClick={download} style={{ marginTop: 12 }}>Download predictions CSV</button>
            {r.truncated && <p className="note">The download holds the first 20,000 rows only.</p>}
          </>
        )}
      </section>
    </div>
  );
}
