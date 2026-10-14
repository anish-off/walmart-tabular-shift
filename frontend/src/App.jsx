import React, { useEffect, useMemo, useState } from "react";
import { API_BASE } from "./config.js";
import ForecastVisualizer from "./ForecastVisualizer.jsx";
import ScoreCsv from "./ScoreCsv.jsx";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

const MODELS = ["xgboost", "lightgbm", "tabpfn", "tabm", "tabr"];
const LABEL = { xgboost: "XGBoost", lightgbm: "LightGBM", tabpfn: "TabPFN v2", tabm: "TabM", tabr: "TabR" };
const EXPS = {
  e1: { name: "E1 Random split", desc: "Train and test drawn from the same days. Baseline distribution." },
  e2: { name: "E2 Temporal split", desc: "Train on earlier years, test on year 5." },
  e3: { name: "E3 Cold-start items", desc: "Items with under 30 nonzero training days." },
};
const col = (m) => `var(--m-${m})`;
const pct = (x) => (x == null ? "n/a" : (x * 100).toFixed(1) + "%");
const pct0 = (x) => Math.round(x * 100) + "%";
const f3 = (x) => (x == null ? "n/a" : x.toFixed(3));

function Tip({ active, payload, label, fmt = f3 }) {
  if (!active || !payload?.length) return null;
  return (
    <div className="tip">
      <b>{label}</b>
      {payload.map((p) => (
        <div key={p.dataKey ?? p.name}>
          <span className="chip" style={{ background: p.color || p.fill }} />
          {LABEL[p.name] || p.name}: {fmt(p.value)}
        </div>
      ))}
    </div>
  );
}

function Segmented({ value, onChange, options, label }) {
  return (
    <div className="seg" role="group" aria-label={label}>
      {options.map(([k, text]) => (
        <button key={k} aria-pressed={value === k} onClick={() => onChange(k)}>
          {text}
        </button>
      ))}
    </div>
  );
}

function Legend5() {
  return (
    <div className="legend">
      {MODELS.map((m) => (
        <span key={m}>
          <i className="dot" style={{ background: col(m) }} />
          {LABEL[m]}
        </span>
      ))}
    </div>
  );
}

export default function App() {
  const [data, setData] = useState(null);
  const [exp, setExp] = useState("e2");
  const [model, setModel] = useState("tabm");
  const [by, setBy] = useState("cat");

  // Primary page state: default to FORECAST VISUALIZER!
  const [page, setPage] = useState("forecast"); // 'forecast' | 'score' | 'results'

  // Bridge state from CSV to Forecast Visualizer
  const [forecastPrefill, setForecastPrefill] = useState(null);

  // Backend API health state
  const [health, setHealth] = useState(null);

  useEffect(() => {
    fetch("data.json")
      .then((r) => r.json())
      .then(setData)
      .catch((e) => console.warn("data.json not loaded yet:", e));

    fetch(`${API_BASE}/api/health`)
      .then((r) => (r.ok ? r.json() : Promise.reject(r)))
      .then(setHealth)
      .catch(() => setHealth({ unreachable: true }));
  }, []);

  const isApiOnline = health && !health.unreachable;
  const usable = isApiOnline ? Object.entries(health).filter(([, v]) => v.ok).map(([k]) => k) : [];

  const handleForwardProjectFromCsv = (seriesItem, targetModel) => {
    setForecastPrefill({
      model: targetModel,
      sales: seriesItem.sales_history.join(", "),
      end_date: seriesItem.last_date,
      price: seriesItem.last_price || 3.5,
      item_id: seriesItem.item_id || "ITEM_1",
      store_id: seriesItem.store_id || "STORE_1",
      horizon: 14,
    });
    setPage("forecast");
  };

  const view = useMemo(() => {
    if (!data) return null;
    const runs = data.runs.filter((r) => r.exp === exp).sort((a, b) => a.wape - b.wape);
    const e1 = Object.fromEntries(data.runs.filter((r) => r.exp === "e1").map((r) => [r.model, r.wape]));
    const lines = ["e1", "e2", "e3"].map((e) => ({
      exp: e.toUpperCase(),
      ...Object.fromEntries(data.runs.filter((r) => r.exp === e).map((r) => [r.model, r.wape])),
    }));
    const h = data.hist.find((x) => x.model === model && x.exp === exp);
    const hist = h?.counts?.map((c, i) => ({
      bin: i === h.counts.length - 1 ? "1.9+" : data.bins[i].toFixed(1),
      count: c,
    })) || [];
    const gnames = [...new Set(data.groups.filter((g) => g.by === by).map((g) => g.name))].sort();
    const groups = gnames.map((n) => ({
      name: n,
      ...Object.fromEntries(
        data.groups.filter((g) => g.by === by && g.exp === exp && g.name === n).map((g) => [g.model, g.wape])
      ),
    }));
    return { runs, e1, lines, hist, h, groups };
  }, [data, exp, model, by]);

  return (
    <div className="wrap">
      {/* ── TOP NAVIGATION BAR ── */}
      <nav className="top-nav">
        <div className="brand">
          <span className="brand-pip" />
          <span className="brand-title">WALMART DEMAND LAB</span>
          <span className="brand-sub">Tabular Shift Forecasting Engine</span>
        </div>

        <div className="seg" role="group" aria-label="Main Navigation">
          <button aria-pressed={page === "forecast"} onClick={() => setPage("forecast")}>
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
              <polyline points="22 12 18 12 15 21 9 3 6 12 2 12" />
            </svg>
            Forecast Visualizer
          </button>
          <button aria-pressed={page === "score"} onClick={() => setPage("score")}>
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
              <polyline points="17 8 12 3 7 8" />
              <line x1="12" y1="3" x2="12" y2="15" />
            </svg>
            Batch Ingestion
          </button>
          <button aria-pressed={page === "results"} onClick={() => setPage("results")}>
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M10 2v7.31M14 2v7.31M8.5 2h7M14 9.3a6.5 6.5 0 1 1-4 0" />
            </svg>
            Benchmark Study (E1–E3)
          </button>
        </div>

        <div className="status-pill" title="Forecasting telemetry online">
          <span className="status-dot online" />
          <span>System Online</span>
        </div>
      </nav>

      {/* ── PAGE 1: DEDICATED FORECAST VISUALIZER DASHBOARD (DEFAULT) ── */}
      {page === "forecast" && (
        <ForecastVisualizer usable={usable} health={health} prefill={forecastPrefill} />
      )}

      {/* ── PAGE 2: CSV INGESTION & SCORING ── */}
      {page === "score" && (
        <ScoreCsv usable={usable} health={health} onForwardProject={handleForwardProjectFromCsv} />
      )}

      {/* ── PAGE 3: BENCHMARK STUDY RESULTS ── */}
      {page === "results" && (
        <>
          {!data ? (
            <p style={{ padding: 40 }}>Loading benchmark dataset...</p>
          ) : (
            <>
              <header className="top" style={{ paddingTop: 20 }}>
                <h1>New items break every model, time barely does.</h1>
                <p>
                  Five tabular models forecast Walmart M5 sales under three kinds of distribution shift. Moving from a
                  random split to year-5 data adds about 3 to 5% to error. Moving to cold-start items adds about 45 to 50%.
                </p>
                <div className="controls">
                  <Segmented
                    label="Experiment"
                    value={exp}
                    onChange={setExp}
                    options={Object.entries(EXPS).map(([k, v]) => [k, v.name])}
                  />
                  <span className="desc">{EXPS[exp].desc}</span>
                </div>
              </header>

              <div className="stats" aria-live="polite">
                <div className="stat">
                  <div className="v">{LABEL[view.runs[0]?.model]}</div>
                  <div className="l">Lowest WAPE in {exp.toUpperCase()}, at {pct(view.runs[0]?.wape)}</div>
                </div>
                <div className="stat">
                  <div className="v">{pct(view.runs[view.runs.length - 1]?.wape)}</div>
                  <div className="l">Highest WAPE in {exp.toUpperCase()} ({LABEL[view.runs[view.runs.length - 1]?.model]})</div>
                </div>
                <div className="stat">
                  <div className="v">
                    +
                    {(
                      ((Math.min(...data.runs.filter((r) => r.exp === "e3").map((r) => r.wape)) /
                        Math.min(...data.runs.filter((r) => r.exp === "e1").map((r) => r.wape))) -
                        1) *
                      100
                    ).toFixed(0)}
                    %
                  </div>
                  <div className="l">Best-model error, cold-start vs random split</div>
                </div>
                <div className="stat">
                  <div className="v">{view.runs[0]?.n_series.toLocaleString()}</div>
                  <div className="l">Series in the {exp.toUpperCase()} test set</div>
                </div>
              </div>

              <section className="panel">
                <div className="head">
                  <div>
                    <h2>WAPE by model, {EXPS[exp].name}</h2>
                    <p>Weighted absolute percentage error across test series. Lower is better.</p>
                  </div>
                </div>
                <div style={{ height: 300 }}>
                  <ResponsiveContainer>
                    <BarChart data={view.runs} layout="vertical" margin={{ left: 8, right: 70 }}>
                      <CartesianGrid horizontal={false} stroke="var(--line)" />
                      <XAxis type="number" domain={[0, "dataMax"]} tickFormatter={pct} stroke="var(--ink-3)" tick={{ fill: "var(--ink-2)" }} />
                      <YAxis type="category" dataKey="model" tickFormatter={(m) => LABEL[m]} width={90} stroke="var(--ink-3)" tick={{ fill: "var(--ink)" }} />
                      <Tooltip
                        cursor={{ fill: "var(--bg)" }}
                        content={({ active, payload }) => {
                          if (!active || !payload?.length) return null;
                          const p = payload[0].payload;
                          return (
                            <div className="tip">
                              <b>{LABEL[p.model]}</b>
                              <div>WAPE {pct(p.wape)}</div>
                              {exp !== "e1" && <div>Increase vs E1: +{(((p.wape / view.e1[p.model]) - 1) * 100).toFixed(1)}%</div>}
                              <div>Train rows {p.n_train.toLocaleString()}</div>
                              <div>Runtime {p.runtime_min} min</div>
                            </div>
                          );
                        }}
                      />
                      <Bar
                        dataKey="wape"
                        radius={[0, 4, 4, 0]}
                        barSize={30}
                        label={{ position: "right", formatter: pct, fill: "var(--ink)", fontSize: 13 }}
                      >
                        {view.runs.map((r) => (
                          <Cell key={r.model} fill={col(r.model)} />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </section>

              <section className="panel">
                <div className="head">
                  <div>
                    <h2>Sensitivity to shift</h2>
                    <p>Each model's WAPE as test condition becomes harsher (E1 &rarr; E2 &rarr; E3).</p>
                  </div>
                </div>
                <Legend5 />
                <div style={{ height: 320 }}>
                  <ResponsiveContainer>
                    <LineChart data={view.lines} margin={{ right: 20, left: 0 }}>
                      <CartesianGrid vertical={false} stroke="var(--line)" />
                      <XAxis dataKey="exp" stroke="var(--ink-3)" tick={{ fill: "var(--ink)" }} />
                      <YAxis domain={[0.55, 1.05]} tickFormatter={pct0} stroke="var(--ink-3)" tick={{ fill: "var(--ink-2)" }} width={60} />
                      <Tooltip content={<Tip fmt={pct} />} />
                      {MODELS.map((m) => (
                        <Line
                          key={m}
                          type="linear"
                          dataKey={m}
                          name={m}
                          stroke={col(m)}
                          strokeWidth={2}
                          dot={{ r: 4, stroke: "var(--surface)", strokeWidth: 2, fill: col(m) }}
                          activeDot={{ r: 6 }}
                        />
                      ))}
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </section>

              <div className="grid2">
                <section className="panel">
                  <div className="head">
                    <div>
                      <h2>Per-series error spread</h2>
                      <p>WAPE distribution across evaluated items.</p>
                    </div>
                    <Segmented label="Model" value={model} onChange={setModel} options={MODELS.map((m) => [m, LABEL[m]])} />
                  </div>
                  <div style={{ height: 260 }}>
                    <ResponsiveContainer>
                      <BarChart data={view.hist} margin={{ left: 0, right: 8 }}>
                        <CartesianGrid vertical={false} stroke="var(--line)" />
                        <XAxis dataKey="bin" stroke="var(--ink-3)" tick={{ fill: "var(--ink-2)", fontSize: 11 }} interval={1} />
                        <YAxis stroke="var(--ink-3)" tick={{ fill: "var(--ink-2)" }} width={60} />
                        <Tooltip content={<Tip fmt={(v) => v.toLocaleString() + " series"} />} cursor={{ fill: "var(--bg)" }} />
                        <Bar dataKey="count" name="Series" fill={col(model)} radius={[3, 3, 0, 0]} />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </section>

                <section className="panel">
                  <div className="head">
                    <div>
                      <h2>Where the error sits</h2>
                      <p>WAPE split by {by === "cat" ? "product category" : "state"}.</p>
                    </div>
                    <Segmented label="Group by" value={by} onChange={setBy} options={[["cat", "Category"], ["state", "State"]]} />
                  </div>
                  <Legend5 />
                  <div style={{ height: 240 }}>
                    <ResponsiveContainer>
                      <BarChart data={view.groups} margin={{ left: 0, right: 8 }}>
                        <CartesianGrid vertical={false} stroke="var(--line)" />
                        <XAxis dataKey="name" stroke="var(--ink-3)" tick={{ fill: "var(--ink)", fontSize: 12 }} />
                        <YAxis tickFormatter={pct0} stroke="var(--ink-3)" tick={{ fill: "var(--ink-2)" }} width={60} />
                        <Tooltip content={<Tip fmt={pct} />} cursor={{ fill: "var(--bg)" }} />
                        {MODELS.map((m) => (
                          <Bar key={m} dataKey={m} name={m} fill={col(m)} radius={[3, 3, 0, 0]} />
                        ))}
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </section>
              </div>

              <ColdStart rows={data.cold} />
              <Significance rows={data.wilcoxon.filter((r) => r.experiment === exp)} exp={exp} />
            </>
          )}
        </>
      )}

      <footer style={{ marginTop: 40, borderTop: "1px solid var(--line)", paddingTop: 16 }}>
        Walmart M5 Tabular Shift Study &bull; TabM Deep Learning Inference Engine
      </footer>
    </div>
  );
}

function ColdStart({ rows }) {
  const [q, setQ] = useState("");
  const [sort, setSort] = useState({ key: "actual", dir: -1 });
  const list = useMemo(() => {
    const f = rows.filter((r) => r.id.toLowerCase().includes(q.toLowerCase()));
    const val = (r) => (typeof r[sort.key] === "string" ? r[sort.key] : r[sort.key] ?? -1);
    return f.sort((a, b) => (val(a) > val(b) ? 1 : val(a) < val(b) ? -1 : 0) * sort.dir).slice(0, 25);
  }, [rows, q, sort]);
  const th = (key, text) => (
    <th key={key} aria-sort={sort.key === key ? (sort.dir > 0 ? "ascending" : "descending") : "none"}>
      <button onClick={() => setSort((s) => ({ key, dir: s.key === key ? -s.dir : -1 }))}>
        {text}
        {sort.key === key ? (sort.dir > 0 ? " ▲" : " ▼") : ""}
      </button>
    </th>
  );
  return (
    <section className="panel">
      <div className="head">
        <div>
          <h2>Cold-start item explorer</h2>
          <p>{rows.length.toLocaleString()} E3 series. Per-series WAPE for each model. Showing top 25.</p>
        </div>
        <input
          type="search"
          placeholder="Filter by item id, e.g. FOODS_3"
          value={q}
          onChange={(e) => setQ(e.target.value)}
          aria-label="Filter items"
        />
      </div>
      <div className="scroll">
        <table>
          <thead>
            <tr>
              {th("id", "Item")}
              {th("actual", "Units sold")}
              {MODELS.map((m) => th(m, LABEL[m]))}
            </tr>
          </thead>
          <tbody>
            {list.map((r) => {
              const min = Math.min(...MODELS.map((m) => r[m] ?? Infinity));
              return (
                <tr key={r.id}>
                  <td>{r.id}</td>
                  <td>{r.actual.toLocaleString()}</td>
                  {MODELS.map((m) => (
                    <td key={m} className={r[m] === min ? "best" : ""}>
                      {pct(r[m])}
                    </td>
                  ))}
                </tr>
              );
            })}
            {!list.length && <tr><td colSpan={7}>No items match "{q}".</td></tr>}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function Significance({ rows, exp }) {
  const get = (a, b) => rows.find((r) => (r.model_a === a && r.model_b === b) || (r.model_a === b && r.model_b === a));
  const fmtP = (p) => (p === 0 ? "<1e-300" : p < 0.001 ? p.toExponential(0) : p.toFixed(3));
  return (
    <section className="panel">
      <div className="head">
        <div>
          <h2>Are the differences real?</h2>
          <p>Wilcoxon signed-rank p-values on per-series WAPE for {exp.toUpperCase()}.</p>
        </div>
      </div>
      <div className="scroll">
        <table>
          <thead>
            <tr>
              <th>Model</th>
              {MODELS.map((m) => (
                <th key={m}>{LABEL[m]}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {MODELS.map((a) => (
              <tr key={a}>
                <td>
                  <span className="chip" style={{ background: col(a) }} />
                  {LABEL[a]}
                </td>
                {MODELS.map((b) => {
                  const r = a === b ? null : get(a, b);
                  return <td key={b} className={r && r.p_value < 0.05 ? "" : "muted"}>{a === b ? "–" : r ? fmtP(r.p_value) : "n/a"}</td>;
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}
