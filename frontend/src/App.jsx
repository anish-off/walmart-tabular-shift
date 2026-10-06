import React, { useEffect, useMemo, useState } from "react";
import RunModel from "./RunModel.jsx";
import {
  Bar, BarChart, CartesianGrid, Cell, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";

const MODELS = ["xgboost", "lightgbm", "tabpfn", "tabm", "tabr"];
const LABEL = { xgboost: "XGBoost", lightgbm: "LightGBM", tabpfn: "TabPFN v2", tabm: "TabM", tabr: "TabR" };
const EXPS = {
  e1: { name: "E1 Random split", desc: "Train and test drawn from the same days. The baseline." },
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
        <button key={k} aria-pressed={value === k} onClick={() => onChange(k)}>{text}</button>
      ))}
    </div>
  );
}

function Legend5() {
  return (
    <div className="legend">
      {MODELS.map((m) => (<span key={m}><i className="dot" style={{ background: col(m) }} />{LABEL[m]}</span>))}
    </div>
  );
}

export default function App() {
  const [data, setData] = useState(null);
  const [exp, setExp] = useState("e2");
  const [model, setModel] = useState("tabm");
  const [by, setBy] = useState("cat");
  const [page, setPage] = useState("results");
  useEffect(() => { fetch("data.json").then((r) => r.json()).then(setData); }, []);

  const view = useMemo(() => {
    if (!data) return null;
    const runs = data.runs.filter((r) => r.exp === exp).sort((a, b) => a.wape - b.wape);
    const e1 = Object.fromEntries(data.runs.filter((r) => r.exp === "e1").map((r) => [r.model, r.wape]));
    const lines = ["e1", "e2", "e3"].map((e) => ({
      exp: e.toUpperCase(),
      ...Object.fromEntries(data.runs.filter((r) => r.exp === e).map((r) => [r.model, r.wape])),
    }));
    const h = data.hist.find((x) => x.model === model && x.exp === exp);
    const hist = h.counts.map((c, i) => ({
      bin: i === h.counts.length - 1 ? "1.9+" : data.bins[i].toFixed(1), count: c,
    }));
    const gnames = [...new Set(data.groups.filter((g) => g.by === by).map((g) => g.name))].sort();
    const groups = gnames.map((n) => ({
      name: n,
      ...Object.fromEntries(
        data.groups.filter((g) => g.by === by && g.exp === exp && g.name === n).map((g) => [g.model, g.wape]),
      ),
    }));
    return { runs, e1, lines, hist, h, groups };
  }, [data, exp, model, by]);

  if (!data) return <div className="wrap"><p style={{ padding: 56 }}>Loading results…</p></div>;
  const best = view.runs[0];
  const worst = view.runs[view.runs.length - 1];
  const bestE1 = Math.min(...data.runs.filter((r) => r.exp === "e1").map((r) => r.wape));
  const bestE3 = Math.min(...data.runs.filter((r) => r.exp === "e3").map((r) => r.wape));

  const tabs = (
    <nav className="tabs">
      <div className="seg" role="group" aria-label="Page">
        <button aria-pressed={page === "results"} onClick={() => setPage("results")}>Study results</button>
        <button aria-pressed={page === "run"} onClick={() => setPage("run")}>Run the model</button>
      </div>
    </nav>
  );
  if (page === "run") return <div className="wrap">{tabs}<RunModel /></div>;

  return (
    <div className="wrap">
      {tabs}
      <header className="top">
        <h1>New items break every model, time barely does.</h1>
        <p>
          Five tabular models forecast Walmart M5 sales under three kinds of distribution shift. Moving from a random
          split to year-5 data adds about 3 to 5% to error. Moving to cold-start items adds about 45 to 50%.
        </p>
        <div className="controls">
          <Segmented label="Experiment" value={exp} onChange={setExp}
            options={Object.entries(EXPS).map(([k, v]) => [k, v.name])} />
          <span className="desc">{EXPS[exp].desc}</span>
        </div>
      </header>

      <div className="stats" aria-live="polite">
        <div className="stat"><div className="v">{LABEL[best.model]}</div><div className="l">Lowest WAPE in {exp.toUpperCase()}, at {pct(best.wape)}</div></div>
        <div className="stat"><div className="v">{pct(worst.wape)}</div><div className="l">Highest WAPE in {exp.toUpperCase()} ({LABEL[worst.model]})</div></div>
        <div className="stat"><div className="v">+{(((bestE3 / bestE1) - 1) * 100).toFixed(0)}%</div><div className="l">Best-model error, cold-start vs random split</div></div>
        <div className="stat"><div className="v">{best.n_series.toLocaleString()}</div><div className="l">Series in the {exp.toUpperCase()} test set</div></div>
      </div>

      <section className="panel">
        <div className="head">
          <div>
            <h2>WAPE by model, {EXPS[exp].name}</h2>
            <p>Weighted absolute percentage error across all test series. Lower is better. Hover a bar to see the increase over that model's E1 score.</p>
          </div>
        </div>
        <div style={{ height: 300 }}>
          <ResponsiveContainer>
            <BarChart data={view.runs} layout="vertical" margin={{ left: 8, right: 70 }}>
              <CartesianGrid horizontal={false} stroke="var(--line)" />
              <XAxis type="number" domain={[0, "dataMax"]} tickFormatter={pct} stroke="var(--ink-3)" tick={{ fill: "var(--ink-2)" }} />
              <YAxis type="category" dataKey="model" tickFormatter={(m) => LABEL[m]} width={90} stroke="var(--ink-3)" tick={{ fill: "var(--ink)" }} />
              <Tooltip cursor={{ fill: "var(--bg)" }} content={({ active, payload }) => {
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
              }} />
              <Bar dataKey="wape" radius={[0, 4, 4, 0]} barSize={30}
                label={{ position: "right", formatter: pct, fill: "var(--ink)", fontSize: 13 }}>
                {view.runs.map((r) => <Cell key={r.model} fill={col(r.model)} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </section>

      <section className="panel">
        <div className="head">
          <div>
            <h2>Sensitivity to shift</h2>
            <p>Each model's WAPE as the test condition gets harder. The jump to E3 dwarfs the move to E2, and the ranking barely changes.</p>
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
                <Line key={m} type="linear" dataKey={m} name={m} stroke={col(m)} strokeWidth={2}
                  dot={{ r: 4, stroke: "var(--surface)", strokeWidth: 2, fill: col(m) }} activeDot={{ r: 6 }} />
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
              <p>How many series fall in each WAPE band. A long right tail means a few series dominate the miss.</p>
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
          <p className="muted" style={{ fontSize: ".9rem" }}>
            Median per-series WAPE for {LABEL[model]} in {exp.toUpperCase()}: <b>{pct(view.h.median)}</b>. Values above 2.0 are grouped in the last bin.
          </p>
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
                {MODELS.map((m) => <Bar key={m} dataKey={m} name={m} fill={col(m)} radius={[3, 3, 0, 0]} />)}
              </BarChart>
            </ResponsiveContainer>
          </div>
        </section>
      </div>

      <ColdStart rows={data.cold} />
      <Significance rows={data.wilcoxon.filter((r) => r.experiment === exp)} exp={exp} />

      <footer>
        Source: results/*.json and per-series parquet files from the shift study. Regenerate the data with
        <code> python scripts/export_frontend_data.py</code>.
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
        {text}{sort.key === key ? (sort.dir > 0 ? " ▲" : " ▼") : ""}
      </button>
    </th>
  );
  return (
    <section className="panel">
      <div className="head">
        <div>
          <h2>Cold-start item explorer</h2>
          <p>{rows.length.toLocaleString()} E3 series. Per-series WAPE for each model; the lowest in each row is bold. Click a header to sort. Showing the top 25.</p>
        </div>
        <input type="search" placeholder="Filter by item id, e.g. FOODS_3" value={q}
          onChange={(e) => setQ(e.target.value)} aria-label="Filter items" />
      </div>
      <div className="scroll">
        <table>
          <thead><tr>{th("id", "Item")}{th("actual", "Units sold")}{MODELS.map((m) => th(m, LABEL[m]))}</tr></thead>
          <tbody>
            {list.map((r) => {
              const min = Math.min(...MODELS.map((m) => r[m] ?? Infinity));
              return (
                <tr key={r.id}>
                  <td>{r.id}</td><td>{r.actual.toLocaleString()}</td>
                  {MODELS.map((m) => <td key={m} className={r[m] === min ? "best" : ""}>{pct(r[m])}</td>)}
                </tr>
              );
            })}
            {!list.length && <tr><td colSpan={7}>No items match "{q}". Try a category such as HOBBIES_1.</td></tr>}
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
          <p>Wilcoxon signed-rank p-values on per-series WAPE for {exp.toUpperCase()}. Small values mean the two models differ on the same series.</p>
        </div>
      </div>
      <div className="scroll">
        <table>
          <thead><tr><th>Model</th>{MODELS.map((m) => <th key={m}>{LABEL[m]}</th>)}</tr></thead>
          <tbody>
            {MODELS.map((a) => (
              <tr key={a}>
                <td><span className="chip" style={{ background: col(a) }} />{LABEL[a]}</td>
                {MODELS.map((b) => {
                  const r = a === b ? null : get(a, b);
                  return <td key={b} className={r && r.p_value < 0.05 ? "" : "muted"}>{a === b ? "–" : r ? fmtP(r.p_value) : "n/a"}</td>;
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="note">These p-values are uncorrected across 30 comparisons. Apply Holm or Bonferroni before claiming significance.</p>
    </section>
  );
}
