import React, { useEffect, useMemo, useRef, useState } from "react";
import { CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

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

async function api(path, opts) {
  let res;
  try {
    res = await fetch(path, opts);
  } catch {
    throw new Error("Cannot reach model API. Start backend with: .venv\\Scripts\\python -m uvicorn app:app --app-dir backend --port 8000");
  }
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(typeof body.detail === "string" ? body.detail : "The request failed (" + res.status + ").");
  return body;
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

export default function ScoreCsv({ usable, health, onForwardProject }) {
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
    <div className="grid2" style={{ marginTop: 8 }}>
      {/* LEFT INGESTION PANEL */}
      <section className="panel" style={{ marginTop: 0 }}>
        <div className="head">
          <div>
            <h2>CSV Batch Ingestion & Verification</h2>
            <p>Ingest multi-series retail sales CSVs. Each series requires at least 36 consecutive days for lag_35 features.</p>
          </div>
        </div>

        <form onSubmit={runScoring} className="form">
          <label className="field">
            <span>Model Checkpoint</span>
            <select value={model} onChange={(e) => setModel(e.target.value)}>
              {["e2", "e1", "e3", "test"].map((k) => {
                const isOk = usable.includes(k);
                const isConf = health && health[k];
                return (
                  <option key={k} value={k} disabled={!isOk}>
                    TabM ({EXP_LABEL[k] || k}) {isOk ? "— [Ready]" : isConf ? "— [Incomplete]" : "— [Unavailable]"}
                  </option>
                );
              })}
            </select>
          </label>

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
                      {(file.size / 1024).toFixed(1)} KB &bull; {inspect?.rowCount ? `${inspect.rowCount.toLocaleString()} records` : "Inspecting..."}
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
      <section className="panel" style={{ marginTop: 0 }}>
        <div className="head">
          <div>
            <h2>Model Performance & Residuals</h2>
            <p>Weighted Absolute Percentage Error (WAPE) compared against repeat-last-week (lag_7) seasonal baseline.</p>
          </div>
        </div>

        {state.error && <div className="callout bad">{state.error}</div>}

        {!result && !state.error && !state.busy && (
          <div style={{ padding: "60px 10px", textAlign: "center", color: "var(--ink-2)" }}>
            <p>Upload a CSV dataset and score it to generate residual diagnostics and forward forecast bridges.</p>
          </div>
        )}

        {state.busy && (
          <div style={{ padding: "60px 10px", textAlign: "center" }}>
            <div className="spinner" style={{ width: 32, height: 32, borderWidth: 3, margin: "0 auto 16px" }} />
            <p className="muted">Running feature extraction & PyTorch batch inference...</p>
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
                <div className="label">Scored Timesteps</div>
                <div className="val">{result.n_rows.toLocaleString()}</div>
                <div className="sub">{result.n_series} unique series</div>
              </div>
            </div>

            {/* SERIES SWITCHER & 1-CLICK FORWARD FORECAST BUTTON */}
            {result.series?.length > 0 && (
              <>
                <div style={{ margin: "18px 0 8px", display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: 8 }}>
                  <span className="small muted">Evaluated Series:</span>
                  {activeSeries?.sales_history && (
                    <button
                      type="button"
                      className="btn-primary"
                      style={{ padding: "6px 14px", fontSize: "0.82rem" }}
                      onClick={() => onForwardProject(activeSeries, model)}
                    >
                      Project Forecast in Visualizer ({activeSeries.id}) &rarr;
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
                <span className="small muted">Residual Record Explorer ({filteredRows.length} rows)</span>
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
