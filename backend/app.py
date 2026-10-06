"""Inference API for the shift-study TabM checkpoints.

  .venv\\Scripts\\python -m uvicorn app:app --app-dir backend --port 8000

Endpoints
  GET  /api/health      which checkpoints (e1/e2/e3) load
  POST /api/forecast    forecast the next N days for one series (recursive)
  POST /api/score       score an uploaded multi-series CSV against its own actuals

Environment
  SHIFT_MODEL_DIR   directory holding tabm_{e1,e2,e3}_seed42.pt (default: ./results)
"""
import io
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from shift_study.features import FEATURES, add_features
from shift_study.models.tabm import TabMModel

MODEL_DIR = Path(os.environ.get("SHIFT_MODEL_DIR", "results"))
EXPS = ("e1", "e2", "e3")
MIN_HISTORY = 36          # lag_35 plus the target day
MAX_ROWS = 300_000

app = FastAPI(title="Shift study inference")

# Label codes the M5 training run produced (alphabetical category codes).
# item_id has 3,049 values and no catalogue ships with the repo, so it maps to 0.
CAT = {"FOODS": 0, "HOBBIES": 1, "HOUSEHOLD": 2}
DEPT = {"FOODS_1": 0, "FOODS_2": 1, "FOODS_3": 2, "HOBBIES_1": 3, "HOBBIES_2": 4,
        "HOUSEHOLD_1": 5, "HOUSEHOLD_2": 6}
STATE = {"CA": 0, "TX": 1, "WI": 2}
STORE = {"CA_1": 0, "CA_2": 1, "CA_3": 2, "CA_4": 3, "TX_1": 4, "TX_2": 5, "TX_3": 6,
         "WI_1": 7, "WI_2": 8, "WI_3": 9}
EVENT = {"Cultural": 0, "National": 1, "Religious": 2, "Sporting": 3}
NO_EVENT = 4


# ── model loading ───────────────────────────────────────────────────────────

def _ckpt_path(exp: str) -> Path:
    return MODEL_DIR / f"tabm_{exp}_seed42.pt"


@lru_cache(maxsize=None)
def _load(exp: str) -> TabMModel:
    return TabMModel.load(_ckpt_path(exp))


def get_model(exp: str) -> TabMModel:
    if exp not in EXPS:
        raise HTTPException(400, f"Unknown experiment '{exp}'. Use one of {', '.join(EXPS)}.")
    p = _ckpt_path(exp)
    if not p.exists():
        raise HTTPException(503, f"Checkpoint {p} not found.")
    try:
        return _load(exp)
    except Exception as e:  # corrupt or truncated file
        raise HTTPException(
            503,
            f"Checkpoint {p.name} could not be read ({type(e).__name__}). "
            "The file is likely incomplete; copy it again from the training machine.",
        )


@app.get("/api/health")
def health():
    out = {}
    for e in EXPS:
        p = _ckpt_path(e)
        try:
            get_model(e)
            out[e] = {"ok": True, "bytes": p.stat().st_size}
        except HTTPException as ex:
            out[e] = {"ok": False, "error": ex.detail, "bytes": p.stat().st_size if p.exists() else 0}
    return out


# ── feature helpers ─────────────────────────────────────────────────────────

def _split_names(item_id: str, store_id: str):
    """Derive M5-style dept/cat/state from ids when they follow M5 naming."""
    parts = str(item_id).split("_")
    dept = "_".join(parts[:2]) if len(parts) >= 3 else "UNKNOWN"
    cat = parts[0] if len(parts) >= 3 else "UNKNOWN"
    st = str(store_id).split("_")[0] if "_" in str(store_id) else "UNKNOWN"
    return dept, cat, st


def _calendar(df: pd.DataFrame) -> pd.DataFrame:
    d = pd.to_datetime(df["date"])
    df["date"] = d
    df["wday"] = (((d.dt.dayofweek + 2) % 7) + 1).astype("int8")   # M5: 1=Sat, 2=Sun, 3=Mon
    df["month"] = d.dt.month.astype("int8")
    iso = d.dt.isocalendar()
    df["wm_yr_wk"] = (iso["year"].astype(int) * 100 + iso["week"].astype(int)).astype("int32")
    return df


def _override_codes(feats: pd.DataFrame) -> pd.DataFrame:
    """Replace dataset-local label codes with the codes seen at training time."""
    feats["cat_id_enc"] = feats["cat_id"].astype(str).map(CAT).fillna(0).astype("int16")
    feats["dept_id_enc"] = feats["dept_id"].astype(str).map(DEPT).fillna(0).astype("int16")
    feats["state_id_enc"] = feats["state_id"].astype(str).map(STATE).fillna(0).astype("int16")
    feats["store_id_enc"] = feats["store_id"].astype(str).map(STORE).fillna(0).astype("int16")
    feats["event_type_1_enc"] = feats["event_type_1"].astype(str).map(EVENT).fillna(NO_EVENT).astype("int16")
    feats["item_id_enc"] = np.int16(0)
    return feats


def _wape(actual, pred):
    actual = np.asarray(actual, dtype=float)
    pred = np.asarray(pred, dtype=float)
    tot = actual.sum()
    return float(np.abs(actual - pred).sum() / tot) if tot > 0 else None


# ── single-series forecast ──────────────────────────────────────────────────

class ForecastReq(BaseModel):
    model: str = "e2"
    sales: list[float] = Field(..., description="Daily unit sales, oldest first, at least 36 values")
    end_date: str = Field(..., description="Date of the last value, YYYY-MM-DD")
    price: float | list[float] = Field(..., description="One price, or one per history day")
    future_price: float | None = Field(None, description="Price for forecast days (default: last price)")
    horizon: int = Field(14, ge=1, le=56)
    item_id: str = "ITEM_1"
    store_id: str = "STORE_1"
    cat_id: str | None = None
    dept_id: str | None = None
    state_id: str | None = None
    snap_days: list[str] = Field(default_factory=list, description="Forecast dates (YYYY-MM-DD) with SNAP benefits")
    event_days: dict[str, str] = Field(default_factory=dict, description="date -> event type, e.g. National")


def _frame(req: ForecastReq, n: int) -> pd.DataFrame:
    dates = pd.date_range(end=req.end_date, periods=n, freq="D")
    price = np.full(n, req.price, dtype="float32") if isinstance(req.price, (int, float)) \
        else np.asarray(req.price, dtype="float32")
    if len(price) != n:
        raise HTTPException(422, f"price has {len(price)} values but sales has {n}.")
    dept, cat, st = _split_names(req.item_id, req.store_id)
    df = pd.DataFrame({
        "id": "series", "item_id": req.item_id, "store_id": req.store_id,
        "dept_id": req.dept_id or dept, "cat_id": req.cat_id or cat, "state_id": req.state_id or st,
        "date": dates, "sales": np.asarray(req.sales, dtype="float32"), "sell_price": price,
        "event_type_1": "", "snap": 0,
    })
    df["d_int"] = np.arange(1, n + 1, dtype="int32")
    return df


@app.post("/api/forecast")
def forecast(req: ForecastReq):
    model = get_model(req.model)
    n = len(req.sales)
    if n < MIN_HISTORY:
        raise HTTPException(422, f"Need at least {MIN_HISTORY} days of history; got {n}.")
    if any(s < 0 for s in req.sales):
        raise HTTPException(422, "Sales values cannot be negative.")
    try:
        df = _calendar(_frame(req, n))
    except ValueError as e:
        raise HTTPException(422, f"Invalid end_date or price: {e}")
    fprice = req.future_price if req.future_price is not None else float(df["sell_price"].iloc[-1])
    snap = set(req.snap_days)
    events = req.event_days
    out = []
    for _ in range(req.horizon):
        nd = df["date"].iloc[-1] + pd.Timedelta(days=1)
        key = nd.strftime("%Y-%m-%d")
        row = df.iloc[[-1]].copy()
        row["date"] = nd
        row["d_int"] = int(df["d_int"].iloc[-1]) + 1
        row["sales"] = np.float32(0.0)   # placeholder: lag and rolling features never read the target day
        row["sell_price"] = np.float32(fprice)
        row["snap"] = np.int8(1 if key in snap else 0)
        row["event_type_1"] = events.get(key, "")
        row = _calendar(row)
        df = pd.concat([df, row], ignore_index=True)
        feats = add_features(df.tail(120).copy())
        feats["price_rank_cat"] = np.float32(0.5)   # single series has no peers to rank against
        feats = _override_codes(feats)
        last = feats[feats["d_int"] == df["d_int"].iloc[-1]]
        if last.empty:
            raise HTTPException(500, "Feature builder dropped the forecast row.")
        pred = max(0.0, float(model.predict(last[FEATURES])[0]))
        df.loc[df.index[-1], "sales"] = np.float32(pred)
        out.append({"date": key, "forecast": round(pred, 3)})
    return {
        "model": req.model,
        "forecast": out,
        "total": round(sum(r["forecast"] for r in out), 2),
        "history_tail": [
            {"date": d.strftime("%Y-%m-%d"), "sales": float(s)}
            for d, s in zip(pd.date_range(end=req.end_date, periods=n)[-60:], req.sales[-60:])
        ],
    }


# ── CSV scoring ─────────────────────────────────────────────────────────────

@app.post("/api/score")
async def score(file: UploadFile = File(...), model: str = Form("e2")):
    m = get_model(model)
    raw = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(raw), parse_dates=["date"])
    except Exception as e:
        raise HTTPException(422, f"Could not read the CSV: {e}")
    need = {"date", "item_id", "store_id", "sales", "sell_price"}
    if need - set(df.columns):
        raise HTTPException(422, f"CSV is missing columns: {', '.join(sorted(need - set(df.columns)))}. "
                                 "Required: date, item_id, store_id, sales, sell_price.")
    if len(df) > MAX_ROWS:
        raise HTTPException(413, f"CSV has {len(df):,} rows; the limit is {MAX_ROWS:,}.")
    df["id"] = df["item_id"].astype(str) + "_" + df["store_id"].astype(str)
    df = df.sort_values(["id", "date"]).reset_index(drop=True)
    for c, default in [("event_type_1", ""), ("snap", 0)]:
        if c not in df.columns:
            df[c] = default
    names = df.apply(lambda r: _split_names(r["item_id"], r["store_id"]), axis=1, result_type="expand")
    for i, c in enumerate(["dept_id", "cat_id", "state_id"]):
        if c not in df.columns:
            df[c] = names[i]
    df["event_type_1"] = df["event_type_1"].fillna("")
    df["sales"] = df["sales"].astype("float32")
    df["sell_price"] = df["sell_price"].astype("float32")
    df["snap"] = df["snap"].fillna(0).astype("int8")
    df["d_int"] = df.groupby("id")["date"].transform(lambda s: (s - s.min()).dt.days + 1).astype("int32")
    if df.duplicated(["id", "d_int"]).any():
        raise HTTPException(422, "Each item/store/date must appear once.")
    df = _calendar(df)
    feats = add_features(df)
    if feats.empty:
        raise HTTPException(422, f"No usable rows: each series needs at least {MIN_HISTORY} consecutive days.")
    feats = _override_codes(feats)
    feats["pred"] = np.clip(m.predict(feats[FEATURES]), 0, None)
    top = feats.groupby("id")["sales"].sum().nlargest(6).index
    series = []
    for sid in top:
        g = feats[feats["id"] == sid].tail(120)
        series.append({
            "id": sid,
            "wape": _wape(feats.loc[feats["id"] == sid, "sales"], feats.loc[feats["id"] == sid, "pred"]),
            "points": [{"date": d.strftime("%Y-%m-%d"), "actual": float(a), "pred": round(float(p), 3)}
                       for d, a, p in zip(g["date"], g["sales"], g["pred"])],
        })
    cap = 20_000
    rows = feats[["id", "date", "sales", "pred"]].head(cap)
    return {
        "model": model,
        "n_rows": int(len(feats)),
        "n_series": int(feats["id"].nunique()),
        "wape": _wape(feats["sales"], feats["pred"]),
        "wape_lag7_baseline": _wape(feats["sales"], feats["lag_7"]),
        "series": series,
        "rows": [{"id": r.id, "date": r.date.strftime("%Y-%m-%d"), "actual": float(r.sales),
                  "pred": round(float(r.pred), 3)} for r in rows.itertuples()],
        "truncated": bool(len(feats) > cap),
    }
