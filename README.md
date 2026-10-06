# tabular-shift-study

This project benchmarks five tabular ML models — XGBoost, LightGBM, TabPFN v2, TabM, and TabR — against the M5 Walmart sales forecasting dataset under three progressively harder distribution-shift conditions: E1 (random train/test split, baseline), E2 (temporal split, year-5 test period), and E3 (cold-start items with fewer than 30 nonzero training days). Performance is measured with WAPE and the Delta degradation metric (relative WAPE increase vs E1). A cold-start crossover curve (Figure 2) identifies the training-history threshold below which TabPFN v2 outperforms LightGBM, motivating the hybrid deployment scenario analyzed in the paper.

## Setup

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
pip install -e .
```

**Kaggle credentials**: place `~/.kaggle/kaggle.json` (or `%USERPROFILE%\.kaggle\kaggle.json` on Windows) before running `download_data.py`.

**TabPFN v2 weights**: Prior Labs requires a token on first use. Set the environment variable `TABPFN_TOKEN=<your-token>` before running any TabPFN experiment.

## Run order

```bash
python scripts/download_data.py
python scripts/build_features.py                  # one-time, ~42M rows

pytest -q                                         # unit tests

# smoke test first:
python scripts/run_experiment.py --model lightgbm --experiment e2 --sample 500

# full study (--skip-existing makes it safe to rerun after a crash; completed cells are skipped):
for m in xgboost lightgbm tabpfn tabm tabr; do
  for e in e1 e2 e3; do python scripts/run_experiment.py --model $m --experiment $e --skip-existing; done
done

python scripts/run_crossover.py
python scripts/make_report.py
```

> **Note:** all scripts expect to be run from the repository root so that
> `configs/*.yaml` relative paths resolve correctly.

### Optional: hyperparameter tuning (E2 validation window only — test never touched)

```bash
python scripts/tune.py --model lightgbm --trials 30 --sample 3000
```

## Expected runtimes (single GPU / CPU)

| Model    | Hardware | Approx. runtime     |
|----------|----------|---------------------|
| XGBoost  | CPU      | 50–70 min           |
| LightGBM | CPU      | 30–45 min           |
| TabPFN   | GPU      | chunked, ~1–2 h     |
| TabM     | GPU      | 2–3 h               |
| TabR     | GPU      | 8–10 h (slowest)    |

TabR may be dropped if behind schedule; the three remaining tree/TabPFN/TabM results are sufficient for the paper's main claims.

## Dashboard and model API

A web app in `frontend/` shows the study results and runs the saved TabM models on your own sales data.
A FastAPI backend in `backend/app.py` loads `results/tabm_{e1,e2,e3}_seed42.pt` for the "Run the model" page.

```bash
# one-time: install the API's packages into the venv
.venv\Scripts\python -m pip install torch --extra-index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python -m pip install fastapi "uvicorn[standard]" python-multipart httpx2 pyyaml
.venv\Scripts\python -m pip install -e . --no-deps

python scripts/export_frontend_data.py                                   # refresh frontend/public/data.json from results/
.venv\Scripts\python -m uvicorn app:app --app-dir backend --port 8000   # terminal 1: model API
cd frontend && npm install && npm run dev                                # terminal 2: http://localhost:5173
```

- **Study results**: WAPE by model and experiment, shift sensitivity, per-series error spread, category and state splits, a cold-start item table, and the Wilcoxon p-values.
- **Run the model**: forecast one item from a pasted sales history, or upload a CSV (`date,item_id,store_id,sales,sell_price`) and compare TabM with a repeat-last-week baseline.
- Only TabM has saved weights, so the other four models cannot be run from the app.
- If a checkpoint is incomplete or missing, the page names the file and asks you to copy it again.
- Item IDs are not mapped to training codes, so forecasts for unseen items rely on lags, rolling statistics, calendar and price. See `frontend/README.md` for details.

## Results layout

```
results/
  {model}_{experiment}.json        # WAPE mean/std, per-seed, runtime
  {model}_{experiment}_items.parquet  # per-item errors for Wilcoxon tests
  crossover.csv / crossover.json   # cold-start crossover curve data
  report/
    master_table.csv               # 5 models × E1/E2/E3 WAPE + Delta
    wilcoxon.csv                   # pairwise Wilcoxon p-values
    fig1_shift_sensitivity.png
    fig2_crossover.png
    report.md
```

## Experimental design

| Split | Train days    | Val days      | Test days     | Notes                                      |
|-------|---------------|---------------|---------------|--------------------------------------------|
| E1    | random 70%    | random 10%    | random 20%    | temporal leakage by design — Delta baseline |
| E2    | d < 1400      | 1400–1521     | 1522–1913     | pure temporal shift                        |
| E3    | d < 1400      | 1400–1521     | 1522–1913     | restricted to cold-start items (<30 nonzero training days) |
