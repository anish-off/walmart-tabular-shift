# Shift study dashboard

React + Vite + Recharts front end, plus a FastAPI backend (`../backend/app.py`) that runs the saved TabM checkpoints.

## Run everything (from the repo root, Windows)

```bash
# one-time: venv with the packages the API needs
.venv\Scripts\python -m pip install torch --extra-index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python -m pip install fastapi "uvicorn[standard]" python-multipart httpx2 pandas numpy pyarrow pyyaml scikit-learn scipy
.venv\Scripts\python -m pip install -e . --no-deps

# terminal 1: model API on :8000
.venv\Scripts\python -m uvicorn app:app --app-dir backend --port 8000

# terminal 2: web app on :5173 (proxies /api to :8000)
cd frontend
npm install
npm run dev
```

## Pages

- **Study results**: charts built from `public/data.json`. Regenerate it with
  `.venv\Scripts\python scripts\export_frontend_data.py` after re-running experiments.
- **Run the model**: forecast one item from a pasted sales history, or upload a CSV
  (`date,item_id,store_id,sales,sell_price`) and score TabM against your actuals.

## Checkpoints

The API loads `results/tabm_{e1,e2,e3}_seed42.pt`. Set `SHIFT_MODEL_DIR` to load them from another folder.
`GET /api/health` reports which ones load. Only TabM has saved weights; the other four models are not runnable here.

## Known limits of inference on new data

- `item_id` has 3,049 label codes at training time and no catalogue is saved, so it is set to 0.
- Category, department, state and store codes use the M5 alphabetical codes when names follow M5 patterns.
- For a single item, `price_rank_cat` (rank of the price among peers) is set to a neutral 0.5.
