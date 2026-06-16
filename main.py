# %% [markdown]
# # M5 Tabular Shift Study — Local Runner
# 
# **Inputs attached:**
# - Project: `/home/23alr106/code/walmart-tabular-shift/`
# - M5 data: `/home/23alr106/code/walmart-tabular-shift/data/raw/`
# 
# | Cell | What it does | Time |
# |------|-------------|------|
# | 1 | Install deps + add `shift_study` to path | ~2 min |
# | 2 | Verify data & GPU | <1 min |
# | 3 | Build `features.parquet` | ~15 min |
# | 4 | Smoke test (LightGBM E2, 500 series) | ~2 min |
# | 5 | Clean smoke artefacts | instant |
# | 6 | **Session 1:** XGBoost / LightGBM / TabM | ~4–6 h |
# | 7 | **Session 2:** TabR only (*Save & Run All*) | ~8–10 h |
# | 8 | Crossover curve + report | ~1 h |
# | 9 | Display results inline | instant |
# | 10 | List all output files | instant |
# 

# %% [markdown]
# ## Cell 1 — Install Dependencies

""" # %%
import subprocess, sys, os

PROJ = "/home/23alr106/code/walmart-tabular-shift"

# 1) Install third-party requirements (read-only path is fine for -r)
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q",
    "-r", f"{PROJ}/requirements.txt",
])

# 2) Make shift_study importable — no build needed, just add src/ to path
src = f"{PROJ}/src"
if src not in sys.path:
    sys.path.insert(0, src)

# 3) Also expose for every subprocess spawned later
os.environ["PYTHONPATH"] = src + ":" + os.environ.get("PYTHONPATH", "")

import shift_study  # smoke-import to confirm it works
print(f"\n✅ shift_study importable from {src}")
 """
# %% [markdown]
# ## Cell 2 — Setup & Verify

# %%
import os, shutil, torch, datetime, time, subprocess
from pathlib import Path

PROJ    = "/home/23alr106/code/walmart-tabular-shift"
RAW_DIR = "/home/23alr106/code/walmart-tabular-shift/data/raw"

os.environ["SS_PROJ"]    = PROJ
os.environ["SS_RAW_DIR"] = RAW_DIR

def run_command_with_logging(cmd, cwd=None, env=None, step_name="", check=True):
    cmd_str = " ".join(cmd)
    t0 = time.time()
    start_dt = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n[{start_dt}] [{step_name}] RUNNING: {cmd_str}")
    if cwd:
        print(f"[{start_dt}] [{step_name}] CWD: {cwd}")
    
    # Run the subprocess
    res = subprocess.run(cmd, cwd=cwd, env=env)
    
    elapsed = time.time() - t0
    end_dt = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    if res.returncode == 0:
        print(f"[{end_dt}] [{step_name}] ✅ SUCCESS: {cmd_str}")
        print(f"[{end_dt}] [{step_name}] Elapsed: {elapsed:.2f}s ({elapsed/60:.2f}m)\n")
    else:
        status_msg = f"FAILED (exit status {res.returncode})"
        print(f"[{end_dt}] [{step_name}] ❌ {status_msg}: {cmd_str}")
        print(f"[{end_dt}] [{step_name}] Elapsed: {elapsed:.2f}s ({elapsed/60:.2f}m)\n")
        if check:
            raise subprocess.CalledProcessError(res.returncode, cmd)
    return res

now_dt = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print(f"[{now_dt}] [SETUP] Starting setup verification...")
print(f"[{now_dt}] [SETUP] Project directory: {PROJ}")
print(f"[{now_dt}] [SETUP] Raw data directory: {RAW_DIR}")

print("\n=== Data files ===")
for fname in ["calendar.csv", "sell_prices.csv",
              "sales_train_validation.csv",
              "sales_train_evaluation.csv",
              "sample_submission.csv"]:
    p = Path(RAW_DIR) / fname
    status = f"✅ {p.stat().st_size/1e6:.0f} MB" if p.exists() else "❌ MISSING"
    print(f"  {fname:<45s} {status}")

print("\n=== GPU ===")
print(f"  CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print(f"  {torch.cuda.get_device_name(0)}  {props.total_memory/1e9:.1f} GB")

print("\n✅ Setup complete.")

# %% [markdown]
# ## Cell 3 — Build Feature Table (~15 min)

# %%
import subprocess, os, sys
PROJ    = os.environ["SS_PROJ"]
RAW_DIR = os.environ["SS_RAW_DIR"]

env = {**os.environ, "PYTHONPATH": f"{PROJ}/src:" + os.environ.get("PYTHONPATH", "")}

run_command_with_logging([
    sys.executable, f"{PROJ}/scripts/build_features.py",
    "--raw-dir", RAW_DIR,
    "--out",     f"{PROJ}/features.parquet",
], cwd=PROJ, env=env, step_name="BUILD_FEATURES", check=True)

# %% [markdown]
# ## Cell 4 — Smoke Test

# %%
import subprocess, os, sys
PROJ = os.environ["SS_PROJ"]
env  = {**os.environ, "PYTHONPATH": f"{PROJ}/src:" + os.environ.get("PYTHONPATH", "")}

run_command_with_logging([
    sys.executable, f"{PROJ}/scripts/run_experiment.py",
    "--model", "lightgbm", "--experiment", "e2",
    "--features", f"{PROJ}/features.parquet",
    "--out",      f"{PROJ}/results/smoke",
    "--sample",   "500",
], cwd=PROJ, env=env, step_name="SMOKE_TEST", check=True)

# %% [markdown]
# ## Cell 5 — Clean Smoke Artefacts

# %%
import shutil, os, datetime
PROJ = os.environ.get("SS_PROJ", "/home/23alr106/code/walmart-tabular-shift")

now_dt = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print(f"[{now_dt}] [CLEANUP] Cleaning smoke test artefacts...")
smoke_dir = os.path.join(PROJ, "results", "smoke")
removed_count = 0
if os.path.exists(smoke_dir):
    shutil.rmtree(smoke_dir)
    print(f"[{now_dt}] [CLEANUP] removed directory {smoke_dir}")
    removed_count = 1

print(f"[{now_dt}] [CLEANUP] ✅ Cleaned {removed_count} files. Ready for full study.")

# %% [markdown]
# ## Cell 6 — Full Study: Session 1
# XGBoost / LightGBM / TabM × E1/E2/E3. Resumable — rerun after any session death.
# 
# > ⚠️ TabR excluded here — run it in Cell 7 in a dedicated session.

# %%
import subprocess, itertools, time, os, sys, datetime
PROJ = os.environ["SS_PROJ"]
env  = {**os.environ, "PYTHONPATH": f"{PROJ}/src:" + os.environ.get("PYTHONPATH", "")}

t0 = time.time()
loop_idx = 1
total_loops = len(list(itertools.product(["xgboost", "lightgbm", "tabm"], ["e1", "e2", "e3"])))
for model, exp in itertools.product(
    ["xgboost", "lightgbm", "tabm"], ["e1", "e2", "e3"]
):
    now_dt = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n{'='*55}")
    print(f"[{now_dt}] [SESSION 1] Loop {loop_idx}/{total_loops}: ▶ {model} × {exp}")
    print(f"{'='*55}")
    run_command_with_logging([
        sys.executable, f"{PROJ}/scripts/run_experiment.py",
        "--model", model, "--experiment", exp,
        "--features", f"{PROJ}/features.parquet",
        "--out", f"{PROJ}/results",
        "--skip-existing",
    ], cwd=PROJ, env=env, step_name=f"SESSION_1_{model.upper()}_{exp.upper()}", check=False)
    loop_idx += 1

now_dt = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print(f"[{now_dt}] [SESSION 1] ✅ Session 1 done in {(time.time()-t0)/3600:.2f} h")

# %% [markdown]
# ## Cell 7 — TabR (Dedicated Session 2, ~8–10 h)
# Start a fresh session, run Cells 1–3, then run **only this cell** with *Persistence → Save & Run All*.

# %%
import subprocess, time, os, sys, datetime
PROJ = os.environ["SS_PROJ"]

t0 = time.time()
loop_idx = 1
total_loops = len(["e1", "e2", "e3"])
for exp in ["e1", "e2", "e3"]:
    now_dt = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n{'='*55}")
    print(f"[{now_dt}] [SESSION 2] Loop {loop_idx}/{total_loops}: ▶ tabr × {exp}")
    print(f"{'='*55}")
    run_command_with_logging([
        sys.executable, f"{PROJ}/scripts/run_experiment.py",
        "--model",      "tabr",
        "--experiment", exp,
        "--features",   f"{PROJ}/features.parquet",
        "--out",        f"{PROJ}/results",
        "--skip-existing",
    ], cwd=PROJ, step_name=f"SESSION_2_TABR_{exp.upper()}", check=False)
    loop_idx += 1

now_dt = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print(f"[{now_dt}] [SESSION 2] ✅ TabR done in {(time.time()-t0)/3600:.2f} h")

# %% [markdown]
# ## Cell 8 — Crossover Curve + Report

# %%
import subprocess, os, sys
PROJ = os.environ["SS_PROJ"]

run_command_with_logging([
    sys.executable, f"{PROJ}/scripts/run_crossover.py",
    "--features", f"{PROJ}/features.parquet",
    "--out",      f"{PROJ}/results",
], cwd=PROJ, step_name="RUN_CROSSOVER", check=True)

run_command_with_logging([
    sys.executable, f"{PROJ}/scripts/make_report.py",
    "--results", f"{PROJ}/results",
], cwd=PROJ, step_name="MAKE_REPORT", check=True)

# %% [markdown]
# ## Cell 9 — Display Results

# %%
import pandas as pd
from IPython.display import display, Image, Markdown
from pathlib import Path
import datetime

PROJ = os.environ.get("SS_PROJ", "/home/23alr106/code/walmart-tabular-shift")
REPORT = Path(PROJ) / "results" / "report"

now_dt = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print(f"[{now_dt}] [DISPLAY] Displaying results from {REPORT}...")

mt = REPORT / "master_table.csv"
if mt.exists():
    print(f"[{now_dt}] [DISPLAY] Found master table: {mt}")
    display(Markdown("### Master Table — WAPE & Δ"))
    display(pd.read_csv(mt).style.format(precision=4))
else:
    print(f"[{now_dt}] [DISPLAY] ⚠️ Master table {mt} not found. Run Cell 8 first.")

for fname, title in [
    ("fig1_shift_sensitivity.png", "Figure 1: Shift Degradation"),
    ("fig2_crossover.png",         "Figure 2: Cold-Start Crossover"),
]:
    p = REPORT / fname
    if p.exists():
        print(f"[{now_dt}] [DISPLAY] Found image: {p}")
        display(Markdown(f"### {title}"))
        display(Image(str(p), width=720))
    else:
        print(f"[{now_dt}] [DISPLAY] ⚠️ Image {fname} not found.")

wt = REPORT / "wilcoxon.csv"
if wt.exists():
    print(f"[{now_dt}] [DISPLAY] Found Wilcoxon tests: {wt}")
    display(Markdown("### Wilcoxon Signed-Rank Tests"))
    display(pd.read_csv(wt).style.format(precision=4))

# %% [markdown]
# ## Cell 10 — List Output Files

# %%
import os, datetime
PROJ = os.environ.get("SS_PROJ", "/home/23alr106/code/walmart-tabular-shift")

now_dt = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print(f"[{now_dt}] [LIST_FILES] Listing output files in {PROJ}/results/...")

results_dir = f"{PROJ}/results/"
if not os.path.exists(results_dir):
    print(f"[{now_dt}] [LIST_FILES] ⚠️ Results directory {results_dir} does not exist.")
else:
    for root, dirs, files in os.walk(results_dir):
        dirs[:] = [d for d in sorted(dirs) if d != "configs"]
        lvl    = root.replace(results_dir, "").count(os.sep)
        indent = "  " * lvl
        print(f"{indent}{os.path.basename(root) or 'results'}/")
        for f in sorted(files):
            size = os.path.getsize(os.path.join(root, f))
            unit = f"{size/1e6:.1f} MB" if size > 1e6 else f"{size/1e3:.0f} KB"
            print(f"{indent}  {f}  ({unit})")
