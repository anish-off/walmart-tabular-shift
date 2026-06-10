"""Download M5 data via the Kaggle API into data/raw/.
Requires ~/.kaggle/kaggle.json (Kaggle > Account > Create API Token)."""
import argparse
import subprocess
import sys
import zipfile
from pathlib import Path

COMPETITION = "m5-forecasting-accuracy"
NEEDED = ["sales_train_evaluation.csv", "calendar.csv", "sell_prices.csv"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if all((out / f).exists() for f in NEEDED):
        print("All M5 files already present, nothing to do.")
        return
    cmd = ["kaggle", "competitions", "download", "-c", COMPETITION, "-p", str(out)]
    print("Running:", " ".join(cmd))
    res = subprocess.run(cmd)
    if res.returncode != 0:
        sys.exit("kaggle download failed - is kaggle.json configured and the "
                 "competition rules accepted on the website?")
    for z in out.glob("*.zip"):
        print("Extracting", z)
        with zipfile.ZipFile(z) as f:
            f.extractall(out)
        z.unlink()
    missing = [f for f in NEEDED if not (out / f).exists()]
    if missing:
        sys.exit(f"Missing after download: {missing}")
    print("Done. Files in", out)


if __name__ == "__main__":
    main()
