"""One-time feature build: raw CSVs -> data/processed/features.parquet.
Every model consumes this exact table. Run with --sample 500 for a smoke test."""
import argparse
import time
from pathlib import Path

from shift_study.data import build_base_table
from shift_study.features import add_features, FEATURES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="data/raw")
    ap.add_argument("--out", default="data/processed/features.parquet")
    ap.add_argument("--sample", type=int, default=None,
                    help="keep only N random series (smoke test)")
    args = ap.parse_args()

    t0 = time.time()
    print("Building base table...")
    base = build_base_table(args.raw_dir, sample_items=args.sample)
    print(f"  rows={len(base):,}  series={base['id'].nunique():,}  "
          f"days={base['d_int'].min()}..{base['d_int'].max()}")

    print("Adding features...")
    df = add_features(base)
    keep_cols = ["id", "d_int", "sales", "item_id", "store_id", "cat_id"] + FEATURES
    # deduplicate while preserving order
    seen = set()
    keep = []
    for c in keep_cols:
        if c not in seen:
            keep.append(c)
            seen.add(c)
    df = df[keep]
    print(f"  rows after NaN-lag drop={len(df):,}  features={len(FEATURES)}")
    assert not df[FEATURES].isna().any().any(), "NaNs remain in feature columns"

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"Wrote {out} ({out.stat().st_size / 1e6:.0f} MB) "
          f"in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
