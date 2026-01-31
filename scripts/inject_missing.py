#!/usr/bin/env python3
"""inject_missing.py

Simple script to inject missing values into a CSV using a single call.

Example:
  python scripts/inject_missing.py -i data/data.csv -o data/data_with_nans.csv -c tenure MonthlyCharges -p 0.1 -S 42

The script prints a short summary of how many NaNs were added per column.
"""

from typing import Iterable, Optional, Union, Dict
import argparse
import numpy as np
import pandas as pd


def inject_missing(
    df: pd.DataFrame,
    columns: Optional[Iterable[str]] = None,
    pct: Union[float, Dict[str, float]] = 0.1,
    strategy: str = "random",
    block_size: int = 0,
    random_state: Optional[int] = None,
    condition: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """Return a copy of df with missing values injected.

    Parameters:
    - columns: iterable of column names to modify. If None, all columns are eligible.
    - pct: fraction to set missing (0-1) or dict mapping column->pct.
    - strategy: 'random' (default), 'block', or 'condition'.
    - block_size: used when strategy=='block' to set contiguous block length.
    - random_state: seed for reproducibility.
    - condition: boolean Series used when strategy=='condition' (must align with df).
    """
    rng = np.random.RandomState(random_state)
    df = df.copy()
    cols = list(columns) if columns is not None else list(df.columns)

    def get_pct(c):
        return pct[c] if isinstance(pct, dict) else float(pct)

    n = len(df)
    for c in cols:
        if c not in df.columns:
            print(f"warning: column '{c}' not found in input - skipping")
            continue

        p = get_pct(c)
        if p <= 0:
            continue

        if strategy == "random":
            k = int(np.floor(p * n))
            if k <= 0:
                continue
            idx = rng.choice(n, size=k, replace=False)
            df.loc[df.index[idx], c] = np.nan

        elif strategy == "block":
            if block_size <= 0:
                raise ValueError("block_size must be > 0 for 'block' strategy")
            total = int(np.floor(p * n))
            if total <= 0:
                continue
            starts = rng.choice(max(1, n - block_size + 1), size=max(1, int(np.ceil(total / block_size))), replace=False)
            filled = 0
            for s in starts:
                if filled >= total:
                    break
                end = min(s + block_size, n)
                to_fill = min(total - filled, end - s)
                df.loc[df.index[s:s + to_fill], c] = np.nan
                filled += to_fill

        elif strategy == "condition":
            if condition is None:
                raise ValueError("condition must be provided for 'condition' strategy")
            df.loc[condition, c] = np.nan

        else:
            raise ValueError("strategy must be 'random', 'block', or 'condition'")

    return df


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Inject missing values into a CSV file")
    p.add_argument("-i", "--input", default="data/data.csv", help="Input CSV path")
    p.add_argument("-o", "--output", default="data/data_with_nans.csv", help="Output CSV path")
    p.add_argument("-c", "--columns", nargs="+", default=None, help="Columns to inject (default: all)")
    p.add_argument("-p", "--pct", type=float, default=0.1, help="Fraction to set missing (0-1) or per-column mapping not supported via CLI)")
    p.add_argument("-s", "--strategy", choices=["random", "block", "condition"], default="random")
    p.add_argument("--block-size", type=int, default=0, help="Block size for 'block' strategy")
    p.add_argument("-S", "--seed", type=int, default=None, help="Random seed for reproducibility")
    p.add_argument("--condition-col", default=None, help="Column name to use as boolean condition for 'condition' strategy")
    return p.parse_args()


def main():
    args = parse_args()
    df = pd.read_csv(args.input)
    condition = None
    if args.strategy == "condition":
        if args.condition_col is None:
            raise SystemExit("--condition-col is required for 'condition' strategy")
        if args.condition_col not in df.columns:
            raise SystemExit(f"condition column '{args.condition_col}' not in input")
        condition = df[args.condition_col].astype(bool)

    before = df.isnull().sum()
    df_out = inject_missing(
        df,
        columns=args.columns,
        pct=args.pct,
        strategy=args.strategy,
        block_size=args.block_size,
        random_state=args.seed,
        condition=condition,
    )
    df_out.to_csv(args.output, index=False)

    added = df_out.isnull().sum() - before
    print(f"Saved: {args.output}")
    if (added > 0).any():
        print("Newly added missing values per column:")
        print(added[added > 0].to_string())
    else:
        print("No missing values were added (check pct or columns).")


if __name__ == "__main__":
    main()
