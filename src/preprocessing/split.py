"""
Chronological train/validation/test splitting.

Solar output is strongly seasonal, so a random split would let the model see
July while being tested on July and report a score it could never achieve in
deployment. Splits are therefore strictly ordered in time, and the test period
is sized to a full twelve months so the headline result covers every season
rather than a flattering stretch of summer.

Run:
    python -m src.preprocessing.split
"""

from typing import Dict, Iterator, List, Tuple

import pandas as pd

FEATURES_PATH = "data/dayahead/features.parquet"

# Test takes the final full year, leaving validation a half-year buffer ahead
# of it. Training is the shorter side of the trade, but a test set that misses
# winter cannot support a claim about year-round skill.
TRAIN_END = "2025-03-01"
VAL_END = "2025-09-01"


def chronological_split(
    df: pd.DataFrame,
    train_end: str = TRAIN_END,
    val_end: str = VAL_END,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split a time-indexed frame into train / validation / test by date.

    Boundaries are half-open so no timestamp appears in two splits.
    """
    train_end_ts = pd.Timestamp(train_end, tz="UTC")
    val_end_ts = pd.Timestamp(val_end, tz="UTC")

    train = df[df.index < train_end_ts]
    val = df[(df.index >= train_end_ts) & (df.index < val_end_ts)]
    test = df[df.index >= val_end_ts]

    _assert_no_overlap(train, val, test)
    return train, val, test


def rolling_origin_folds(
    df: pd.DataFrame,
    dev_end: str = VAL_END,
    n_folds: int = 4,
    fold_months: int = 3,
) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Expanding-window folds over the development period, for hyperparameter search.

    A single held-out block inherits whatever season it happens to cover — the
    earlier summer-only validation window tuned early stopping on the easiest
    months of the year and generalised poorly to winter. Rolling origin instead
    validates on several consecutive blocks spanning different seasons, and
    each fold trains only on data preceding its validation block, so the
    temporal ordering that matters in deployment is preserved throughout.

    Parameters
    ----------
    dev_end : str
        End of the development period. Everything at or after this is test
        data and is never touched here.
    n_folds : int
        Number of validation blocks, taken consecutively backwards from
        ``dev_end``.
    fold_months : int
        Length of each validation block.

    Returns
    -------
    list of (train, validation) frames, ordered oldest first.
    """
    dev = df[df.index < pd.Timestamp(dev_end, tz="UTC")]
    folds = []

    for i in range(n_folds, 0, -1):
        val_start = pd.Timestamp(dev_end, tz="UTC") - pd.DateOffset(months=fold_months * i)
        val_stop = val_start + pd.DateOffset(months=fold_months)

        train = dev[dev.index < val_start]
        val = dev[(dev.index >= val_start) & (dev.index < val_stop)]

        if len(train) and len(val):
            folds.append((train, val))

    if not folds:
        raise ValueError("No usable folds — check dev_end and fold sizing.")
    return folds


def _assert_no_overlap(*splits: pd.DataFrame) -> None:
    """Fail loudly if the splits are not strictly ordered and disjoint."""
    for earlier, later in zip(splits, splits[1:]):
        if len(earlier) and len(later) and earlier.index.max() >= later.index.min():
            raise ValueError(
                f"Splits overlap: {earlier.index.max()} >= {later.index.min()}"
            )


def summarise(splits: Dict[str, pd.DataFrame], target: str = "cf") -> None:
    """Print split sizes, spans, and seasonal coverage."""
    total = sum(len(s) for s in splits.values())
    print(f"{'split':<8} {'rows':>8} {'share':>7}  {'daytime':>8}  span")
    print("-" * 78)
    for name, split in splits.items():
        if not len(split):
            print(f"{name:<8} {0:>8}")
            continue
        day = int((split["is_day"] == 1).sum())
        months = sorted({ts.month for ts in split.index})
        print(f"{name:<8} {len(split):>8,} {100*len(split)/total:>6.1f}% "
              f"{day:>8,}  {split.index.min():%Y-%m-%d} -> {split.index.max():%Y-%m-%d}"
              f"  ({len(months)} distinct months)")

    for name, split in splits.items():
        if len(split):
            day = split[split["is_day"] == 1]
            print(f"\n{name}: daytime mean capacity factor "
                  f"{day[target].mean():.4f}  (n={len(day):,})")


if __name__ == "__main__":
    frame = pd.read_parquet(FEATURES_PATH)
    train, val, test = chronological_split(frame)
    summarise({"train": train, "val": val, "test": test})

    print("\nRolling-origin folds (development period only):")
    print(f"{'fold':<6} {'train rows':>11} {'val rows':>9}  train span -> val span")
    print("-" * 84)
    for i, (tr, va) in enumerate(rolling_origin_folds(frame), start=1):
        print(f"{i:<6} {len(tr):>11,} {len(va):>9,}  "
              f"{tr.index.min():%Y-%m} to {tr.index.max():%Y-%m} -> "
              f"{va.index.min():%Y-%m} to {va.index.max():%Y-%m}")
