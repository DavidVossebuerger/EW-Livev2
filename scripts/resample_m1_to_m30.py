from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def resample_m1_to_m30(src: Path, dst: Path, chunksize: int = 1_000_000) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    buffer = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close"])
    first = True
    for chunk in pd.read_csv(src, chunksize=chunksize):
        chunk["date"] = pd.to_datetime(chunk["timestamp"], unit="ms")
        work = pd.concat([buffer, chunk], ignore_index=True)
        if work.empty:
            continue
        work["bucket"] = work["date"].dt.floor("30T")
        last_bucket = work["bucket"].iloc[-1]
        buffer = work[work["bucket"] == last_bucket].copy()
        work = work[work["bucket"] != last_bucket]
        if work.empty:
            continue
        agg = (
            work.groupby("bucket", sort=True)
            .agg(open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last"))
            .reset_index()
            .rename(columns={"bucket": "date"})
        )
        agg.to_csv(dst, mode="w" if first else "a", index=False, header=first)
        first = False
    if not buffer.empty:
        agg = (
            buffer.groupby("bucket", sort=True)
            .agg(open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last"))
            .reset_index()
            .rename(columns={"bucket": "date"})
        )
        agg.to_csv(dst, mode="w" if first else "a", index=False, header=first)


def main() -> None:
    parser = argparse.ArgumentParser(description="Resample M1 OHLC to M30.")
    parser.add_argument("src", type=Path, help="Path to M1 CSV with timestamp, open, high, low, close")
    parser.add_argument("dst", type=Path, help="Path to output M30 CSV")
    parser.add_argument("--chunksize", type=int, default=1_000_000, help="Rows per chunk")
    args = parser.parse_args()
    resample_m1_to_m30(args.src, args.dst, args.chunksize)


if __name__ == "__main__":
    main()
