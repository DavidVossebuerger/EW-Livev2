"""
Resample XAUUSD M30 data to H1 and Daily timeframes.

Usage:
    python scripts/resample_xauusd.py

Creates:
    - daten/xauusd_m30.csv
    - daten/xauusd_h1.csv  
    - daten/xauusd_daily.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent.parent
M30_SOURCE = BASE_DIR / "xauusd-m30-bid-2010-01-13-2025-11-26.csv"
OUTPUT_DIR = BASE_DIR / "daten"

def load_m30_data(filepath: Path) -> pd.DataFrame:
    """Load M30 data and convert timestamps."""
    print(f"Loading M30 data from {filepath}...")
    
    df = pd.read_csv(filepath)
    
    # Convert millisecond timestamp to datetime
    df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    # Standardize column names
    df = df.rename(columns={
        'open': 'Open',
        'high': 'High', 
        'low': 'Low',
        'close': 'Close'
    })
    
    # Set date as index for resampling
    df = df.set_index('date')
    df = df[['Open', 'High', 'Low', 'Close']]
    
    print(f"  Loaded {len(df):,} M30 bars")
    print(f"  Date range: {df.index.min()} to {df.index.max()}")
    
    return df


def resample_ohlc(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Resample OHLC data to a higher timeframe."""
    resampled = df.resample(freq).agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last'
    }).dropna()
    
    return resampled


def add_indicators(df: pd.DataFrame, atr_period: int = 14) -> pd.DataFrame:
    """Add ATR and other indicators needed by backtester."""
    df = df.copy()
    
    # True Range
    high = df['High']
    low = df['Low']
    close = df['Close']
    
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(atr_period, min_periods=1).mean()
    
    # EMA for filters
    df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()
    
    # Volume proxy (using range as proxy since we don't have volume)
    df['Volume'] = (df['High'] - df['Low']) * 1000  # Range-based pseudo-volume
    
    return df


def save_for_backtester(df: pd.DataFrame, filepath: Path, tf_name: str):
    """Save in format expected by backtester."""
    df = df.copy()
    df = df.reset_index()
    
    # Rename columns to lowercase (backtester expects this)
    df.columns = [c.lower() for c in df.columns]
    
    # Ensure proper column order
    cols = ['date', 'open', 'high', 'low', 'close', 'atr', 'ema20', 'ema50', 'volume']
    df = df[cols]
    
    df.to_csv(filepath, index=False)
    print(f"  Saved {tf_name}: {len(df):,} bars -> {filepath}")


def main():
    print("=" * 60)
    print("XAUUSD Data Resampler")
    print("=" * 60)
    
    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Load M30 source data
    m30 = load_m30_data(M30_SOURCE)
    
    # Resample to H1 (2 x M30 bars)
    print("\nResampling to H1...")
    h1 = resample_ohlc(m30, '1h')
    h1 = add_indicators(h1)
    print(f"  Created {len(h1):,} H1 bars")
    
    # Resample to Daily
    print("\nResampling to Daily...")
    daily = resample_ohlc(m30, '1D')
    daily = add_indicators(daily)
    print(f"  Created {len(daily):,} Daily bars")
    
    # Add indicators to M30 as well
    print("\nAdding indicators to M30...")
    m30 = add_indicators(m30)
    
    # Save all timeframes
    print("\nSaving files...")
    save_for_backtester(m30, OUTPUT_DIR / "xauusd_m30.csv", "M30")
    save_for_backtester(h1, OUTPUT_DIR / "xauusd_h1.csv", "H1")
    save_for_backtester(daily, OUTPUT_DIR / "xauusd_daily.csv", "Daily")
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary:")
    print(f"  M30:   {len(m30):>8,} bars | {m30.index.min().date()} to {m30.index.max().date()}")
    print(f"  H1:    {len(h1):>8,} bars | {h1.index.min().date()} to {h1.index.max().date()}")
    print(f"  Daily: {len(daily):>8,} bars | {daily.index.min().date()} to {daily.index.max().date()}")
    print("=" * 60)
    print("Done! Use with: python EW_Backtester.py --local-data daten/xauusd")


if __name__ == "__main__":
    main()
