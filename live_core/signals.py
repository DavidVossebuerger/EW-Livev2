"""Signal-Engine für die Live-Strategie auf Basis der ursprünglichen Backtest-Logik."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import LiveConfig
from .ml_model import MLProbabilityProvider


class Dir(str, Enum):
    UP = "UP"
    DOWN = "DOWN"


@dataclass
class Pivot:
    idx: int
    price: float
    kind: str


@dataclass
class Impulse:
    direction: Dir
    points: List[Pivot]


@dataclass
class ABC:
    direction: Dir
    points: List[Pivot]


@dataclass
class Setup:
    name: str
    direction: Dir
    start_time: pd.Timestamp
    start_idx: int
    zone: Tuple[float, float]
    stop_ref: float
    tp1: float
    tp2: float
    entry_tf: str = "H1"


@dataclass
class EntrySignal:
    entry_time: pd.Timestamp
    direction: Dir
    entry_price: float
    stop_loss: float
    take_profit: float
    tp2: float
    confidence: float
    setup: str
    entry_zone: Optional[Tuple[float, float]] = None
    entry_tf: str = "H1"


class ElliottEngine:
    def __init__(self, zz_pct: float, zz_atr_mult: float, min_impulse_atr: float):
        self.zz_pct = zz_pct
        self.zz_atr_mult = zz_atr_mult
        self.min_imp = min_impulse_atr

    @staticmethod
    def _thr(base: float, atr: float, pct: float, atr_mult: float) -> float:
        if pd.isna(atr):
            return base * pct
        return max(base * pct, atr * atr_mult)

    def zigzag(self, close: np.ndarray, atr: np.ndarray) -> List[Pivot]:
        pivots: List[Pivot] = []
        if len(close) < 3:
            return pivots
        last = close[0]
        hi = last
        lo = last
        hi_i = 0
        lo_i = 0
        direction: Optional[Dir] = None
        for i in range(1, len(close)):
            price = close[i]
            thr = self._thr(last, atr[i] if i < len(atr) else np.nan, self.zz_pct, self.zz_atr_mult)
            if direction in (None, Dir.UP):
                if price > hi:
                    hi = price
                    hi_i = i
                if hi - price >= thr:
                    pivots.append(Pivot(hi_i, float(hi), "H"))
                    last = hi
                    lo = price
                    lo_i = i
                    direction = Dir.DOWN
            if direction in (None, Dir.DOWN):
                if price < lo:
                    lo = price
                    lo_i = i
                if price - lo >= thr:
                    pivots.append(Pivot(lo_i, float(lo), "L"))
                    last = lo
                    hi = price
                    hi_i = i
                    direction = Dir.UP
        pivots.sort(key=lambda p: p.idx)
        cleaned: List[Pivot] = []
        for p in pivots:
            if not cleaned or cleaned[-1].kind != p.kind:
                cleaned.append(p)
            else:
                if (p.kind == "H" and p.price >= cleaned[-1].price) or (p.kind == "L" and p.price <= cleaned[-1].price):
                    cleaned[-1] = p
        return cleaned

    def detect_impulses(self, piv: List[Pivot], close: np.ndarray, atr: np.ndarray) -> List[Impulse]:
        res: List[Impulse] = []
        i = 0
        while i <= len(piv) - 6:
            window = piv[i : i + 6]
            kinds = "".join(p.kind for p in window)
            if kinds == "LHLHLH":
                p0, p1, p2, p3, p4, p5 = window
                w1 = p1.price - p0.price
                w3 = p3.price - p2.price
                if p2.price <= p0.price or w1 <= 0 or w3 < 0.6 * w1:
                    i += 1
                    continue
                if p4.price <= p1.price * 0.98:
                    i += 1
                    continue
                atr_b = atr[min(p3.idx, len(atr) - 1)] if len(atr) > 0 else np.nan
                if atr_b > 0 and (w3 / atr_b) < self.min_imp:
                    i += 1
                    continue
                res.append(Impulse(Dir.UP, [p0, p1, p2, p3, p4, p5]))
                i += 3
            elif kinds == "HLHLHL":
                p0, p1, p2, p3, p4, p5 = window
                w1 = p0.price - p1.price
                w3 = p2.price - p3.price
                if p2.price >= p0.price or w1 <= 0 or w3 < 0.6 * w1:
                    i += 1
                    continue
                if p4.price >= p1.price * 1.02:
                    i += 1
                    continue
                atr_b = atr[min(p3.idx, len(atr) - 1)] if len(atr) > 0 else np.nan
                if atr_b > 0 and (abs(w3) / atr_b) < self.min_imp:
                    i += 1
                    continue
                res.append(Impulse(Dir.DOWN, [p0, p1, p2, p3, p4, p5]))
                i += 3
            else:
                i += 1
        return res

    def detect_abcs(self, piv: List[Pivot]) -> List[ABC]:
        out: List[ABC] = []
        i = 0
        while i <= len(piv) - 4:
            window = piv[i : i + 4]
            kinds = "".join(p.kind for p in window)
            if kinds == "HLHL":
                h0, l1, h1, l2 = window
                A = h0.price - l1.price
                B = h1.price - l1.price
                if A > 0 and 0.3 <= B / A <= 0.86 and l2.price < l1.price:
                    out.append(ABC(Dir.DOWN, [h0, l1, h1, l2]))
                    i += 2
                    continue
            elif kinds == "LHLH":
                l0, h1, l1, h2 = window
                A = h1.price - l0.price
                B = h1.price - l1.price
                if A > 0 and 0.3 <= B / A <= 0.86 and h2.price > h1.price:
                    out.append(ABC(Dir.UP, [l0, h1, l1, h2]))
                    i += 2
                    continue
            i += 1
        return out

    @staticmethod
    def fib_zone(A: float, B: float, direction: Dir, zone: Tuple[float, float]) -> Tuple[float, float]:
        lo, hi = sorted(zone)
        if direction == Dir.UP:
            L = B - A
            zL = B - L * hi
            zH = B - L * lo
        else:
            L = A - B
            zL = B + L * lo
            zH = B + L * hi
        return (min(zL, zH), max(zL, zH))

    @staticmethod
    def fib_ext(A: float, B: float, direction: Dir, ext: float) -> float:
        if direction == Dir.UP:
            return B + (B - A) * (ext - 1.0)
        return B - (A - B) * (ext - 1.0)


def add_indicators(df: pd.DataFrame, cfg: LiveConfig) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    if "close" not in df.columns:
        return df
    for col in ["high", "low", "close"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    high = df["high"].fillna(df["close"])
    low = df["low"].fillna(df["close"])
    close = df["close"].ffill()
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    df["ATR"] = tr.rolling(cfg.atr_period, min_periods=1).mean()
    df["ATR_PCT"] = (df["ATR"] / close) * 100.0
    df["EMA_FAST"] = close.ewm(span=cfg.ema_fast, adjust=False).mean()
    df["EMA_SLOW"] = close.ewm(span=cfg.ema_slow, adjust=False).mean()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values("date").reset_index(drop=True)
    return df


def idx_from_time(df: pd.DataFrame, ts: pd.Timestamp) -> Optional[int]:
    idx = df[df["date"] >= ts].index
    return idx.min() if len(idx) > 0 else None


def first_touch(df: pd.DataFrame, start_ts: pd.Timestamp, zone: Tuple[float, float], window: int) -> Optional[int]:
    start_i = idx_from_time(df, start_ts)
    if start_i is None:
        return None
    zl, zh = zone
    end = min(start_i + window, len(df) - 1)
    for i in range(start_i, end + 1):
        row = df.iloc[i]
        lo = float(row["low"])
        hi = float(row["high"])
        cl = float(row["close"])
        if (lo <= zh and hi >= zl) or (zl <= cl <= zh):
            return i
    return None


def confirm_idx(df: pd.DataFrame, touch_i: int, direction: Dir, bars: int, cfg: LiveConfig) -> Optional[int]:
    end = min(touch_i + bars, len(df) - 1)
    base_idx = max(0, touch_i - 1)
    prev_hi = float(df.iloc[base_idx]["high"])
    prev_lo = float(df.iloc[base_idx]["low"])
    for i in range(touch_i, end + 1):
        row = df.iloc[i]
        close = float(row["close"])
        if "break_prev_extreme" in cfg.confirm_rules:
            if direction == Dir.UP and close > prev_hi:
                return i
            if direction == Dir.DOWN and close < prev_lo:
                return i
        if "ema_fast_cross" in cfg.confirm_rules:
            ef = float(row.get("EMA_FAST", close))
            es = float(row.get("EMA_SLOW", close))
            if direction == Dir.UP and close > ef and ef > es:
                return i
            if direction == Dir.DOWN and close < ef and ef < es:
                return i
    return end if cfg.allow_touch_if_no_confirm else None


def vol_ok(row: pd.Series, cfg: LiveConfig) -> bool:
    atr_pct = float(row.get("ATR_PCT", 0.0))
    return cfg.atr_pct_min <= atr_pct <= cfg.atr_pct_max


class SignalEngine:
    def __init__(self, cfg: LiveConfig, ml_provider: Optional[MLProbabilityProvider] = None):
        self.cfg = cfg
        self.primary_engine = ElliottEngine(cfg.primary_zz_pct, cfg.primary_zz_atr_mult, cfg.primary_min_imp_atr)
        self.h1_engine = ElliottEngine(cfg.h1_zz_pct, cfg.h1_zz_atr_mult, cfg.h1_min_imp_atr)
        self.ml_provider = ml_provider

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        return add_indicators(df, self.cfg)

    def build_signals(self, df: pd.DataFrame, symbol: str) -> List[EntrySignal]:
        if df.empty:
            return []
        setups = self._build_setups(df)
        signals: List[EntrySignal] = []
        for setup in setups:
            sig = self._evaluate_setup(df, setup)
            if sig:
                ml_prob = self._apply_ml_probability(symbol, sig)
                if ml_prob is not None:
                    sig.confidence = ml_prob
                signals.append(sig)
        return signals

    def _build_setups(self, df: pd.DataFrame) -> List[Setup]:
        pivots = self.h1_engine.zigzag(df["close"].astype(float).values, df.get("ATR", df["close"]).astype(float).values)
        impulses = self.h1_engine.detect_impulses(pivots, df["close"].astype(float).values, df.get("ATR", df["close"]).astype(float).values)
        abcs = self.h1_engine.detect_abcs(pivots)
        setups: List[Setup] = []
        for impulse in impulses:
            p0, p1, p2, p3, p4, _p5 = impulse.points
            idx_entry = min(p2.idx + 1, len(df) - 1)
            entry_time = df.iloc[idx_entry]["date"]
            if impulse.direction == Dir.UP:
                zone = self.h1_engine.fib_zone(p0.price, p1.price, Dir.UP, self.cfg.entry_zone_w3)
                tp1 = self.h1_engine.fib_ext(p0.price, p1.price, Dir.UP, self.cfg.tp1)
                tp2 = self.h1_engine.fib_ext(p0.price, p1.price, Dir.UP, self.cfg.tp2)
                setups.append(Setup("W3", Dir.UP, entry_time, idx_entry, zone, p0.price, tp1, tp2))
            else:
                zone = self.h1_engine.fib_zone(p0.price, p1.price, Dir.DOWN, self.cfg.entry_zone_w3)
                tp1 = self.h1_engine.fib_ext(p0.price, p1.price, Dir.DOWN, self.cfg.tp1)
                tp2 = self.h1_engine.fib_ext(p0.price, p1.price, Dir.DOWN, self.cfg.tp2)
                setups.append(Setup("W3", Dir.DOWN, entry_time, idx_entry, zone, p0.price, tp1, tp2))
            if self.cfg.use_w5:
                idx_w5 = min(p4.idx + 1, len(df) - 1)
                entry_time_w5 = df.iloc[idx_w5]["date"]
                zone = self.h1_engine.fib_zone(p2.price, p3.price, impulse.direction, self.cfg.entry_zone_w5)
                tp1 = self.h1_engine.fib_ext(p2.price, p3.price, impulse.direction, self.cfg.tp1)
                tp2 = self.h1_engine.fib_ext(p2.price, p3.price, impulse.direction, self.cfg.tp2)
                setups.append(Setup("W5", impulse.direction, entry_time_w5, idx_w5, zone, p2.price, tp1, tp2))
        for abc in abcs:
            a0, a1, b1, c1 = abc.points
            idx_entry = min(b1.idx + 1, len(df) - 1)
            entry_time = df.iloc[idx_entry]["date"]
            zone = self.h1_engine.fib_zone(a0.price, a1.price, abc.direction, self.cfg.entry_zone_c)
            tp1 = self.h1_engine.fib_ext(a0.price, a1.price, abc.direction, self.cfg.tp1)
            tp2 = self.h1_engine.fib_ext(a0.price, a1.price, abc.direction, self.cfg.tp2)
            setups.append(Setup("C", abc.direction, entry_time, idx_entry, zone, b1.price, tp1, tp2))
        setups.sort(key=lambda s: s.start_time)
        return setups

    def _evaluate_setup(self, df: pd.DataFrame, setup: Setup) -> Optional[EntrySignal]:
        start_idx = idx_from_time(df, setup.start_time)
        if start_idx is None:
            return None
        row = df.iloc[start_idx]
        if not vol_ok(row, self.cfg):
            return None
        if self.cfg.timeframe.upper() == "30M":
            window = self.cfg.entry_window_m30
            bars = self.cfg.confirm_bars_m30
        else:
            window = self.cfg.entry_window_h1
            bars = self.cfg.confirm_bars_h1
        first_touch_idx = first_touch(df, setup.start_time, setup.zone, window)
        if first_touch_idx is None:
            return None
        confirm_index = confirm_idx(df, first_touch_idx, setup.direction, bars, self.cfg)
        if confirm_index is None:
            return None
        entry_row = df.iloc[confirm_index]
        atr = float(entry_row.get("ATR", entry_row["close"] * 0.01))
        buffer = atr * self.cfg.atr_mult_buffer
        if setup.direction == Dir.UP:
            stop = setup.stop_ref - buffer
        else:
            stop = setup.stop_ref + buffer
        entry_price = float(entry_row["close"])
        tp = setup.tp1
        confidence = self.cfg.ml_default_probability
        return EntrySignal(
            entry_time=pd.to_datetime(entry_row["date"]),
            direction=setup.direction,
            entry_price=entry_price,
            stop_loss=stop,
            take_profit=tp,
            tp2=setup.tp2,
            confidence=confidence,
            setup=setup.name,
            entry_zone=setup.zone,
            entry_tf=setup.entry_tf,
        )

    def _apply_ml_probability(self, symbol: str, signal: EntrySignal) -> Optional[float]:
        if not self.ml_provider:
            return None
        entry = signal.entry_time
        entry_dt = entry.to_pydatetime() if hasattr(entry, "to_pydatetime") else entry
        return self.ml_provider.get_probability(symbol, entry_dt, signal.setup, signal.direction.value)
