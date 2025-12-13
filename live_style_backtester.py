"""Backtester, der die Live-Strategie mit den Live-Settings simuliert.

- nutzt die LiveConfig (aggressives Profil) aus live_core.config
- erzeugt Signale über live_core.signals.SignalEngine (gleiche Entry/Setup-Logik wie live)
- berechnet Positionsgröße nach dem Live-Risiko-Sizer (_calculate_volume aus live_core.execution)
- simuliert Stop/TP + optional ATR-basiertes Trailing-SL

Eingabe: resampelte OHLC CSVs im Ordner ./daten (daily/h1/m30). Beispiel: h1_XAUUSD.csv
Ausgabe: einfache Kennzahlen + CSV der Trades.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from live_core.config import LiveConfig
from live_core.signals import Dir, EntrySignal, SignalEngine, add_indicators


@dataclass
class SymbolInfo:
    # ICMarkets-EU Werte für XAUUSD (Metals):
    trade_tick_size: float = 0.01
    trade_tick_value: float = 1.0  # 100 XAU Kontrakt -> 0.01 Tick = $1
    trade_contract_size: float = 100.0
    point: float = 0.01
    volume_step: float = 0.01
    volume_min: float = 0.01
    volume_max: float = 100.0
    leverage: float = 30.0  # Anfangliche Margin 100:1? Anzeige zeigt 100, ESMA 1:30 -> wir nutzen 30
    commission_round_per_lot: float = 7.0  # $7/lot Round-Turn (anpassbar)


@dataclass
class TradeResult:
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    direction: str
    entry: float
    exit: float
    stop: float
    tp: float
    lots: float
    pnl: float
    pnl_pct: float
    risk_amount: float
    rr: float
    bars_held: int


class LiveStyleBacktester:
    def __init__(self, cfg: LiveConfig, symbol_info: Optional[SymbolInfo] = None, trailing_atr_mult: float = 1.0):
        self.cfg = cfg
        self.symbol_info = symbol_info or SymbolInfo()
        self.trailing_atr_mult = trailing_atr_mult
        self._recent_returns: List[float] = []
        self.equity = float(cfg.account_balance)
        self.high_equity = self.equity
        self.trades: List[TradeResult] = []

    def _drawdown_percent(self) -> float:
        if self.high_equity <= 0:
            return 0.0
        return (self.equity / self.high_equity - 1.0) * 100.0

    def _risk_multiplier_for_dd(self, dd_percent: float) -> float:
        if not self.cfg.dynamic_dd_risk:
            return 1.0
        multiplier = 1.0
        for thr, val in sorted(self.cfg.dd_risk_steps, key=lambda t: t[0]):
            if dd_percent <= thr:
                multiplier = val
        return multiplier

    def _vol_multiplier(self) -> float:
        if not self.cfg.use_vol_target or len(self._recent_returns) < 5:
            return 1.0
        std = float(np.std(self._recent_returns))
        if std <= 1e-9:
            return 1.0
        scale = self.cfg.target_annual_vol / (std * 4)
        return max(0.4, min(1.6, scale))

    def _dynamic_trend_scale(self, signal: EntrySignal) -> float:
        scale = 1.0
        entry_tf = (signal.entry_tf or "").upper()
        scale *= self.cfg.tf_size_factors.get(entry_tf, 1.0)
        setup_key = (signal.setup or "").upper()
        scale *= self.cfg.setup_size_factors.get(setup_key, 1.0)
        return max(0.6, min(scale, 1.5))

    def _calculate_volume(self, signal: EntrySignal, stop_price: float, entry_price: float) -> Tuple[float, float, float]:
        stop_distance = abs(entry_price - stop_price)
        balance = self.equity
        base_risk_amount = max(0.0, self.cfg.risk_per_trade * balance)
        dd_multiplier = self._risk_multiplier_for_dd(abs(self._drawdown_percent()))
        vol_multiplier = self._vol_multiplier()
        scaled_risk_amount = base_risk_amount * dd_multiplier * vol_multiplier
        risk_fraction = scaled_risk_amount / max(balance, 1e-9)
        risk_fraction = max(self.cfg.risk_per_trade_min, min(risk_fraction, self.cfg.risk_per_trade_max))
        risk_amount = risk_fraction * balance
        if stop_distance <= 0 or risk_amount <= 0:
            return self.cfg.min_lot, risk_amount, stop_distance

        info = self.symbol_info
        tick_size = info.trade_tick_size or info.point or 0.01
        tick_value = info.trade_tick_value or (info.trade_contract_size or 1.0)
        ticks = stop_distance / tick_size
        risk_per_lot = ticks * tick_value
        if risk_per_lot <= 0:
            risk_per_lot = stop_distance * (info.trade_contract_size or 1.0)

        lots = (risk_amount / risk_per_lot) if risk_per_lot else self.cfg.min_lot
        if self.cfg.size_by_prob:
            probability = float(signal.confidence or 0.0)
            base_thr = max(0.5, self.cfg.ml_probability_threshold)
            frac = max(0.0, (probability - base_thr) / max(1e-6, 1.0 - base_thr))
            scale = self.cfg.prob_size_min + (self.cfg.prob_size_max - self.cfg.prob_size_min) * frac
            lots *= scale
        if signal.direction == Dir.DOWN:
            lots *= self.cfg.size_short_factor
        lots *= self._dynamic_trend_scale(signal)
        lots = max(self.cfg.min_lot, min(lots, self.cfg.max_lot))
        lots = self._cap_to_exposure(balance, entry_price, lots)
        return lots, risk_amount, stop_distance

    def _exposure_value(self, price: float, lots: float) -> float:
        info = self.symbol_info
        return price * (info.trade_contract_size or 1.0) * lots

    def _cap_to_exposure(self, balance: float, entry_price: float, lots: float) -> float:
        if lots <= 0:
            return lots
        info = self.symbol_info
        notional = self._exposure_value(entry_price, lots)
        limit_pct = max(self.cfg.max_gross_exposure_pct, 0.0)
        if limit_pct > 0:
            remaining = balance * limit_pct
            if notional > remaining:
                factor = remaining / notional if notional > 0 else 0.0
                lots = max(self.cfg.min_lot, lots * factor)
        # Margin check per leverage (simple approximation)
        if info.leverage and info.leverage > 0:
            margin_per_lot = (entry_price * (info.trade_contract_size or 1.0)) / info.leverage
            max_lots_margin = balance / max(margin_per_lot, 1e-9)
            lots = min(lots, max_lots_margin)
        lots = max(self.cfg.min_lot, min(lots, info.volume_max))
        # normalize to step
        step = max(info.volume_step, 1e-9)
        steps = math.floor(lots / step)
        lots = steps * step
        return max(self.cfg.min_lot, min(lots, info.volume_max))

    def _trailing_stop(self, direction: Dir, best: float, atr: float) -> float:
        if self.trailing_atr_mult <= 0:
            return None  # type: ignore
        if direction == Dir.UP:
            return best - atr * self.trailing_atr_mult
        return best + atr * self.trailing_atr_mult

    def _simulate_trade(self, df: pd.DataFrame, signal: EntrySignal) -> Optional[TradeResult]:
        entry_idx = df[df["date"] >= signal.entry_time].index.min()
        if entry_idx is None or pd.isna(entry_idx):
            return None
        entry_idx = int(entry_idx)
        entry_row = df.iloc[entry_idx]
        entry_price = float(entry_row["close"])
        stop_price = float(signal.stop_loss)
        tp_price = float(signal.take_profit)

        lots, risk_amount, stop_dist = self._calculate_volume(signal, stop_price, entry_price)
        if lots <= 0:
            return None

        best = entry_price
        trailing = stop_price
        bars = 0
        exit_price = entry_price
        exit_time = entry_row["date"]
        direction = signal.direction

        for i in range(entry_idx + 1, len(df)):
            row = df.iloc[i]
            high = float(row["high"])
            low = float(row["low"])
            atr = float(row.get("ATR", 0.0) or 0.0)
            bars += 1
            if direction == Dir.UP:
                best = max(best, high)
                trailing = max(trailing, self._trailing_stop(direction, best, atr) or trailing)
                if low <= trailing:
                    exit_price = trailing
                    exit_time = row["date"]
                    break
                if high >= tp_price:
                    exit_price = tp_price
                    exit_time = row["date"]
                    break
            else:
                best = min(best, low)
                trailing = min(trailing, self._trailing_stop(direction, best, atr) or trailing)
                if high >= trailing:
                    exit_price = trailing
                    exit_time = row["date"]
                    break
                if low <= tp_price:
                    exit_price = tp_price
                    exit_time = row["date"]
                    break
        else:
            # keine Auslösung -> letzter Kurs
            last = df.iloc[-1]
            exit_price = float(last["close"])
            exit_time = last["date"]

        direction_sign = 1 if direction == Dir.UP else -1
        info = self.symbol_info
        tick_size = info.trade_tick_size or info.point or 0.01
        tick_value = info.trade_tick_value or (info.trade_contract_size or 1.0)
        ticks = (exit_price - entry_price) / tick_size
        pnl = ticks * tick_value * direction_sign * lots
        commission = info.commission_round_per_lot * lots if info.commission_round_per_lot else 0.0
        pnl -= commission
        pnl_pct = pnl / max(self.equity, 1e-9) * 100.0
        rr = (abs(exit_price - entry_price) / max(stop_dist, 1e-9)) * direction_sign
        self.equity += pnl
        self.high_equity = max(self.high_equity, self.equity)
        if risk_amount > 0:
            self._recent_returns.append(pnl / risk_amount)
        return TradeResult(
            entry_time=entry_row["date"],
            exit_time=exit_time,
            direction=direction.value,
            entry=entry_price,
            exit=exit_price,
            stop=stop_price,
            tp=tp_price,
            lots=lots,
            pnl=pnl,
            pnl_pct=pnl_pct,
            risk_amount=risk_amount,
            rr=rr,
            bars_held=bars,
        )

    def run(self, df: pd.DataFrame, signals: List[EntrySignal]) -> List[TradeResult]:
        for sig in signals:
            res = self._simulate_trade(df, sig)
            if res:
                self.trades.append(res)
        return self.trades

    def summary(self) -> dict:
        if not self.trades:
            return {}
        trades_sorted = sorted(self.trades, key=lambda t: t.exit_time)
        start_equity = self.cfg.account_balance
        equity = start_equity
        equity_points = []
        for t in trades_sorted:
            equity += t.pnl
            equity_points.append((t.exit_time, equity))
        equity_df = pd.DataFrame(equity_points, columns=["date", "equity"])
        equity_df = equity_df.set_index("date").sort_index()
        max_dd_pct, max_dd_abs = self._max_drawdown(equity_df["equity"], start_equity)
        total_return_pct = (equity / start_equity - 1.0) * 100.0
        wins = [t for t in self.trades if t.pnl > 0]
        losses = [t for t in self.trades if t.pnl <= 0]
        profit_factor = (sum(t.pnl for t in wins) / abs(sum(t.pnl for t in losses))) if losses else float("inf")
        avg_pnl = np.mean([t.pnl for t in self.trades])
        median_pnl = np.median([t.pnl for t in self.trades])
        avg_rr = float(np.mean([t.rr for t in self.trades])) if self.trades else 0.0
        avg_hold_bars = float(np.mean([t.bars_held for t in self.trades])) if self.trades else 0.0
        sharpe = self._sharpe(equity_df["equity"])
        cagr = self._cagr(equity_df.index, start_equity, equity)
        return {
            "trades": len(self.trades),
            "winrate_pct": len(wins) / len(self.trades) * 100.0,
            "pnl_abs": equity - start_equity,
            "total_return_pct": total_return_pct,
            "profit_factor": profit_factor,
            "avg_pnl": avg_pnl,
            "median_pnl": median_pnl,
            "avg_rr": avg_rr,
            "avg_hold_bars": avg_hold_bars,
            "max_dd_pct": max_dd_pct,
            "max_dd_abs": max_dd_abs,
            "sharpe": sharpe,
            "cagr": cagr,
            "equity_end": equity,
        }

    def _max_drawdown(self, equity: pd.Series, start_equity: float) -> Tuple[float, float]:
        peak = start_equity
        max_dd_pct = 0.0
        max_dd_abs = 0.0
        for val in equity:
            if val > peak:
                peak = val
            dd_abs = peak - val
            dd_pct = (peak - val) / peak * 100.0
            if dd_abs > max_dd_abs:
                max_dd_abs = dd_abs
            if dd_pct > max_dd_pct:
                max_dd_pct = dd_pct
        return -max_dd_pct, -max_dd_abs

    def _sharpe(self, equity: pd.Series) -> float:
        if equity.empty:
            return 0.0
        daily = equity.resample("1D").last().ffill()
        rets = daily.pct_change().dropna()
        if rets.empty:
            return 0.0
        mean = rets.mean()
        std = rets.std()
        if std <= 0:
            return 0.0
        return float((mean / std) * math.sqrt(252))

    def _cagr(self, dates: pd.Index, start_equity: float, end_equity: float) -> float:
        if len(dates) == 0 or start_equity <= 0 or end_equity <= 0:
            return 0.0
        days = (pd.to_datetime(dates[-1]) - pd.to_datetime(dates[0])).days
        if days <= 0:
            return 0.0
        return float((end_equity / start_equity) ** (365.0 / days) - 1.0)


def load_ohlc_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"CSV nicht gefunden: {path}")
    df = pd.read_csv(path)
    if "date" not in df.columns and "timestamp" in df.columns:
        df.rename(columns={"timestamp": "date"}, inplace=True)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    return df


def main():
    base = Path(__file__).resolve().parent
    m30_path = base / "daten" / "m30_XAUUSD_from_m1.csv"
    cfg = LiveConfig.load_from_file(None)
    cfg = cfg.with_overrides({
        "symbol": "XAUUSD",
        "timeframe": "M30",
        "allow_shorts": True,
        "account_balance": 10_000.0,
        "risk_per_trade": 0.04,  # 4% pro Trade am SL
        "risk_per_trade_min": 0.04,
        "risk_per_trade_max": 0.04,
        "max_gross_exposure_pct": 0.0,  # kein zusätzliches Exponierungslimit
    })

    raw = load_ohlc_csv(m30_path)
    processed = add_indicators(raw, cfg)
    engine = SignalEngine(cfg)
    processed = engine.preprocess(processed)
    signals = engine.build_signals(processed, cfg.symbol)

    def run_scenario(label: str, cfg_overrides: dict, trailing_atr_mult: float = 1.0) -> dict:
        local_cfg = cfg.with_overrides(cfg_overrides)
        bt = LiveStyleBacktester(local_cfg, trailing_atr_mult=trailing_atr_mult)
        bt.run(processed, signals)
        summary = bt.summary()
        summary["label"] = label
        summary["risk_per_trade"] = local_cfg.risk_per_trade
        summary["max_gross_exposure_pct"] = local_cfg.max_gross_exposure_pct
        summary["trailing_atr_mult"] = trailing_atr_mult
        summary["size_by_prob"] = local_cfg.size_by_prob
        return summary

    scenarios = [
        ("04pct_cap_05pct_atr0_prob", {
            "risk_per_trade": 0.04,
            "risk_per_trade_min": 0.04,
            "risk_per_trade_max": 0.04,
            "max_gross_exposure_pct": 0.05,
            "size_by_prob": True,
        }, 0.0),
        ("04pct_cap_05pct_atr0_flat", {
            "risk_per_trade": 0.04,
            "risk_per_trade_min": 0.04,
            "risk_per_trade_max": 0.04,
            "max_gross_exposure_pct": 0.05,
            "size_by_prob": False,
        }, 0.0),
    ]

    results = []
    for label, overrides, trailing_atr_mult in scenarios:
        res = run_scenario(label, overrides, trailing_atr_mult=trailing_atr_mult)
        results.append(res)
        print(f"\nScenario: {label}")
        for k, v in res.items():
            if k == "label":
                continue
            print(f"  {k}: {v}")

    results_df = pd.DataFrame(results)
    if not results_df.empty:
        results_df["return_to_dd"] = results_df.apply(
            lambda r: r["total_return_pct"] / abs(r["max_dd_pct"]) if r.get("max_dd_pct") not in (None, 0, np.nan) else np.nan,
            axis=1,
        )
        top_sharpe = results_df.sort_values("sharpe", ascending=False).head(5)
        top_return_dd = results_df.sort_values("return_to_dd", ascending=False).head(5)

        print("\nTop 5 nach Sharpe:")
        for _, row in top_sharpe.iterrows():
            print(
                f"  {row['label']}: sharpe={row['sharpe']:.2f}, ret={row['total_return_pct']:.2f}%, dd={row['max_dd_pct']:.2f}%"
            )

        print("\nTop 5 nach Return/DD:")
        for _, row in top_return_dd.iterrows():
            print(
                f"  {row['label']}: ret/dd={row['return_to_dd']:.2f}, ret={row['total_return_pct']:.2f}%, dd={row['max_dd_pct']:.2f}%"
            )

    trades_path = base / "Ergebnisse" / "CSV" / "live_style_trades.csv"
    trades_path.parent.mkdir(parents=True, exist_ok=True)
    # Save last run trades (base scenario) only
    last_bt = LiveStyleBacktester(cfg, trailing_atr_mult=1.0)
    last_bt.run(processed, signals)
    if last_bt.trades:
        pd.DataFrame([t.__dict__ for t in last_bt.trades]).to_csv(trades_path, index=False)
        print(f"\nTrades gespeichert (base_4pct_no_cap): {trades_path}")
    # Save scenario summary table
    summary_path = base / "Ergebnisse" / "CSV" / "live_style_scenarios.csv"
    results_df.to_csv(summary_path, index=False)
    print(f"Scenario-Summary gespeichert: {summary_path}")


if __name__ == "__main__":
    main()
