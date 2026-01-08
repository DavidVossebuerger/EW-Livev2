"""Cycle orchestrator to run a full sweep over configured assets."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Iterable, Optional, Sequence

import pandas as pd

from .config import LiveConfig
from .execution import ExecutionCycleStats, OrderManager
from .mt5_adapter import MetaTrader5Adapter
from .signals import EntrySignal, SignalEngine


@dataclass
class CycleSummary:
    index: int
    start_time: datetime
    duration_seconds: float
    symbols_processed: int
    total_signals: int
    validated_signals: int
    duplicate_signals: int
    executed_trades: int
    dry_run: bool


def build_dataframe(rates: Sequence[dict]) -> pd.DataFrame:
    if not rates:
        return pd.DataFrame()
    df = pd.DataFrame(rates)
    if "time" in df.columns:
        df["date"] = pd.to_datetime(df["time"], unit="s")
    if "date" in df.columns:
        df = df.sort_values("date").reset_index(drop=True)
    return df


class CycleRunner:
    def __init__(
        self,
        cfg: LiveConfig,
        adapter: MetaTrader5Adapter,
        engine: SignalEngine,
        manager: OrderManager,
        logger: Callable[[str, str], None],
        struct_logger: Optional[logging.Logger] = None,
    ) -> None:
        self.cfg = cfg
        self.adapter = adapter
        self.engine = engine
        self.manager = manager
        self.logger = logger
        self.struct_logger = struct_logger
        self.cycle_index = 0

    def run_cycle(self, symbols: Iterable[str], dry_run: bool) -> CycleSummary:
        self.cycle_index += 1
        start_time = datetime.now(timezone.utc)
        total_signals = 0
        validated = 0
        duplicates = 0
        executed = 0
        momentum_exits = 0
        symbols_list = list(symbols)

        # Momentum-Exit Check vor neuen Signalen (Backtest-Parität)
        if not dry_run and getattr(self.cfg, "use_momentum_exit", False):
            momentum_exits = self.manager.check_momentum_exits()
            if momentum_exits > 0:
                self._struct_log(
                    "momentum_exits",
                    {"cycle": self.cycle_index, "closed": momentum_exits},
                )

        for symbol in symbols_list:
            self.cfg.symbol = symbol
            if getattr(self.cfg, "use_topdown_structure", False):
                daily_bars = int(getattr(self.cfg, "daily_lookback_bars", self.cfg.lookback_bars) or self.cfg.lookback_bars)
                h1_bars = int(getattr(self.cfg, "h1_lookback_bars", self.cfg.lookback_bars) or self.cfg.lookback_bars)
                m30_bars = int(getattr(self.cfg, "m30_lookback_bars", self.cfg.lookback_bars) or self.cfg.lookback_bars)

                daily_rates = self.adapter.get_rates(symbol, "D1", daily_bars)
                h1_rates = self.adapter.get_rates(symbol, "H1", h1_bars)
                m30_rates = self.adapter.get_rates(symbol, "M30", m30_bars)

                daily_df = build_dataframe(daily_rates)
                h1_df = build_dataframe(h1_rates)
                m30_df = build_dataframe(m30_rates)

                # Update volatility forecast if enabled (uses daily data)
                if getattr(self.cfg, 'use_vola_forecast', False) and not daily_df.empty:
                    try:
                        self.manager.risk_manager.update_vola_forecast(symbol, daily_df)
                    except Exception as e:
                        self.logger(symbol, f"Vola forecast update failed: {e}")

                signals = self.engine.build_signals_topdown(
                    daily_df=daily_df,
                    h1_df=h1_df,
                    m30_df=m30_df,
                    symbol=symbol,
                )
                tf_label = "TOPDOWN"
            else:
                rates = self.adapter.get_rates(symbol, self.cfg.timeframe, self.cfg.lookback_bars)
                df = build_dataframe(rates)
                signals = self.engine.build_signals(df, symbol)
                tf_label = self.cfg.timeframe
            total_signals += len(signals)
            last_signal = signals[-1] if signals else "none"
            self.logger(symbol, f"Signals={len(signals)} LastEntry={last_signal}")
            self._struct_log(
                "symbol_signals",
                {
                    "symbol": symbol,
                    "cycle": self.cycle_index,
                    "signals": len(signals),
                    "timeframe": tf_label,
                    "last_entry": self._summarize_signal(last_signal) if isinstance(last_signal, EntrySignal) else None,
                },
            )
            stats = self.manager.evaluate_signals(symbol, signals) if not dry_run else ExecutionCycleStats()
            if stats is None:
                stats = ExecutionCycleStats()
            validated += stats.validated_signals
            duplicates += stats.duplicate_signals
            executed += stats.executed_trades
        duration = (datetime.now(timezone.utc) - start_time).total_seconds()
        cycle_stats = ExecutionCycleStats(
            signals_received=total_signals,
            validated_signals=validated,
            duplicate_signals=duplicates,
            executed_trades=executed,
        )
        if not dry_run:
            self.manager.adjust_for_cycle(cycle_stats)
        return CycleSummary(
            index=self.cycle_index,
            start_time=start_time,
            duration_seconds=duration,
            symbols_processed=len(symbols_list),
            total_signals=total_signals,
            validated_signals=validated,
            duplicate_signals=duplicates,
            executed_trades=executed,
            dry_run=dry_run,
        )

    def _summarize_signal(self, signal: EntrySignal) -> dict:
        return {
            "entry_time": getattr(signal.entry_time, "isoformat", lambda: str(signal.entry_time))(),
            "direction": getattr(signal.direction, "value", signal.direction),
            "entry_price": getattr(signal, "entry_price", None),
            "stop_loss": getattr(signal, "stop_loss", None),
            "take_profit": getattr(signal, "take_profit", None),
            "confidence": getattr(signal, "confidence", None),
            "setup": getattr(signal, "setup", None),
            "entry_tf": getattr(signal, "entry_tf", None),
        }

    def _struct_log(self, code: str, payload: dict) -> None:
        if not self.struct_logger:
            return
        event = {"code": code, "ts": datetime.now(timezone.utc).isoformat()}
        event.update(payload)
        try:
            self.struct_logger.info(json.dumps(event, ensure_ascii=False))
        except Exception:
            self.struct_logger.info("%s %s", code, payload)