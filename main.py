"""Entry-Point für das Live-System (MT5 + Vantage)."""
from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Sequence

import pandas as pd

from live_core.config import LiveConfig
from live_core.cycle import CycleRunner
from live_core.execution import OrderManager
from live_core.ml_model import MLProbabilityProvider
from live_core.mt5_adapter import MetaTrader5Adapter
from live_core.signals import SignalEngine

logger = logging.getLogger("ew_live")
DEFAULT_REMOTE_SEGMENTS = Path(r"C:\Users\Administrator\Documents\EW-Livev2.1\logs\segments")
LOCAL_RESULTS_SEGMENTS = Path.cwd() / "Ergebnisse" / "segments"


class SegmentBufferHandler(logging.Handler):
    """Collects log records so they can be flushed into per-cycle snapshot files."""

    def __init__(self) -> None:
        super().__init__()
        self._buffer: List[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = self.format(record)
        except Exception:  # pragma: no cover - formatting fallback
            message = record.getMessage()
        self._buffer.append(message)

    def pop_lines(self) -> List[str]:
        lines = self._buffer
        self._buffer = []
        return lines

    def has_data(self) -> bool:
        return bool(self._buffer)


class LogSegmentWriter:
    """Writes buffered log lines to timestamped files after N cycles."""

    def __init__(self, handler: SegmentBufferHandler, target_dir: Path, cycles_per_file: int) -> None:
        self.handler = handler
        self.target_dir = target_dir
        self.cycles_per_file = max(cycles_per_file, 1)
        self.target_dir.mkdir(parents=True, exist_ok=True)

    def maybe_flush(self, cycle_index: int) -> Optional[Path]:
        if cycle_index % self.cycles_per_file != 0:
            return None
        return self.flush(cycle_index)

    def flush(self, cycle_index: Optional[int] = None) -> Optional[Path]:
        lines = self.handler.pop_lines()
        if not lines:
            return None
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        suffix = f"_cycle{cycle_index:05d}" if cycle_index is not None else ""
        path = self.target_dir / f"log_{timestamp}{suffix}.txt"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    def flush_remaining(self) -> Optional[Path]:
        if not self.handler.has_data():
            return None
        return self.flush()


def configure_logging(log_path: Path, extra_handlers: Optional[Sequence[logging.Handler]] = None) -> None:
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s", datefmt="%Y-%m-%dT%H:%M:%S%z")
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.handlers.clear()
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    if extra_handlers:
        for handler in extra_handlers:
            handler.setFormatter(formatter)
            logger.addHandler(handler)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live-Automation für EW-Strategie ohne EMA/ML")
    parser.add_argument("--config", "-c", help="Pfad zur JSON-Konfigurationsdatei")
    parser.add_argument("--dry-run", action="store_true", help="Nur Signale berechnen, nichts senden")
    parser.add_argument("--once", action="store_true", help="Nur ein Zyklus statt Dauerschleife")
    parser.add_argument("--symbols-file", help="Pfad zur Datei mit einem Symbol pro Zeile (Standard: Symbols.txt)")
    parser.add_argument("--log-file", default="live_execution.log", help="Pfad zur Logdatei (Standard: live_execution.log)")
    parser.add_argument(
        "--log-segment-dir",
        help=(
            "Optionaler Ordner für segmentierte Logs (Standard: C:\\Users\\Administrator\\Documents"
            "\\EW-Livev2.1\\logs\\segments, sonst <log>/segments)"
        ),
    )
    parser.add_argument(
        "--segment-cycles",
        type=int,
        default=5,
        help="Anzahl der Zyklen pro Logsegment (0 deaktiviert Segmentierung)",
    )
    return parser.parse_args()


def load_symbols(path: Optional[str]) -> List[str]:
    if not path:
        return []
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = Path.cwd() / path
    if not candidate.exists():
        return []
    return [line.strip() for line in candidate.read_text(encoding="utf-8").splitlines() if line.strip() and not line.strip().startswith("#")]


def verify_mt5_connection(adapter: MetaTrader5Adapter) -> None:
    adapter.connect()
    info = adapter.get_account_info()
    if not info:
        raise ConnectionError("MT5 meldet keine Kontoinformationen")
    logger.info(
        "MT5 verbunden: Konto %(login)s %(currency)s Balance=%(balance).2f Leverage=%(leverage)s",
        {
            "login": info.get("login"),
            "currency": info.get("currency"),
            "balance": info.get("balance"),
            "leverage": info.get("leverage"),
        },
    )


def main() -> None:
    args = parse_args()
    base_config = LiveConfig.load_from_file(args.config)
    env_overrides = LiveConfig.env_overrides()
    cfg = base_config.with_overrides(env_overrides, provided_keys=set(env_overrides.keys()))
    cfg = cfg.apply_aggressive_profile()
    logger.info(
        "LiveConfig aktiv: symbol=%s timeframe=%s risk_per_trade=%.3f dynamic_dd_risk=%s use_ml_filters=%s size_short_factor=%.2f use_vol_target=%s",
        cfg.symbol,
        cfg.timeframe,
        cfg.risk_per_trade,
        cfg.dynamic_dd_risk,
        cfg.use_ml_filters,
        cfg.size_short_factor,
        cfg.use_vol_target,
    )
    symbols_file = args.symbols_file or cfg.symbols_file
    cfg.symbols_file = symbols_file
    log_path = Path(args.log_file)
    if not log_path.is_absolute():
        log_path = Path.cwd() / log_path
    log_path.parent.mkdir(parents=True, exist_ok=True)
    segment_handler: Optional[SegmentBufferHandler] = None
    segment_writer: Optional[LogSegmentWriter] = None
    if args.segment_cycles and args.segment_cycles > 0:
        segment_handler = SegmentBufferHandler()
        extra_handlers = (segment_handler,)
    else:
        extra_handlers = ()

    configure_logging(log_path, extra_handlers)

    if segment_handler:
        segment_dir = _resolve_segment_dir(log_path, args.log_segment_dir)
        segment_writer = LogSegmentWriter(segment_handler, segment_dir, args.segment_cycles)
    symbols = load_symbols(symbols_file)
    if not symbols:
        logger.warning(f"Symbols-Datei '{symbols_file}' leer oder fehlt -> verwende '{cfg.symbol}'")
        symbols = [cfg.symbol]
    adapter = MetaTrader5Adapter()
    try:
        verify_mt5_connection(adapter)
    except Exception as exc:
        logger.error(f"MT5-Verbindung fehlgeschlagen: {exc}")
        sys.exit(1)
    ml_provider: Optional[MLProbabilityProvider] = None
    if cfg.ml_probability_path:
        try:
            ml_provider = MLProbabilityProvider(cfg.ml_probability_path)
        except FileNotFoundError as exc:
            logger.warning(f"ML-Wahrscheinlichkeiten nicht geladen: {exc}")
    engine = SignalEngine(cfg, ml_provider)
    manager = OrderManager(adapter, cfg)

    def log_live(symbol: str, message: str) -> None:
        logger.info(f"[{symbol}] {message}")

    runner = CycleRunner(cfg, adapter, engine, manager, log_live)

    try:
        while True:
            summary = runner.run_cycle(symbols, args.dry_run)
            validation_rate = summary.validated_signals / summary.total_signals if summary.total_signals else 0.0
            execution_rate = summary.executed_trades / summary.validated_signals if summary.validated_signals else 0.0
            duplicate_rate = summary.duplicate_signals / summary.total_signals if summary.total_signals else 0.0
            log_live(
                "cycle",
                (
                    f"Cycle #{summary.index} summary: Symbols={summary.symbols_processed} Signals={summary.total_signals} "
                    f"Validated={summary.validated_signals} Executed={summary.executed_trades} "
                    f"Duplicates={summary.duplicate_signals} ValidationRate={validation_rate:.3f} "
                    f"ExecutionRate={execution_rate:.3f} DuplicateRate={duplicate_rate:.3f} "
                    f"Dauer={summary.duration_seconds:.2f}s DryRun={summary.dry_run}"
                ),
            )
            exposure_stats = manager.report_cycle_metrics()
            log_live(
                "cycle",
                (
                    f"Balance={exposure_stats['balance']:.2f} Exposure={exposure_stats['exposure']:.2f} "
                    f"ExposurePct={exposure_stats['exposure_pct']:.2f}% Drawdown={exposure_stats['drawdown']:.2f}%"
                ),
            )
            if segment_writer:
                rotated_path = segment_writer.maybe_flush(summary.index)
                if rotated_path:
                    logger.info(f"[cycle] Logsegment gespeichert: {rotated_path.name}")
            if args.once or args.dry_run:
                break
            time.sleep(10)
    finally:
        for handler in logger.handlers:
            handler.flush()
        if segment_writer:
            segment_writer.flush_remaining()
        adapter.disconnect()


def _resolve_segment_dir(log_path: Path, override: Optional[str]) -> Path:
    if override:
        candidate = Path(override).expanduser()
        if not candidate.is_absolute():
            candidate = (log_path.parent / candidate).resolve()
        candidate.mkdir(parents=True, exist_ok=True)
        return candidate

    try:
        DEFAULT_REMOTE_SEGMENTS.mkdir(parents=True, exist_ok=True)
        return DEFAULT_REMOTE_SEGMENTS
    except (OSError, PermissionError):
        LOCAL_RESULTS_SEGMENTS.mkdir(parents=True, exist_ok=True)
        return LOCAL_RESULTS_SEGMENTS


if __name__ == "__main__":
    main()