"""Order-Manager fÃ¼r das Live-System, trifft Entscheidungen basierend auf Signalen."""
from __future__ import annotations

import hashlib
import json
import logging
import math
import ssl
import urllib.request
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from statistics import pstdev, StatisticsError
from typing import Any, Dict, Deque, List, Optional, Tuple
from urllib.error import HTTPError

import certifi

from .config import LiveConfig
from .mt5_adapter import MetaTrader5Adapter
from .order_store import OrderStore
from .signals import Dir, EntrySignal
from .risk_manager import RiskManager
from .volatility_gate import VolatilityGate, VolatilityDecision

logger = logging.getLogger("ew_live")


@dataclass
class ExecutionCycleStats:
    signals_received: int = 0  # neue, im Zyklus berÃ¼cksichtigte Signale (keine Duplikate)
    validated_signals: int = 0
    duplicate_signals: int = 0
    executed_trades: int = 0


class OrderManager:
    SUCCESS_RETCODE = 10009
    STOP_AFTER_RETCODES = {10017, 10018, 10033}
    LONG_ONLY_RETCODE = 10042
    SHORT_ONLY_RETCODE = 10043
    RETCODE_HINTS = {
        10007: "Invalid stops - Abstand liegt unter `trade_stops_level` oder Mindestabstand.",
        10030: "Unsupported filling mode - Broker akzeptiert andere `order_filling_mode`-Typen.",
        10033: "Orders limit reached - Broker-Limit fÃ¼r offene/pendende Orders erreicht (bitte weniger Orders senden).",
    }
    _VALIDATION_TARGET_RATE = 0.65
    _MAX_DYNAMIC_SHIFT = 0.08
    _MIN_DYNAMIC_SHIFT = -0.05
    _DUPLICATE_COOLDOWN_MAX = 30.0

    def __init__(self, mt5_adapter: MetaTrader5Adapter, cfg: LiveConfig):
        self.adapter = mt5_adapter
        self.cfg = cfg
        self.risk_manager = RiskManager(cfg)
        self._order_store = OrderStore(cfg.order_store_path or "logs/orders.db")
        self._volatility_gate = VolatilityGate(cfg)
        self._vola_history_cache: dict[str, tuple[list[float], float]] = {}
        self._long_only_symbols: set[str] = set()
        self._short_only_symbols: set[str] = set()
        self._highest_balance: float = cfg.account_balance
        self._recent_returns: Deque[float] = deque(maxlen=cfg.vol_window_trades)
        self._last_confidence: Dict[str, float] = {}
        self._recent_trade_times: Dict[str, Deque[datetime]] = defaultdict(deque)
        self._last_trade_time: Dict[str, datetime] = {}
        self._webhook_fingerprint = self._compute_webhook_fingerprint(cfg.webhook_url)
        self._account_leverage: float = cfg.exposure_default_leverage
        self._execution_history: list[dict] = self._load_execution_history()
        self._history_active_keys: set[str] = self._initial_active_keys_from_history()
        self._active_store: dict[str, dict] = self._load_active_store()
        self._active_store_max_age = timedelta(hours=12)
        self._active_signal_keys: set[str] = set()
        self._efficiency_history: Deque[dict[str, float]] = deque(maxlen=6)
        self._dynamic_ml_shift: float = 0.0
        self._adaptive_cooldown_minutes: float = 0.0
        if self._webhook_fingerprint:
            logger.info(f"Webhook aktiviert (Fingerprint={self._webhook_fingerprint})")

    def _struct_event(self, code: str, **payload: object) -> None:
        struct_logger = logging.getLogger("ew_struct")
        event = {"code": code, "ts": datetime.now(timezone.utc).isoformat()}
        event.update(payload)
        try:
            struct_logger.info(json.dumps(event, ensure_ascii=False))
        except Exception:
            struct_logger.info("%s %s", code, payload)

    def _apply_risk_manager_adjustment(
        self, symbol: str, signal: EntrySignal, base_volume: float, info: Optional[Any] = None
    ) -> float:
        """Applies risk-manager sensitivity sizing to the base volume."""
        if base_volume <= 0:
            return base_volume

        # Build a minimal market snapshot
        market_data = {
            "price": signal.entry_price,
            "atr": getattr(signal, "atr", None),
            "ema_fast": getattr(signal, "ema_fast", None),
            "ema_slow": getattr(signal, "ema_slow", None),
            "volatility": getattr(signal, "volatility", None),
            "price_history": getattr(signal, "price_history", None),
            "returns_window": getattr(signal, "returns_window", None),
        }

        # Historical outcomes could be sourced from adapter/trade history; keep empty for now
        historical_outcomes: list[dict] = []

        sensitivities = self.risk_manager.calculate_sensitivities(
            signal, market_data, historical_outcomes
        )
        market_uncertainty = {
            "volatility": market_data.get("volatility", 0.0),
            "symbol": symbol,  # For vola forecast lookup
        }
        adjusted = self.risk_manager.adjust_position_size_for_sensitivities(
            base_volume, sensitivities, market_uncertainty
        )
        # Re-align with broker symbol constraints after RM scaling
        if info is None:
            info = self.adapter.get_symbol_info(symbol)
        adjusted_aligned = self._align_with_symbol_constraints(symbol, info, adjusted, log_changes=False)
        if adjusted_aligned != adjusted:
            adjusted = adjusted_aligned
        if adjusted <= 0:
            return 0.0
        if adjusted != base_volume:
            logger.info(
                f"[{symbol}] Volume adjusted by RiskManager: {base_volume:.4f} -> {adjusted:.4f}"
            )
        return adjusted

    def evaluate_signals(self, symbol: str, signals: List[EntrySignal]) -> ExecutionCycleStats:
        stats = ExecutionCycleStats()
        all_positions = self.adapter.get_positions()
        self._sync_order_store_from_mt5(all_positions)
        self._update_active_signal_keys(all_positions)
        if not signals:
            return stats
        candidates: List[EntrySignal] = []
        for signal in signals:
            if not self._signal_passes_validation(signal):
                continue
            stats.validated_signals += 1
            if self._already_executed(symbol, signal):
                stats.duplicate_signals += 1
                continue
            candidates.append(signal)
            stats.signals_received += 1
        if not candidates:
            return stats
        if not self.adapter.connected:
            raise RuntimeError("MT5 nicht verbunden")
        existing_positions = [pos for pos in all_positions if pos.get("symbol") == symbol]
        limited_signals = candidates[: self.cfg.max_open_trades]
        self._vola_history_cache.clear()
        vola_series = self._load_vola_series(symbol, limited_signals[0].entry_tf if limited_signals else self.cfg.timeframe)
        for idx, signal in enumerate(limited_signals):
            open_positions = existing_positions if idx == 0 else self.adapter.get_positions(symbol)
            if self.cfg.use_ml_filters:
                threshold = self.cfg.ml_probability_threshold + self.cfg.ml_threshold_shift
                confidence = float(signal.confidence or 0.0)
                if confidence < threshold:
                    logger.info(
                        f"[{symbol}] Signal {signal.setup} {signal.direction} Ã¼bersprungen: Confidence {confidence:.3f}"
                        f" < Threshold {threshold:.3f}"
                    )
                    continue
            if not self._is_direction_allowed(symbol, signal.direction):
                logger.info(f"[{symbol}] Signal {signal.direction} Ã¼bersprungen: Broker untersagt diese Richtung")
                continue
            # Fix 4: Max 1 Position pro Asset - kein Reinskalieren erlaubt
            if open_positions:
                logger.info(
                    f"[{symbol}] Signal {signal.setup} {signal.direction} Ã¼bersprungen: "
                    f"bereits {len(open_positions)} Position(en) offen - kein Reinskalieren"
                )
                continue
            confidence = float(signal.confidence or 0.0)
            if self._trade_limit_hit(symbol):
                logger.info(
                    f"[{symbol}] Signal {signal.setup} {signal.direction} Ã¼bersprungen: Max "
                    f"{self.cfg.max_trades_per_symbol_per_hour} Trades/Std erreicht"
                )
                continue
            cooldown_remaining = self._cooldown_remaining(symbol)
            if cooldown_remaining is not None:
                minutes = cooldown_remaining.total_seconds() / 60.0
                logger.info(
                    f"[{symbol}] Signal {signal.setup} {signal.direction} Ã¼bersprungen: Cooldown aktiv "
                    f"({minutes:.1f}m verbleibend)"
                )
                continue
            info = self.adapter.get_symbol_info(symbol, refresh=True)
            if info is None:
                logger.warning(f"[{symbol}] Keine Symbolinformationen verfÃ¼gbar -> Signal Ã¼bersprungen")
                continue
            stop_price, take_profit = self._scale_order_levels(signal)
            pf_ok, pf_value = self._profit_factor_ok(signal.entry_price, stop_price, take_profit)
            if not pf_ok:
                logger.info(
                    f"[{symbol}] Signal {signal.setup} {signal.direction} Ã¼bersprungen: Chance/Risiko {pf_value:.2f} < "
                    f"Mindestfaktor {self.cfg.min_profit_factor:.2f}"
                )
                continue
            current_price = self._current_price(symbol, signal.direction)
            if current_price is None:
                logger.warning(f"[{symbol}] Kein aktueller Preis verfÃ¼gbar -> Signal Ã¼bersprungen")
                continue
            execution_price = current_price
            pending_price = self._pending_limit_price(signal)
            use_pending_order = False
            if self.cfg.use_pending_orders and pending_price is not None:
                if self._pending_price_allowed(pending_price, current_price, signal.direction):
                    execution_price = pending_price
                    use_pending_order = True
            if not self._volatility_gate_allows(symbol, signal, current_price, execution_price, vola_series):
                continue
            if not self._price_supports_order(symbol, signal.direction, execution_price, stop_price, take_profit):
                continue
            volume, risk_amount, risk_per_lot, stop_distance = self._calculate_volume(
                symbol, signal, info, stop_price, execution_price
            )
            volume = self._apply_risk_manager_adjustment(symbol, signal, volume, info)
            volume = self._align_with_symbol_constraints(symbol, info, volume)
            if volume <= 0:
                logger.info(
                    f"[{symbol}] Signal {signal.setup} {signal.direction} Ã¼bersprungen: Exposure-Limit lÃ¤sst kein Volumen zu"
                )
                continue
            if self._duplicate_position_present(open_positions, signal.direction, execution_price):
                logger.info(
                    f"[{symbol}] Signal {signal.setup} {signal.direction} Ã¼bersprungen: Position bei {execution_price:.5f} bereits vorhanden"
                )
                continue
            trade_exposure = self._exposure_value(symbol, volume, execution_price, info)
            if not self._within_exposure_limit(symbol, trade_exposure):
                continue
            direction = signal.direction.value if isinstance(signal.direction, Dir) else signal.direction
            try:
                if self._orders_limit_reached():
                    logger.warning(
                        f"[{symbol}] Orders-Limit erreicht -> Ã¼berspringe weitere Orders in diesem Zyklus"
                    )
                    break
                if use_pending_order:
                    expiration_ts = None
                    expiration_ts = 0
                    result = self.adapter.place_limit_order(
                        symbol=symbol,
                        volume=volume,
                        direction=direction,
                        price=execution_price,
                        sl=stop_price,
                        tp=take_profit,
                        expiration=expiration_ts,
                    )
                else:
                    result = self.adapter.place_market_order(
                        symbol=symbol,
                        volume=volume,
                        direction=direction,
                        sl=stop_price,
                        tp=take_profit,
                    )
                self._log_order(
                    symbol,
                    signal,
                    result,
                    volume,
                    risk_amount,
                    risk_per_lot,
                    stop_distance,
                    stop_price,
                    take_profit,
                    execution_price,
                    "pending" if use_pending_order else "market",
                )
                self._log_adjustment(symbol, result)
                self._log_filling_mode(symbol, result)
                self._log_retcode_info(symbol, result)
                self._notify_webhook(
                    symbol,
                    signal,
                    result,
                    volume,
                    risk_amount,
                    risk_per_lot,
                    stop_distance,
                    stop_price,
                    take_profit,
                    execution_price,
                    "pending" if use_pending_order else "market",
                )
                self._record_expected_return(execution_price, stop_price, take_profit)
                should_continue = self._process_execution_result(symbol, signal.direction, result)
                if result.get("retcode") == self.SUCCESS_RETCODE:
                    self._last_confidence[symbol] = confidence
                    self._record_execution(symbol, signal, result)
                    stats.executed_trades += 1
                    # Add simple key to active keys (for position tracking)
                    self._active_signal_keys.add(self._signal_key_simple(symbol, signal))
                if not should_continue:
                    break
            except Exception as exc:
                logger.error(f"[{symbol}] Fehler beim Platzieren der Order: {exc}")

        return stats

    def _current_price(self, symbol: str, direction: Dir) -> Optional[float]:
        tick = self.adapter.get_symbol_tick(symbol)
        price = None
        if tick:
            price = tick.get("ask") if direction == Dir.UP else tick.get("bid")
        if price is None or price <= 0:
            info = self.adapter.get_symbol_info(symbol, refresh=True)
            if info:
                price = getattr(info, "ask", None) if direction == Dir.UP else getattr(info, "bid", None)
                if price is None or price <= 0:
                    price = getattr(info, "last", None)
        if price is None or price <= 0:
            return None
        return float(price)

    def _load_vola_series(self, symbol: str, entry_tf: Optional[str]) -> Optional[tuple[list[float], float]]:
        if not getattr(self.cfg, "use_vola_gate", False):
            return None
        tf = (getattr(self.cfg, "vola_timeframe", None) or entry_tf or self.cfg.timeframe or "H1").upper()
        lookback = max(int(getattr(self.cfg, "vola_lookback_bars", self.cfg.lookback_bars)), 100)
        if tf not in {"H1", "M30"}:
            tf = self.cfg.timeframe.upper()
        cache_key = f"{symbol}:{tf}"
        cached = self._vola_history_cache.get(cache_key)
        if cached:
            return cached
        rates = self.adapter.get_rates(symbol, tf, lookback)
        if not rates:
            return None
        closes: list[float] = []
        for row in rates:
            close_val = row.get("close") if isinstance(row, dict) else None
            try:
                close_float = float(close_val)
            except (TypeError, ValueError):
                continue
            closes.append(close_float)
        if len(closes) < getattr(self.cfg, "vola_min_samples", 0):
            return None
        tf_minutes = 60.0 if tf == "H1" else 30.0
        self._vola_history_cache[cache_key] = (closes, tf_minutes)
        return self._vola_history_cache[cache_key]

    def _volatility_gate_allows(
        self,
        symbol: str,
        signal: EntrySignal,
        current_price: float,
        execution_price: float,
        vola_series: Optional[tuple[list[float], float]],
    ) -> bool:
        if not getattr(self.cfg, "use_vola_gate", False):
            return True
        if vola_series is None:
            logger.info(f"[{symbol}] Vola-Gate Ã¼bersprungen: keine ausreichende Historie")
            self._struct_event(
                "vola_gate_skip",
                symbol=symbol,
                reason="no_history",
                setup=str(signal.setup),
                direction=str(signal.direction),
            )
            return True
        closes, tf_minutes = vola_series
        decision: Optional[VolatilityDecision] = self._volatility_gate.probability_to_reach(
            current_price=current_price,
            target_price=execution_price,
            closes=closes,
            timeframe_minutes=tf_minutes,
        )
        if not self._volatility_gate.allows(decision):
            prob = decision.probability if decision else 0.0
            logger.info(
                f"[{symbol}] Signal {signal.setup} {signal.direction} Ã¼bersprungen: ReachProb {prob:.3f} < "
                f"{self.cfg.vola_probability_threshold:.3f} (horizon {self._volatility_gate.horizon_days:.1f}d)"
            )
            self._struct_event(
                "vola_gate_block",
                symbol=symbol,
                setup=str(signal.setup),
                direction=str(signal.direction),
                reach_prob=float(prob),
                threshold=float(self.cfg.vola_probability_threshold),
                horizon_days=float(self._volatility_gate.horizon_days),
                samples=int(decision.samples_used if decision else 0),
                sigma_daily=float(decision.sigma_daily if decision else 0.0),
                z_score=float(decision.z_score if decision else 0.0),
            )
            return False
        if decision:
            logger.info(
                f"[{symbol}] Vola-Gate ok: reach_prob {decision.probability:.3f} "
                f"sigma_d={decision.sigma_daily:.4f} z={decision.z_score:.2f} samples={decision.samples_used}"
            )
            self._struct_event(
                "vola_gate_allow",
                symbol=symbol,
                setup=str(signal.setup),
                direction=str(signal.direction),
                reach_prob=float(decision.probability),
                threshold=float(self.cfg.vola_probability_threshold),
                horizon_days=float(self._volatility_gate.horizon_days),
                samples=int(decision.samples_used),
                sigma_daily=float(decision.sigma_daily),
                z_score=float(decision.z_score),
            )
        return True

    def _duplicate_position_present(self, positions: List[Dict[str, Any]], direction: Dir, price: float) -> bool:
        if not positions:
            return False
        tolerance = max(abs(price) * 1e-5, 1e-6)
        for pos in positions:
            try:
                volume = float(pos.get("volume", 0.0) or 0.0)
            except (TypeError, ValueError):
                continue
            if volume == 0.0:
                continue
            is_long = volume > 0
            if (direction == Dir.UP and not is_long) or (direction == Dir.DOWN and is_long):
                continue
            raw_price = pos.get("price_open") or pos.get("price_current") or pos.get("price") or 0.0
            try:
                existing_price = abs(float(raw_price))
            except (TypeError, ValueError):
                continue
            if existing_price <= 0.0:
                continue
            if math.isclose(existing_price, abs(price), rel_tol=1e-4, abs_tol=tolerance):
                return True
        return False

    def _price_supports_order(self, symbol: str, direction: Dir, price: float, stop_price: float, take_profit: float) -> bool:
        eps = 1e-6
        margin = max(0.0, self.cfg.price_guard_margin)
        if direction == Dir.UP:
            if price <= stop_price - margin + eps:
                logger.info(
                    f"[{symbol}] Signal {direction} Ã¼bersprungen: aktueller Preis {price:.5f} <= Stop {stop_price:.5f}"
                )
                return False
            if price >= take_profit + margin - eps:
                logger.info(
                    f"[{symbol}] Signal {direction} Ã¼bersprungen: aktueller Preis {price:.5f} >= TP {take_profit:.5f}"
                )
                return False
        else:
            if price >= stop_price + margin - eps:
                logger.info(
                    f"[{symbol}] Signal {direction} Ã¼bersprungen: aktueller Preis {price:.5f} >= Stop {stop_price:.5f}"
                )
                return False
            if price <= take_profit - margin + eps:
                logger.info(
                    f"[{symbol}] Signal {direction} Ã¼bersprungen: aktueller Preis {price:.5f} <= TP {take_profit:.5f}"
                )
                return False
        return True

    def _pending_limit_price(self, signal: EntrySignal) -> Optional[float]:
        zone = signal.entry_zone
        if not zone:
            return None
        low, high = zone
        if not (math.isfinite(low) and math.isfinite(high)):
            return None
        return high if signal.direction == Dir.UP else low

    def _pending_price_allowed(self, limit_price: float, current_price: float, direction: Dir) -> bool:
        eps = 1e-6
        if direction == Dir.UP:
            return limit_price <= current_price - eps
        return limit_price >= current_price + eps

    def _scale_order_levels(self, signal: EntrySignal) -> Tuple[float, float]:
        """Scale stop and TP levels, enforcing minimum stop distance."""
        entry = signal.entry_price
        stop = signal.stop_loss
        tp = signal.take_profit
        stop_delta = abs(entry - stop)
        tp_delta = abs(tp - entry)
        stop_mult = max(1.0, self.cfg.stop_loss_multiplier)
        tp_mult = max(0.1, self.cfg.take_profit_multiplier)
        if stop_delta > 0:
            scaled_stop_delta = stop_delta * stop_mult
            if signal.direction == Dir.UP:
                stop = entry - scaled_stop_delta
            else:
                stop = entry + scaled_stop_delta
        if tp_delta > 0:
            scaled_tp_delta = tp_delta * tp_mult
            if signal.direction == Dir.UP:
                tp = entry + scaled_tp_delta
            else:
                tp = entry - scaled_tp_delta
        
        # CRITICAL: Enforce minimum stop distance to prevent instant stop-outs
        min_stop_pct = getattr(self.cfg, "min_stop_pct", 0.005)
        min_stop_dist = entry * min_stop_pct
        actual_stop_dist = abs(entry - stop)
        
        if actual_stop_dist < min_stop_dist:
            logger.warning(
                f"Stop distance {actual_stop_dist:.5f} below minimum {min_stop_dist:.5f}, expanding stop"
            )
            if signal.direction == Dir.UP:
                stop = entry - min_stop_dist
            else:
                stop = entry + min_stop_dist
        
        return stop, tp

    def _log_order(
        self,
        symbol: str,
        signal: EntrySignal,
        result: dict,
        volume: float,
        risk_amount: float,
        risk_per_lot: float,
        stop_distance: float,
        stop_price: float,
        take_profit: float,
        entry_price: float,
        order_mode: str,
    ) -> None:
        logger.info(
            f"[{symbol}] Order {signal.setup} {signal.direction} @{entry_price:.5f} ({order_mode}) "
            f"SL={stop_price:.2f} TP={take_profit:.2f} "
            f"Vol={volume:.3f} (risk={risk_amount:.2f}, stop={stop_distance:.5f},/lot={risk_per_lot:.3f}) -> {result}"
        )


    def _log_retcode_info(self, symbol: str, result: dict) -> None:
        status = result.get("status")
        if status == MetaTrader5Adapter._SKIPPED_INVALID_STOP:
            reason = result.get("reason", "Stop nicht genÃ¼gend Abstand")
            required = result.get("required_distance")
            actual = result.get("actual_distance")
            self._log_skipped(symbol, reason, required, actual)
            return
        retcode = result.get("retcode")
        if retcode is None:
            if result.get("status") == "fallback":
                logger.warning(f"[{symbol}] MT5-Fallback meldet: {result}")
            return
        if retcode != self.SUCCESS_RETCODE:
            description = result.get("retcode_description") or result.get("comment")
            hint = self.RETCODE_HINTS.get(retcode)
            details = " | ".join(filter(None, [description, hint]))
            details = details or "keine weitere Beschreibung"
            logger.warning(f"[{symbol}] MT5-Retcode {retcode}: {details}")

    def _log_adjustment(self, symbol: str, result: dict) -> None:
        if not result.get("stop_adjusted"):
            return
        adjusted_stop = result.get("adjusted_stop")
        distance = result.get("adjusted_distance")
        if adjusted_stop is None or distance is None:
            return
        logger.info(f"[{symbol}] Stop automatisch angepasst: {adjusted_stop:.6f} (Distanz {distance:.6f})")

    def _log_filling_mode(self, symbol: str, result: dict) -> None:
        attempts = result.get("filling_attempts")
        if not attempts:
            return
        used = result.get("filling_mode_used")
        summary = ", ".join(f"{attempt.get('mode')}({attempt.get('retcode')})" for attempt in attempts)
        if used:
            logger.info(f"[{symbol}] Filling-Mode {used} verwendet (Versuche: {summary})")
        else:
            logger.info(f"[{symbol}] Filling-Versuche: {summary}")

    def _log_volume_insight(
        self,
        symbol: str,
        signal: EntrySignal,
        lots: float,
        risk_amount: float,
        exposure_value: float,
        trend_scale: float,
        dd_multiplier: float,
        vol_multiplier: float,
        prob_scale: float,
        direction_factor: float,
        stop_distance: float,
    ) -> None:
        setup = signal.setup or "unknown"
        entry_tf = (signal.entry_tf or "unknown").upper()
        logger.info(
            f"[{symbol}] Exposure decision: setup={setup} tf={entry_tf} lots={lots:.3f} risk={risk_amount:.2f} "
            f"exposure_decision={exposure_value:.2f} trend_scale={trend_scale:.3f} dd_mult={dd_multiplier:.3f} "
            f"vol_mult={vol_multiplier:.3f} prob_scale={prob_scale:.3f} short_factor={direction_factor:.3f} "
            f"stop_dist={stop_distance:.5f}"
        )

    def adjust_for_cycle(self, stats: ExecutionCycleStats) -> None:
        if stats.signals_received <= 0:
            return
        validation_rate = stats.validated_signals / stats.signals_received
        execution_rate = stats.executed_trades / max(stats.validated_signals, 1)
        duplicate_rate = stats.duplicate_signals / stats.signals_received
        self._efficiency_history.append(
            {"validation": validation_rate, "execution": execution_rate, "duplicate": duplicate_rate}
        )
        avg_validation = sum(entry["validation"] for entry in self._efficiency_history) / len(self._efficiency_history)
        new_shift = self._dynamic_ml_shift
        if avg_validation < self._VALIDATION_TARGET_RATE * 0.9:
            new_shift = max(self._MIN_DYNAMIC_SHIFT, self._dynamic_ml_shift - 0.02)
        elif avg_validation > self._VALIDATION_TARGET_RATE * 1.1:
            new_shift = min(self._MAX_DYNAMIC_SHIFT, self._dynamic_ml_shift + 0.02)
        if new_shift != self._dynamic_ml_shift:
            logger.info(
                f"Adaptive ML-Shift angepasst: {self._dynamic_ml_shift:.3f} -> {new_shift:.3f} (valid {(avg_validation*100):.1f}% )"
            )
            self._dynamic_ml_shift = new_shift
        if duplicate_rate > 0.35:
            cooldown = min(self._DUPLICATE_COOLDOWN_MAX, duplicate_rate * 40)
            if abs(cooldown - self._adaptive_cooldown_minutes) > 0.5:
                logger.warning(
                    f"[adaptive] Duplikate-Rate {duplicate_rate:.2%} -> Cooldown auf {cooldown:.1f} Minuten erhÃ¶ht"
                )
            self._adaptive_cooldown_minutes = cooldown
        else:
            self._adaptive_cooldown_minutes = max(0.0, self._adaptive_cooldown_minutes - 1.0)

    def _resolve_store_path(self) -> Path:
        return Path(self._order_store.path)

    def _load_execution_history(self) -> list[dict]:
        return self._order_store.load_executions()

    def _save_execution_history(self) -> None:
        self._order_store.replace_executions(self._execution_history)

    def _active_store_path(self) -> Path:
        return Path(self._order_store.path)

    def _load_active_store(self) -> dict[str, dict]:
        return self._order_store.load_active()

    def _save_active_store(self) -> None:
        self._order_store.replace_active(self._active_store)

    def _initial_active_keys_from_history(self) -> set[str]:
        """Fallback: wenn MT5-Positionsabruf fehlschlÃ¤gt, verhindere Duplikate mit jÃ¼ngster Historie."""
        if not self._execution_history:
            return set()
        try:
            cutoff = datetime.now(timezone.utc) - timedelta(days=2)
        except Exception:
            return set()
        keys: set[str] = set()
        for record in reversed(self._execution_history[-200:]):  # nur jÃ¼ngste EintrÃ¤ge prÃ¼fen
            ts = record.get("timestamp")
            try:
                dt = datetime.fromisoformat(ts)
            except Exception:
                continue
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            if dt < cutoff:
                break
            key = record.get("key")
            if key:
                keys.add(key)
        return keys

    @staticmethod
    def _signal_key(symbol: str, signal: EntrySignal) -> str:
        """Generate a unique key for signal deduplication.
        
        Uses symbol, direction, setup type, and a time-bucketed identifier
        to properly distinguish between different setups while still preventing
        rapid-fire duplicate orders on the same setup.
        """
        direction = signal.direction.value if isinstance(signal.direction, Dir) else str(signal.direction)
        setup = signal.setup or "unknown"
        # Bucket entry time to 2-hour windows for dedup (prevents same setup spam)
        entry_ts = signal.entry_time
        if hasattr(entry_ts, "timestamp"):
            # Round to 2-hour buckets
            bucket = int(entry_ts.timestamp() // 7200) * 7200
        else:
            bucket = 0
        # Include zone center for additional uniqueness
        zone_id = ""
        if signal.entry_zone:
            zone_center = (signal.entry_zone[0] + signal.entry_zone[1]) / 2
            zone_id = f":{zone_center:.2f}"
        return f"{symbol}:{direction}:{setup}:{bucket}{zone_id}"

    @staticmethod
    def _signal_key_simple(symbol: str, signal: EntrySignal) -> str:
        """Simple key for active position tracking (symbol + direction only)."""
        direction = signal.direction.value if isinstance(signal.direction, Dir) else str(signal.direction)
        return f"{symbol}:{direction}"

    def _already_executed(self, symbol: str, signal: EntrySignal) -> bool:
        """Check if this signal was already executed recently.
        
        Uses two-level check:
        1. Simple key check against active positions (prevents multiple positions same direction)
        2. Full key check against recent execution history (prevents rapid duplicate orders)
        """
        # Level 1: Check if we already have an active position in this direction
        simple_key = self._signal_key_simple(symbol, signal)
        if simple_key in self._active_signal_keys:
            return True
        
        # Level 2: Check recent execution history for same setup within dedup window
        full_key = self._signal_key(symbol, signal)
        dedup_window = getattr(self.cfg, "signal_dedup_window_minutes", 120)
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(minutes=dedup_window)
        
        for record in reversed(self._execution_history[-100:]):
            rec_key = record.get("key", "")
            # Check if keys match (including setup and zone)
            if rec_key == full_key:
                ts_str = record.get("timestamp")
                if ts_str:
                    try:
                        rec_time = datetime.fromisoformat(ts_str)
                        if rec_time.tzinfo is None:
                            rec_time = rec_time.replace(tzinfo=timezone.utc)
                        if rec_time > cutoff:
                            return True
                    except Exception:
                        pass
            # Also block if same symbol+direction executed very recently (within 5 minutes)
            rec_symbol = record.get("symbol")
            rec_dir = record.get("direction")
            rec_dir_val = rec_dir.value if isinstance(rec_dir, Dir) else str(rec_dir) if rec_dir else ""
            signal_dir_val = signal.direction.value if isinstance(signal.direction, Dir) else str(signal.direction)
            if rec_symbol == symbol and rec_dir_val == signal_dir_val:
                ts_str = record.get("timestamp")
                if ts_str:
                    try:
                        rec_time = datetime.fromisoformat(ts_str)
                        if rec_time.tzinfo is None:
                            rec_time = rec_time.replace(tzinfo=timezone.utc)
                        if rec_time > now - timedelta(minutes=5):
                            return True
                    except Exception:
                        pass
        return False

    def _record_execution(self, symbol: str, signal: EntrySignal, result: dict) -> None:
        record = {
            "key": self._signal_key(symbol, signal),
            "simple_key": self._signal_key_simple(symbol, signal),
            "symbol": symbol,
            "direction": signal.direction,
            "setup": signal.setup,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "signal_time": signal.entry_time.isoformat() if signal.entry_time else None,
        }
        ticket = result.get("deal") or result.get("order") or result.get("position")
        if ticket:
            record["ticket"] = ticket
        self._execution_history.append(record)
        self._order_store.append_execution(record)
        if ticket:
            self._record_active_ticket(ticket, record["key"], symbol)

    def _direction_from_position(self, position: dict) -> Optional[Dir]:
        try:
            pos_type = int(position.get("type")) if position.get("type") is not None else None
        except Exception:
            pos_type = None
        if pos_type in {0}:  # buy
            return Dir.UP
        if pos_type in {1}:  # sell
            return Dir.DOWN
        try:
            volume = float(position.get("volume", 0.0) or 0.0)
        except (TypeError, ValueError):
            return None
        if volume > 0:
            return Dir.UP
        if volume < 0:
            return Dir.DOWN
        return None

    def _direction_from_order(self, order: dict) -> Optional[Dir]:
        direction = order.get("direction")
        if isinstance(direction, Dir):
            return direction
        if isinstance(direction, str):
            try:
                return Dir(direction)
            except Exception:
                pass
        order_type = order.get("type")
        try:
            order_type_int = int(order_type)
        except Exception:
            return None
        if order_type_int in {0, 2, 4, 6}:  # buy / buy-limit / buy-stop / buy-stop-limit
            return Dir.UP
        if order_type_int in {1, 3, 5, 7}:  # sell / sell-limit / sell-stop / sell-stop-limit
            return Dir.DOWN
        return None

    def _timestamp_to_iso(self, ts: Any) -> Optional[str]:
        try:
            if ts is None:
                return None
            val = float(ts)
            if val > 1e12:  # millis
                val = val / 1000.0
            return datetime.fromtimestamp(val, tz=timezone.utc).isoformat()
        except Exception:
            return None

    def _position_key_from_entry(self, position: dict) -> Optional[str]:
        symbol = position.get("symbol")
        direction = self._direction_from_position(position)
        if not symbol or direction is None:
            return None
        return f"{symbol}:{direction.value}"

    def _record_active_ticket(self, ticket: Any, key: str, symbol: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        self._active_store[str(ticket)] = {"key": key, "symbol": symbol, "last_seen": now, "opened": now}
        self._order_store.upsert_active(str(ticket), self._active_store[str(ticket)])

    def _sync_order_store_from_mt5(self, positions: Optional[List[dict]] = None) -> None:
        if positions is None:
            positions = self.adapter.get_positions()
        orders = self.adapter.get_orders() if hasattr(self.adapter, "get_orders") else []
        now_iso = datetime.now(timezone.utc).isoformat()
        existing_exec_tickets: set[str] = {
            str(rec.get("ticket"))
            for rec in self._execution_history
            if rec.get("ticket") is not None
        }
        seen_tickets: set[str] = set()

        def record_active(ticket: str, key: str, symbol: str, opened_iso: Optional[str]) -> None:
            entry = {
                "key": key,
                "symbol": symbol,
                "last_seen": now_iso,
                "opened": opened_iso or now_iso,
            }
            self._active_store[ticket] = entry
            self._order_store.upsert_active(ticket, entry)

        def maybe_append_execution(ticket: str, key: str, symbol: str, ts_iso: Optional[str]) -> None:
            if not ticket or ticket in existing_exec_tickets:
                return
            record = {
                "key": key,
                "symbol": symbol,
                "direction": key.split(":")[-1],
                "timestamp": ts_iso or now_iso,
                "ticket": ticket,
            }
            self._execution_history.append(record)
            self._order_store.append_execution(record)
            existing_exec_tickets.add(ticket)

        for pos in positions or []:
            direction = self._direction_from_position(pos)
            symbol = pos.get("symbol")
            ticket_raw = pos.get("ticket") or pos.get("position") or pos.get("order")
            ticket = str(ticket_raw) if ticket_raw is not None else ""
            if not direction or not symbol or not ticket:
                continue
            key = f"{symbol}:{direction.value}"
            opened_iso = self._timestamp_to_iso(pos.get("time") or pos.get("time_msc")) or now_iso
            record_active(ticket, key, symbol, opened_iso)
            maybe_append_execution(ticket, key, symbol, opened_iso)
            seen_tickets.add(ticket)

        for order in orders or []:
            direction = self._direction_from_order(order)
            symbol = order.get("symbol")
            ticket_raw = order.get("ticket") or order.get("order")
            ticket = str(ticket_raw) if ticket_raw is not None else ""
            if not direction or not symbol or not ticket:
                continue
            key = f"{symbol}:{direction.value}"
            opened_iso = (
                self._timestamp_to_iso(order.get("time_setup"))
                or self._timestamp_to_iso(order.get("time"))
                or now_iso
            )
            record_active(ticket, key, symbol, opened_iso)
            maybe_append_execution(ticket, key, symbol, opened_iso)
            seen_tickets.add(ticket)

        self._prune_active_store(datetime.now(timezone.utc), seen_tickets)

    def _prune_active_store(self, now: datetime, seen_tickets: set[str]) -> None:
        if not self._active_store:
            return
        cutoff = now - self._active_store_max_age
        to_delete = []
        for ticket, entry in self._active_store.items():
            if ticket in seen_tickets:
                continue
            ts = entry.get("last_seen") or entry.get("opened")
            try:
                dt = datetime.fromisoformat(ts)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
            except Exception:
                dt = None
            if dt is None or dt < cutoff:
                to_delete.append(ticket)
        for ticket in to_delete:
            self._active_store.pop(ticket, None)
        if to_delete:
            self._order_store.delete_tickets(to_delete)

    def _update_active_signal_keys(self, positions: Optional[List[dict]] = None) -> None:
        if positions is None:
            positions = self.adapter.get_positions()
        keys: set[str] = set()
        seen_tickets: set[str] = set()
        now = datetime.now(timezone.utc)
        for position in positions or []:
            key = self._position_key_from_entry(position)
            ticket = position.get("ticket") or position.get("position") or position.get("order")
            if key:
                keys.add(key)
            if ticket and key:
                seen_tickets.add(str(ticket))
                self._active_store[str(ticket)] = {
                    "key": key,
                    "symbol": position.get("symbol"),
                    "last_seen": now.isoformat(),
                }
        # Fallback auf Historie, falls Positionsabruf nichts liefert (z. B. Adapter-Fehler)
        if not keys and self._history_active_keys:
            keys |= self._history_active_keys
        # Entferne veraltete oder geschlossene Tickets
        self._prune_active_store(now, seen_tickets)
        if not keys:
            # nutze aktive Store-EintrÃ¤ge (jÃ¼ngst gesehen) als letzte Sicherung gegen Duplikate
            keys |= {entry.get("key") for entry in self._active_store.values() if entry.get("key")}
        self._active_signal_keys = keys
    
    def _signal_passes_validation(self, signal: EntrySignal) -> bool:
        if signal.entry_zone is None:
            return False
        if signal.entry_time is None:
            return False
        if self.cfg.use_ml_filters:
            shift = self.cfg.ml_threshold_shift + self._dynamic_ml_shift
            threshold = self.cfg.ml_probability_threshold + shift
            probability = float(signal.confidence or 0.0)
            if probability < threshold:
                return False
        return True

    def _log_skipped(self, symbol: str, reason: str, required: Optional[float], actual: Optional[float]) -> None:
        parts = [reason]
        if actual is not None and required is not None:
            parts.append(f"Abstand {actual:.6f} < {required:.6f}")
        logger.info(f"[{symbol}] Order Ã¼bersprungen: {' | '.join(parts)}")

    def _profit_factor_ok(self, entry: float, stop_price: float, take_profit: float) -> Tuple[bool, float]:
        min_factor = max(0.0, self.cfg.min_profit_factor)
        if min_factor <= 0:
            return True, float("inf")
        stop_distance = abs(entry - stop_price)
        tp_distance = abs(take_profit - entry)
        if stop_distance <= 0:
            return False, 0.0
        factor = tp_distance / stop_distance if stop_distance > 0 else 0.0
        return factor >= min_factor, factor

    def _validate_stop_distance(self, entry_price: float, stop_price: float) -> Tuple[bool, float, float]:
        min_pct = max(0.0, self.cfg.min_stop_distance_pct)
        if min_pct <= 0 or not math.isfinite(entry_price) or not math.isfinite(stop_price):
            return True, 0.0, 0.0
        baseline = max(abs(entry_price), 1e-9)
        required = baseline * min_pct
        actual = abs(entry_price - stop_price)
        return actual >= required, required, actual

    def _within_exposure_limit(self, symbol: str, additional_exposure: float) -> bool:
        limit_pct = max(0.0, self.cfg.max_gross_exposure_pct)
        if limit_pct <= 0:
            return True
        balance = self._refresh_account_balance()
        allowed = balance * limit_pct
        current = self._current_gross_exposure()
        projected = current + max(0.0, additional_exposure)
        if projected > allowed:
            projected_pct = (projected / balance * 100) if balance > 0 else 0.0
            logger.info(
                f"[{symbol}] Signal Ã¼bersprungen: Exponierung {projected:.2f} > Limit {allowed:.2f} (max {limit_pct*100:.2f}% vom Konto, aktuell {projected_pct:.2f}% vom Konto, Basis {self.cfg.exposure_basis})"
            )
            return False
        return True

    def _current_gross_exposure(self) -> float:
        positions = self.adapter.get_positions()
        total = 0.0
        for pos in positions:
            symbol = pos.get("symbol")
            volume = abs(float(pos.get("volume", 0.0) or 0.0))
            price = float(pos.get("price_current") or pos.get("price_open") or 0.0)
            if not symbol or volume <= 0 or price <= 0:
                continue
            info = self.adapter.get_symbol_info(symbol)
            total += self._exposure_value(symbol, volume, price, info)
        return total

    def _notional_value(self, symbol: str, volume: float, price: float, info: Optional[Any]) -> float:
        if not symbol or volume <= 0 or price <= 0:
            return 0.0
        contract_size = getattr(info, "trade_contract_size", None) or getattr(info, "contract_size", None) or 1.0
        return abs(volume) * float(contract_size) * price

    def _current_leverage(self) -> float:
        if self._account_leverage and self._account_leverage > 0:
            return self._account_leverage
        if self.cfg.exposure_default_leverage and self.cfg.exposure_default_leverage > 0:
            return self.cfg.exposure_default_leverage
        return 1.0

    def _exposure_factor(self) -> float:
        basis = (self.cfg.exposure_basis or "notional").lower()
        if basis == "margin":
            leverage = self._current_leverage()
            return 1.0 / max(leverage, 1.0)
        if basis == "custom":
            return max(self.cfg.exposure_custom_factor, 0.0)
        return 1.0

    def _exposure_value(self, symbol: str, volume: float, price: float, info: Optional[Any]) -> float:
        base = self._notional_value(symbol, volume, price, info)
        return base * self._exposure_factor()

    def _notify_webhook(
        self,
        symbol: str,
        signal: EntrySignal,
        result: dict,
        volume: float,
        risk_amount: float,
        risk_per_lot: float,
        stop_distance: float,
        stop_price: float,
        take_profit: float,
        entry_price: float,
        order_mode: str,
    ) -> None:
        url = self.cfg.webhook_url
        if not url:
            return
        if result.get("retcode") != self.SUCCESS_RETCODE:
            return
        timestamp = datetime.utcnow().replace(tzinfo=timezone.utc).isoformat()
        embed = {
            "title": f"Live Trade ausgefÃ¼hrt: {symbol} {signal.direction.value}",
            "color": 0x1abc9c,
            "fields": [
                {"name": "Setup", "value": signal.setup, "inline": True},
                {"name": "Volume", "value": f"{volume:.3f} Lots", "inline": True},
                {"name": "Entry", "value": f"{entry_price:.4f}", "inline": True},
                {"name": "Stop", "value": f"{stop_price:.4f}", "inline": True},
                {"name": "TP", "value": f"{take_profit:.4f}", "inline": True},
                {"name": "Order-Modus", "value": order_mode, "inline": True},
                {"name": "Risk", "value": f"{risk_amount:.2f} | per lot {risk_per_lot:.3f}", "inline": True},
                {"name": "Retcode", "value": result.get("retcode_description", str(result.get("retcode"))), "inline": True},
            ],
            "timestamp": timestamp,
        }
        payload = {"username": "EW Live Executor", "embeds": [embed]}
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        context = ssl.create_default_context(cafile=certifi.where())
        try:
            urllib.request.urlopen(request, timeout=5, context=context)
        except HTTPError as exc:
            body = exc.read().decode("utf-8", errors="ignore") if exc.fp else ""
            body = body.strip().replace("\n", " ")
            detail = f"HTTP {exc.code}: {exc.reason}"
            if body:
                detail = f"{detail} | {body}"
            fp = self._webhook_fingerprint or "unknown"
            logger.warning(f"[{symbol}] Webhook fehlgeschlagen ({detail}) [fp={fp}]")
        except Exception as exc:
            fp = self._webhook_fingerprint or "unknown"
            logger.warning(f"[{symbol}] Webhook-Benachrichtigung fehlgeschlagen: {exc} [fp={fp}]")

    @staticmethod
    def _compute_webhook_fingerprint(url: Optional[str]) -> Optional[str]:
        if not url:
            return None
        return hashlib.sha256(url.encode("utf-8")).hexdigest()[:10]

    def _process_execution_result(self, symbol: str, direction: Dir, result: dict) -> bool:
        retcode = result.get("retcode")
        if retcode == self.SUCCESS_RETCODE:
            self._register_successful_trade(symbol)
        self._update_symbol_restrictions(symbol, direction, result)
        if retcode in self.STOP_AFTER_RETCODES:
            return False
        return True

    def _is_direction_allowed(self, symbol: str, direction: Dir) -> bool:
        if not self.cfg.allow_shorts and direction == Dir.DOWN:
            return False
        if direction == Dir.DOWN and symbol in self._long_only_symbols:
            return False
        if direction == Dir.UP and symbol in self._short_only_symbols:
            return False
        return True

    def _update_symbol_restrictions(self, symbol: str, direction: Dir, result: dict) -> None:
        retcode = result.get("retcode")
        if retcode == self.LONG_ONLY_RETCODE and direction == Dir.DOWN:
            self._long_only_symbols.add(symbol)
            self._short_only_symbols.discard(symbol)
            logger.warning(f"[{symbol}] Broker schrÃ¤nkt auf LONG ein (Retcode {retcode})")
        elif retcode == self.SHORT_ONLY_RETCODE and direction == Dir.UP:
            self._short_only_symbols.add(symbol)
            self._long_only_symbols.discard(symbol)
            logger.warning(f"[{symbol}] Broker schrÃ¤nkt auf SHORT ein (Retcode {retcode})")

    def _refresh_account_balance(self) -> float:
        info = self.adapter.get_account_info()
        balance = float(info.get("balance", self.cfg.account_balance)) if info else self.cfg.account_balance
        self.cfg.account_balance = balance
        if info:
            leverage = info.get("leverage")
            if leverage is not None:
                try:
                    leverage_value = float(leverage)
                except (TypeError, ValueError):
                    leverage_value = None
            else:
                leverage_value = None
            if leverage_value and leverage_value > 0:
                self._account_leverage = leverage_value
        if balance > self._highest_balance:
            self._highest_balance = balance
        return balance

    def _drawdown_percent(self, balance: float) -> float:
        if self._highest_balance <= 0:
            return 0.0
        return (balance / self._highest_balance - 1.0) * 100.0

    def report_cycle_metrics(self) -> dict[str, float]:
        balance = self._refresh_account_balance()
        exposure = self._current_gross_exposure()
        drawdown = self._drawdown_percent(balance)
        drawdown = abs(drawdown)
        exposure_pct = (exposure / balance * 100.0) if balance and balance > 0 else 0.0
        return {
            "balance": balance,
            "exposure": exposure,
            "drawdown": drawdown,
            "exposure_pct": exposure_pct,
        }

    def _risk_multiplier_for_dd(self, dd_percent: float) -> float:
        if not self.cfg.dynamic_dd_risk:
            return 1.0
        multiplier = 1.0
        sorted_steps = sorted(self.cfg.dd_risk_steps, key=lambda step: step[0])
        for threshold, value in sorted_steps:
            if dd_percent <= threshold:
                multiplier = value
        return multiplier

    def _vol_multiplier(self) -> float:
        if not self.cfg.use_vol_target or len(self._recent_returns) < 5:
            return 1.0
        try:
            std = pstdev(self._recent_returns)
        except StatisticsError:
            return 1.0
        if std <= 1e-9:
            return 1.0
        scale = self.cfg.target_annual_vol / (std * 4)
        return max(0.4, min(1.6, scale))

    def _dynamic_trend_scale(self, signal: EntrySignal) -> float:
        if not self.cfg.dynamic_trend_scaling:
            return 1.0
        scale = 1.0
        entry_tf = (signal.entry_tf or "").upper()
        scale *= self.cfg.tf_size_factors.get(entry_tf, 1.0)
        setup_key = (signal.setup or "").upper()
        scale *= self.cfg.setup_size_factors.get(setup_key, 1.0)
        return max(0.6, min(scale, 1.5))

    def _record_expected_return(self, entry_price: float, stop_price: float, take_profit: float) -> None:
        if not self.cfg.use_vol_target:
            return
        stop_distance = abs(entry_price - stop_price)
        tp_distance = abs(take_profit - entry_price)
        if stop_distance <= 0 or tp_distance <= 0:
            return
        self._recent_returns.append(tp_distance / stop_distance)

    def _calculate_volume(
        self, symbol: str, signal: EntrySignal, info: Optional[Any], stop_price: float, entry_price: float
    ) -> Tuple[float, float, float, float]:
        stop_distance = abs(entry_price - stop_price)
        balance = self._refresh_account_balance()
        base_risk_amount = max(0.0, self.cfg.risk_per_trade * balance)
        dd_multiplier = self._risk_multiplier_for_dd(self._drawdown_percent(balance))
        vol_multiplier = self._vol_multiplier()
        scaled_risk_amount = base_risk_amount * dd_multiplier * vol_multiplier
        risk_fraction = scaled_risk_amount / max(balance, 1e-9)
        risk_fraction = max(self.cfg.risk_per_trade_min, min(risk_fraction, self.cfg.risk_per_trade_max))
        risk_amount = risk_fraction * balance
        if stop_distance <= 0 or risk_amount <= 0:
            return self.cfg.min_lot, risk_amount, 0.0, stop_distance

        tick_size = getattr(info, "trade_tick_size", None) or getattr(info, "point", 0.0)
        tick_value = getattr(info, "trade_tick_value", None)
        if tick_size and tick_value:
            ticks = stop_distance / tick_size if tick_size else stop_distance
            risk_per_lot = ticks * tick_value
        else:
            contract_size = getattr(info, "trade_contract_size", None)
            point = getattr(info, "point", 1.0)
            risk_per_lot = stop_distance * (contract_size or 1.0) / point if point else stop_distance * (contract_size or 1.0)
        if risk_per_lot <= 0:
            risk_per_lot = stop_distance
        lots = (risk_amount / risk_per_lot) if risk_per_lot else self.cfg.min_lot
        lots *= self.cfg.lot_size_per_risk_point
        prob_scale = 1.0
        if self.cfg.size_by_prob:
            probability = float(signal.confidence or 0.0)
            base_thr = max(0.5, self.cfg.ml_probability_threshold)
            frac = max(0.0, (probability - base_thr) / max(1e-6, 1.0 - base_thr))
            scale = self.cfg.prob_size_min + (self.cfg.prob_size_max - self.cfg.prob_size_min) * frac
            lots *= scale
            prob_scale = scale
        direction_factor = 1.0
        if signal.direction == Dir.DOWN:
            direction_factor = self.cfg.size_short_factor
            lots *= direction_factor
        trend_scale = self._dynamic_trend_scale(signal)
        lots *= trend_scale
        lots = max(self.cfg.min_lot, lots)
        lots = min(lots, self.cfg.max_lot)
        if self.cfg.max_gross_exposure_pct > 0:
            remaining_exposure = self._exposure_limit_allowance(balance)
            lots = self._cap_volume_to_remaining(symbol, info, entry_price, lots, remaining_exposure)
            if lots <= 0:
                return 0.0, 0.0, risk_per_lot, stop_distance
        lots = self._align_with_symbol_constraints(symbol, info, lots)
        risk_amount = risk_per_lot * lots
        exposure_value = self._exposure_value(symbol, lots, entry_price, info)
        self._log_volume_insight(
            symbol,
            signal,
            lots,
            risk_amount,
            exposure_value,
            trend_scale,
            dd_multiplier,
            vol_multiplier,
            prob_scale,
            direction_factor,
            stop_distance,
        )
        return lots, risk_amount, risk_per_lot, stop_distance

    def _exposure_limit_allowance(self, balance: float) -> float:
        limit_pct = max(self.cfg.max_gross_exposure_pct, 0.0)
        if limit_pct <= 0 or balance <= 0:
            return 0.0
        limit_value = balance * limit_pct
        current = self._current_gross_exposure()
        remaining = limit_value - current
        return max(remaining, 0.0)

    def _cap_volume_to_remaining(
        self, symbol: str, info: Optional[Any], entry_price: float, volume: float, remaining: float
    ) -> float:
        if remaining <= 0 or volume <= 0:
            logger.info(f"[{symbol}] Exponierungslimit erreicht; Volumen auf 0 reduziert.")
            return 0.0
        desired_exposure = self._exposure_value(symbol, volume, entry_price, info)
        if desired_exposure <= remaining:
            return volume
        factor = remaining / desired_exposure if desired_exposure > 0 else 0.0
        scaled_volume = max(self.cfg.min_lot, volume * factor)
        scaled_volume = self._align_with_symbol_constraints(symbol, info, scaled_volume, log_changes=False)
        step = getattr(info, "volume_step", None) or 0.01
        for _ in range(40):
            exposure = self._exposure_value(symbol, scaled_volume, entry_price, info)
            if exposure <= remaining:
                return scaled_volume
            scaled_volume = max(self.cfg.min_lot, scaled_volume - step)
            scaled_volume = self._align_with_symbol_constraints(symbol, info, scaled_volume, log_changes=False)
            if scaled_volume <= 0:
                break
        final_exposure = self._exposure_value(symbol, scaled_volume, entry_price, info)
        if final_exposure <= remaining and scaled_volume > 0:
            logger.info(
                f"[{symbol}] Volumen reduziert auf {scaled_volume:.3f}, um das Exponierungslimit von {remaining:.2f} einzuhalten."
            )
            return scaled_volume
        logger.info(f"[{symbol}] Exponierungslimit erreicht; Volumen auf 0 reduziert.")
        return 0.0

    def _align_with_symbol_constraints(
        self, symbol: str, info: Optional[Any], volume: float, log_changes: bool = True
    ) -> float:
        min_vol = self.cfg.min_lot
        max_vol = self.cfg.max_lot
        step = 0.01
        if info is not None:
            info_min = getattr(info, "volume_min", None)
            info_max = getattr(info, "volume_max", None)
            info_step = getattr(info, "volume_step", None)
            if info_min:
                min_vol = max(min_vol, info_min)
            if info_max:
                max_vol = min(max_vol, info_max)
            if info_step:
                step = max(step, info_step)
        if max_vol and volume > max_vol:
            volume = max_vol
        step = max(step, 1e-9)
        normalized = volume
        try:
            steps = math.floor((volume + 1e-9) / step)
            normalized = steps * step
        except Exception:
            normalized = volume
        if normalized < min_vol:
            normalized = min_vol
        if max_vol and normalized > max_vol:
            normalized = max_vol
        normalized = round(normalized, 6)
        if log_changes and abs(normalized - volume) > 1e-9:
            logger.info(
                f"[{symbol}] Volume an Symbol-Limits angepasst: {normalized:.3f} (min={min_vol:.3f}, max={max_vol:.3f}, step={step:.5f})"
            )
        return normalized

    def _orders_limit_reached(self) -> bool:
        limit = getattr(self.cfg, "orders_soft_limit", 0)
        if limit <= 0:
            return False
        total = self.adapter.get_orders_total()
        if total is None:
            return False
        if total >= limit:
            logger.warning(
                f"[orders] Offene/pendende Orders {total} >= Soft-Limit {limit} -> keine weiteren Orders im Zyklus"
            )
            return True
        return False

    def _cooldown_remaining(self, symbol: str) -> Optional[timedelta]:
        cooldown = max(self.cfg.trade_cooldown_minutes, self._adaptive_cooldown_minutes)
        if cooldown <= 0:
            return None
        last_trade = self._last_trade_time.get(symbol)
        if not last_trade:
            return None
        remaining = timedelta(minutes=cooldown) - (datetime.now(timezone.utc) - last_trade)
        return remaining if remaining.total_seconds() > 0 else None

    def _trade_limit_hit(self, symbol: str) -> bool:
        limit = self.cfg.max_trades_per_symbol_per_hour
        if limit <= 0:
            return False
        history = self._prune_trade_history(symbol, timedelta(hours=1))
        return len(history) >= limit

    def _register_successful_trade(self, symbol: str) -> None:
        now = datetime.now(timezone.utc)
        history = self._prune_trade_history(symbol, timedelta(hours=1))
        history.append(now)
        self._last_trade_time[symbol] = now

    def _prune_trade_history(self, symbol: str, window: timedelta) -> Deque[datetime]:
        history = self._recent_trade_times.setdefault(symbol, deque())
        cutoff = datetime.now(timezone.utc) - window
        while history and history[0] < cutoff:
            history.popleft()
        return history

    # =========================================================================
    # Momentum-Exit: Schließe Positionen früh bei adverser Momentum-Bewegung
    # =========================================================================

    def check_momentum_exits(self, momentum_lookback: int = 10, momentum_threshold: float = -0.002) -> int:
        """Prüft alle offenen Positionen auf adverses Momentum und schließt sie ggf.

        Args:
            momentum_lookback: Anzahl Bars für Momentum-Berechnung
            momentum_threshold: Schwellwert für adverses Momentum (negativ = gegen Position)

        Returns:
            Anzahl geschlossener Positionen
        """
        positions = self.adapter.get_positions()
        if not positions:
            return 0
        closed_count = 0
        for pos in positions:
            symbol = pos.get("symbol")
            ticket = pos.get("ticket")
            pos_type = pos.get("type")  # 0 = BUY, 1 = SELL
            volume = float(pos.get("volume", 0.0) or 0.0)
            if not symbol or ticket is None or pos_type is None or volume <= 0:
                continue
            # Berechne Momentum
            momentum = self._calculate_position_momentum(symbol, momentum_lookback)
            if momentum is None:
                continue
            # Prüfe ob Momentum gegen unsere Position läuft
            is_long = pos_type == 0
            adverse_momentum = False
            if is_long and momentum < momentum_threshold:
                # Long-Position aber Preis fällt stark
                adverse_momentum = True
                logger.warning(
                    f"[{symbol}] Momentum-Exit LONG: Momentum {momentum:.4f} < {momentum_threshold:.4f}"
                )
            elif not is_long and momentum > abs(momentum_threshold):
                # Short-Position aber Preis steigt stark
                adverse_momentum = True
                logger.warning(
                    f"[{symbol}] Momentum-Exit SHORT: Momentum {momentum:.4f} > {abs(momentum_threshold):.4f}"
                )
            if adverse_momentum:
                result = self.adapter.close_position(ticket, symbol, volume, pos_type)
                retcode = result.get("retcode")
                if retcode == self.SUCCESS_RETCODE:
                    logger.info(f"[{symbol}] Position {ticket} erfolgreich geschlossen (Momentum-Exit)")
                    closed_count += 1
                    # Entferne aus active_signal_keys damit Symbol wieder frei ist
                    direction = Dir.UP if is_long else Dir.DOWN
                    key = f"{symbol}:{direction.value}"
                    self._active_signal_keys.discard(key)
                else:
                    logger.error(f"[{symbol}] Momentum-Exit fehlgeschlagen: {result}")
        return closed_count

    def _calculate_position_momentum(self, symbol: str, lookback: int) -> Optional[float]:
        """Berechnet das Momentum für ein Symbol basierend auf letzten Bars.

        Returns:
            Momentum als prozentuale Änderung (positiv = steigend, negativ = fallend)
        """
        rates = self.adapter.get_rates(symbol, self.cfg.timeframe, lookback + 1)
        if not rates or len(rates) < 2:
            return None
        closes = []
        for row in rates:
            close_val = row.get("close") if isinstance(row, dict) else None
            try:
                closes.append(float(close_val))
            except (TypeError, ValueError):
                continue
        if len(closes) < 2:
            return None
        # Momentum = (aktueller Close - Close vor N bars) / Close vor N bars
        first_close = closes[0]
        last_close = closes[-1]
        if first_close <= 0:
            return None
        return (last_close - first_close) / first_close