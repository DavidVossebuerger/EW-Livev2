"""Order-Manager für das Live-System, trifft Entscheidungen basierend auf Signalen."""
from __future__ import annotations

import hashlib
import json
import logging
import math
import ssl
import urllib.request
from collections import defaultdict, deque
from datetime import datetime, timezone, timedelta
from pathlib import Path
from statistics import pstdev, StatisticsError
from typing import Any, Dict, Deque, List, Optional, Tuple
from urllib.error import HTTPError

import certifi

from .config import LiveConfig
from .mt5_adapter import MetaTrader5Adapter
from .signals import Dir, EntrySignal

logger = logging.getLogger("ew_live")


class OrderManager:
    SUCCESS_RETCODE = 10009
    STOP_AFTER_RETCODES = {10017, 10018}
    LONG_ONLY_RETCODE = 10042
    SHORT_ONLY_RETCODE = 10043
    RETCODE_HINTS = {
        10007: "Invalid stops - Abstand liegt unter `trade_stops_level` oder Mindestabstand.",
        10030: "Unsupported filling mode - Broker akzeptiert andere `order_filling_mode`-Typen.",
    }

    def __init__(self, mt5_adapter: MetaTrader5Adapter, cfg: LiveConfig):
        self.adapter = mt5_adapter
        self.cfg = cfg
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
        if self._webhook_fingerprint:
            logger.info(f"Webhook aktiviert (Fingerprint={self._webhook_fingerprint})")

    def evaluate_signals(self, symbol: str, signals: List[EntrySignal]) -> None:
        if not signals:
            return
        signals = [s for s in signals if not self._already_executed(symbol, s)]
        if not signals:
            return
        if not self.adapter.connected:
            raise RuntimeError("MT5 nicht verbunden")
        existing_positions = self.adapter.get_positions(symbol)
        limited_signals = signals[: self.cfg.max_open_trades]
        for idx, signal in enumerate(limited_signals):
            open_positions = existing_positions if idx == 0 else self.adapter.get_positions(symbol)
            if self.cfg.use_ml_filters:
                threshold = self.cfg.ml_probability_threshold + self.cfg.ml_threshold_shift
                confidence = float(signal.confidence or 0.0)
                if confidence < threshold:
                    logger.info(
                        f"[{symbol}] Signal {signal.setup} {signal.direction} übersprungen: Confidence {confidence:.3f}"
                        f" < Threshold {threshold:.3f}"
                    )
                    continue
            if not self._is_direction_allowed(symbol, signal.direction):
                logger.info(f"[{symbol}] Signal {signal.direction} übersprungen: Broker untersagt diese Richtung")
                continue
            # Überprüfe offene Positionen und Confidence-Steigerung
            confidence = float(signal.confidence or 0.0)
            if open_positions:
                last_conf = self._last_confidence.get(symbol, 0.0)
                if confidence < last_conf + 0.07:
                    logger.info(
                        f"[{symbol}] Signal {signal.setup} {signal.direction} übersprungen: Offene Position vorhanden, "
                        f"Confidence {confidence:.3f} nicht um 7% gestiegen (letzte: {last_conf:.3f})"
                    )
                    continue
            if self._trade_limit_hit(symbol):
                logger.info(
                    f"[{symbol}] Signal {signal.setup} {signal.direction} übersprungen: Max "
                    f"{self.cfg.max_trades_per_symbol_per_hour} Trades/Std erreicht"
                )
                continue
            cooldown_remaining = self._cooldown_remaining(symbol)
            if cooldown_remaining is not None:
                minutes = cooldown_remaining.total_seconds() / 60.0
                logger.info(
                    f"[{symbol}] Signal {signal.setup} {signal.direction} übersprungen: Cooldown aktiv "
                    f"({minutes:.1f}m verbleibend)"
                )
                continue
            info = self.adapter.get_symbol_info(symbol)
            if info is None:
                logger.warning(f"[{symbol}] Keine Symbolinformationen verfügbar -> Signal übersprungen")
                continue
            stop_price, take_profit = self._scale_order_levels(signal)
            pf_ok, pf_value = self._profit_factor_ok(signal.entry_price, stop_price, take_profit)
            if not pf_ok:
                logger.info(
                    f"[{symbol}] Signal {signal.setup} {signal.direction} übersprungen: Chance/Risiko {pf_value:.2f} < "
                    f"Mindestfaktor {self.cfg.min_profit_factor:.2f}"
                )
                continue
            current_price = self._current_price(symbol, signal.direction)
            if current_price is None:
                logger.warning(f"[{symbol}] Kein aktueller Preis verfügbar -> Signal übersprungen")
                continue
            execution_price = current_price
            pending_price = self._pending_limit_price(signal)
            use_pending_order = False
            if self.cfg.use_pending_orders and pending_price is not None:
                if self._pending_price_allowed(pending_price, current_price, signal.direction):
                    execution_price = pending_price
                    use_pending_order = True
            if not self._price_supports_order(symbol, signal.direction, execution_price, stop_price, take_profit):
                continue
            volume, risk_amount, risk_per_lot, stop_distance = self._calculate_volume(
                symbol, signal, info, stop_price, execution_price
            )
            if volume <= 0:
                logger.info(
                    f"[{symbol}] Signal {signal.setup} {signal.direction} übersprungen: Exposure-Limit lässt kein Volumen zu"
                )
                continue
            if self._duplicate_position_present(open_positions, signal.direction, execution_price):
                logger.info(
                    f"[{symbol}] Signal {signal.setup} {signal.direction} übersprungen: Position bei {execution_price:.5f} bereits vorhanden"
                )
                continue
            trade_exposure = self._exposure_value(symbol, volume, execution_price, info)
            if not self._within_exposure_limit(symbol, trade_exposure):
                continue
            direction = signal.direction.value if isinstance(signal.direction, Dir) else signal.direction
            try:
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
                if result.get("retcode") == self.SUCCESS_RETCODE:
                    self._record_execution(symbol, signal)
                    self._last_confidence[symbol] = confidence
                if not should_continue:
                    break
            except Exception as exc:
                logger.error(f"[{symbol}] Fehler beim Platzieren der Order: {exc}")

    def _current_price(self, symbol: str, direction: Dir) -> Optional[float]:
        tick = self.adapter.get_symbol_tick(symbol)
        if not tick:
            return None
        price = tick.get("ask") if direction == Dir.UP else tick.get("bid")
        if price is None or price <= 0:
            return None
        return float(price)

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
                    f"[{symbol}] Signal {direction} übersprungen: aktueller Preis {price:.5f} <= Stop {stop_price:.5f}"
                )
                return False
            if price >= take_profit + margin - eps:
                logger.info(
                    f"[{symbol}] Signal {direction} übersprungen: aktueller Preis {price:.5f} >= TP {take_profit:.5f}"
                )
                return False
        else:
            if price >= stop_price + margin - eps:
                logger.info(
                    f"[{symbol}] Signal {direction} übersprungen: aktueller Preis {price:.5f} >= Stop {stop_price:.5f}"
                )
                return False
            if price <= take_profit - margin + eps:
                logger.info(
                    f"[{symbol}] Signal {direction} übersprungen: aktueller Preis {price:.5f} <= TP {take_profit:.5f}"
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
            reason = result.get("reason", "Stop nicht genügend Abstand")
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

    def _resolve_store_path(self) -> Path:
        order_store_path = self.cfg.order_store_path
        if not order_store_path:
            raise RuntimeError("order_store_path is not configured, but execution history is required")
        path = Path(order_store_path)
        if not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def _load_execution_history(self) -> list[dict]:
        path = self._resolve_store_path()
        if not path.exists():
            return []
        try:
            with path.open("r", encoding="utf-8") as fh:
                return json.load(fh)
        except json.JSONDecodeError:
            logger.warning("Unerwartetes Format in order store, beginne mit leerer Historie")
            return []

    def _save_execution_history(self) -> None:
        path = self._resolve_store_path()
        with path.open("w", encoding="utf-8") as fh:
            json.dump(self._execution_history, fh, ensure_ascii=False, indent=2)

    @staticmethod
    def _signal_key(symbol: str, signal: EntrySignal) -> str:
        return f"{symbol}:{signal.direction}"

    def _already_executed(self, symbol: str, signal: EntrySignal) -> bool:
        key = self._signal_key(symbol, signal)
        return any(entry.get("key") == key for entry in self._execution_history)

    def _record_execution(self, symbol: str, signal: EntrySignal) -> None:
        self._execution_history.append({
            "key": self._signal_key(symbol, signal),
            "symbol": symbol,
            "direction": signal.direction,
            "timestamp": signal.entry_time.isoformat()
        })
        self._save_execution_history()

    def _log_skipped(self, symbol: str, reason: str, required: Optional[float], actual: Optional[float]) -> None:
        parts = [reason]
        if actual is not None and required is not None:
            parts.append(f"Abstand {actual:.6f} < {required:.6f}")
        logger.info(f"[{symbol}] Order übersprungen: {' | '.join(parts)}")

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
                f"[{symbol}] Signal übersprungen: Exponierung {projected:.2f} > Limit {allowed:.2f} (max {limit_pct*100:.2f}% vom Konto, aktuell {projected_pct:.2f}% vom Konto, Basis {self.cfg.exposure_basis})"
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
            "title": f"Live Trade ausgeführt: {symbol} {signal.direction.value}",
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
            logger.warning(f"[{symbol}] Broker schränkt auf LONG ein (Retcode {retcode})")
        elif retcode == self.SHORT_ONLY_RETCODE and direction == Dir.UP:
            self._short_only_symbols.add(symbol)
            self._long_only_symbols.discard(symbol)
            logger.warning(f"[{symbol}] Broker schränkt auf SHORT ein (Retcode {retcode})")

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
        if self.cfg.size_by_prob:
            probability = float(signal.confidence or 0.0)
            base_thr = max(0.5, self.cfg.ml_probability_threshold)
            frac = max(0.0, (probability - base_thr) / max(1e-6, 1.0 - base_thr))
            scale = self.cfg.prob_size_min + (self.cfg.prob_size_max - self.cfg.prob_size_min) * frac
            lots *= scale
        if signal.direction == Dir.DOWN:
            lots *= self.cfg.size_short_factor
        lots = max(self.cfg.min_lot, lots)
        lots = min(lots, self.cfg.max_lot)
        if self.cfg.max_gross_exposure_pct > 0:
            remaining_exposure = self._exposure_limit_allowance(balance)
            lots = self._cap_volume_to_remaining(symbol, info, entry_price, lots, remaining_exposure)
            if lots <= 0:
                return 0.0, 0.0, risk_per_lot, stop_distance
        lots = self._align_with_symbol_constraints(symbol, info, lots)
        risk_amount = risk_per_lot * lots
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
        if volume < min_vol:
            normalized = min_vol
        else:
            step = max(step, 1e-9)
            steps = math.floor(volume / step)
            normalized = steps * step
            if normalized < min_vol:
                normalized = min_vol
            if max_vol and normalized > max_vol:
                normalized = max_vol
        if log_changes and abs(normalized - volume) > 1e-9:
            logger.info(
                f"[{symbol}] Volume an Symbol-Limits angepasst: {normalized:.3f} (min={min_vol:.3f}, max={max_vol:.3f}, step={step:.5f})"
            )
        return normalized

    def _cooldown_remaining(self, symbol: str) -> Optional[timedelta]:
        cooldown = self.cfg.trade_cooldown_minutes
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
