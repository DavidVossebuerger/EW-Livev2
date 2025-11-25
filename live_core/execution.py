"""Order-Manager für das Live-System, trifft Entscheidungen basierend auf Signalen."""
from __future__ import annotations

import json
import logging
import math
import urllib.request
from collections import deque
from datetime import datetime, timezone
from statistics import pstdev, StatisticsError
from typing import Any, Dict, Deque, List, Optional, Tuple

from .config import LiveConfig
from .mt5_adapter import MetaTrader5Adapter
from .signals import Dir, EntrySignal

logger = logging.getLogger("ew_live")


class OrderManager:
    SUCCESS_RETCODE = 10009
    STOP_AFTER_RETCODES = {10016, 10017, 10018}
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

    def evaluate_signals(self, symbol: str, signals: List[EntrySignal]) -> None:
        if not signals:
            return
        if not self.adapter.connected:
            raise RuntimeError("MT5 nicht verbunden")
        existing_positions = self.adapter.get_positions(symbol)
        self._try_move_stop_to_breakeven(symbol, existing_positions)
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
            info = self.adapter.get_symbol_info(symbol)
            volume, risk_amount, risk_per_lot, stop_distance = self._calculate_volume(symbol, signal, info)
            direction = signal.direction.value if isinstance(signal.direction, Dir) else signal.direction
            try:
                result = self.adapter.place_market_order(
                    symbol=symbol,
                    volume=volume,
                    direction=direction,
                    sl=signal.stop_loss,
                    tp=signal.take_profit,
                )
                self._log_order(symbol, signal, result, volume, risk_amount, risk_per_lot, stop_distance)
                should_continue = self._process_execution_result(symbol, signal.direction, result)
                if result.get("retcode") == self.SUCCESS_RETCODE:
                    self._last_confidence[symbol] = confidence
                if not should_continue:
                    break
            except Exception as exc:
                logger.error(f"[{symbol}] Fehler beim Platzieren der Order: {exc}")

    def _log_order(
        self,
        symbol: str,
        signal: EntrySignal,
        result: dict,
        volume: float,
        risk_amount: float,
        risk_per_lot: float,
        stop_distance: float,
    ) -> None:
        logger.info(
            f"[{symbol}] Order {signal.setup} {signal.direction} @ {signal.entry_price} "
            f"SL={signal.stop_loss:.2f} TP={signal.take_profit:.2f} "
            f"Vol={volume:.3f} (risk={risk_amount:.2f}, stop={stop_distance:.5f},/lot={risk_per_lot:.3f}) -> {result}"
        )

    def _try_move_stop_to_breakeven(self, symbol: str, positions: Optional[List[dict]] = None) -> None:
        if not self.cfg.enable_breakeven:
            return
        pos_list = positions if positions is not None else self.adapter.get_positions(symbol)
        if not pos_list:
            return
        tick = self.adapter.get_symbol_tick(symbol)
        if not tick:
            return
        ratio = max(0.0, min(0.9999, self.cfg.breakeven_tp_ratio))
        if ratio <= 0.0:
            return
        buffer = max(0.0, self.cfg.breakeven_sl_buffer)
        for pos in pos_list:
            ticket = pos.get("ticket")
            if ticket is None:
                continue
            try:
                direction = Dir.UP if int(pos.get("type", 0)) == 0 else Dir.DOWN
            except (TypeError, ValueError):
                continue
            entry = float(pos.get("price_open") or 0.0)
            tp = float(pos.get("tp") or 0.0)
            sl = float(pos.get("sl") or 0.0)
            if entry <= 0.0 or tp <= 0.0:
                continue
            tp_distance = tp - entry if direction == Dir.UP else entry - tp
            if tp_distance <= 0.0:
                continue
            trigger_price = entry + ratio * tp_distance if direction == Dir.UP else entry - ratio * tp_distance
            current_price = tick.get("bid") if direction == Dir.UP else tick.get("ask")
            if current_price is None:
                continue
            if direction == Dir.UP and current_price < trigger_price:
                continue
            if direction == Dir.DOWN and current_price > trigger_price:
                continue
            target_sl = entry - buffer if direction == Dir.UP else entry + buffer
            already_be = (direction == Dir.UP and sl >= target_sl - 1e-6) or (
                direction == Dir.DOWN and sl <= target_sl + 1e-6
            )
            if already_be:
                continue
            result = self.adapter.modify_position_sl_tp(int(ticket), symbol, target_sl, tp)
            success = result.get("retcode") == self.SUCCESS_RETCODE or result.get("status") == "mock"
            if success:
                logger.info(
                    f"[{symbol}] SL auf Break-even gesetzt (Ticket={ticket}, Preis={target_sl:.5f})"
                )
            else:
                logger.warning(
                    f"[{symbol}] SL-Break-even für Ticket {ticket} fehlgeschlagen: "
                    f"{result.get('retcode_description', result)}"
                )
        self._log_adjustment(symbol, result)
        self._log_filling_mode(symbol, result)
        self._log_retcode_info(symbol, result)
        self._notify_webhook(symbol, signal, result, volume, risk_amount, risk_per_lot, stop_distance)
        self._record_expected_return(signal)

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

    def _log_skipped(self, symbol: str, reason: str, required: Optional[float], actual: Optional[float]) -> None:
        parts = [reason]
        if actual is not None and required is not None:
            parts.append(f"Abstand {actual:.6f} < {required:.6f}")
        logger.info(f"[{symbol}] Order übersprungen: {' | '.join(parts)}")

    def _notify_webhook(
        self,
        symbol: str,
        signal: EntrySignal,
        result: dict,
        volume: float,
        risk_amount: float,
        risk_per_lot: float,
        stop_distance: float,
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
                {"name": "Entry", "value": f"{signal.entry_price:.4f}", "inline": True},
                {"name": "Stop", "value": f"{signal.stop_loss:.4f}", "inline": True},
                {"name": "TP", "value": f"{signal.take_profit:.4f}", "inline": True},
                {"name": "Risk", "value": f"{risk_amount:.2f} | per lot {risk_per_lot:.3f}", "inline": True},
                {"name": "Retcode", "value": result.get("retcode_description", str(result.get("retcode"))), "inline": True},
            ],
            "timestamp": timestamp,
        }
        payload = {"username": "EW Live Executor", "embeds": [embed]}
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        try:
            urllib.request.urlopen(request, timeout=5)
        except Exception as exc:
            logger.warning(f"[{symbol}] Webhook-Benachrichtigung fehlgeschlagen: {exc}")

    def _process_execution_result(self, symbol: str, direction: Dir, result: dict) -> bool:
        retcode = result.get("retcode")
        self._update_symbol_restrictions(symbol, direction, retcode)
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

    def _update_symbol_restrictions(self, symbol: str, direction: Dir, retcode: Optional[int]) -> None:
        if retcode == 10016:
            self._long_only_symbols.add(symbol)
            self._short_only_symbols.discard(symbol)
            logger.warning(f"[{symbol}] Broker schränkt auf LONG ein (Retcode 10016)")
        elif retcode == 10017:
            self._short_only_symbols.add(symbol)
            self._long_only_symbols.discard(symbol)
            logger.warning(f"[{symbol}] Broker schränkt auf SHORT ein (Retcode 10017)")

    def _refresh_account_balance(self) -> float:
        info = self.adapter.get_account_info()
        balance = float(info.get("balance", self.cfg.account_balance)) if info else self.cfg.account_balance
        self.cfg.account_balance = balance
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

    def _record_expected_return(self, signal: EntrySignal) -> None:
        if not self.cfg.use_vol_target:
            return
        stop_distance = abs(signal.entry_price - signal.stop_loss)
        tp_distance = abs(signal.take_profit - signal.entry_price)
        if stop_distance <= 0 or tp_distance <= 0:
            return
        self._recent_returns.append(tp_distance / stop_distance)

    def _calculate_volume(
        self, symbol: str, signal: EntrySignal, info: Optional[Any]
    ) -> Tuple[float, float, float, float]:
        entry_price = signal.entry_price
        stop_price = signal.stop_loss
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
        lots = self._align_with_symbol_constraints(symbol, info, lots)
        return lots, risk_amount, risk_per_lot, stop_distance

    def _align_with_symbol_constraints(
        self, symbol: str, info: Optional[Any], volume: float
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
        if abs(normalized - volume) > 1e-9:
            logger.info(
                f"[{symbol}] Volume an Symbol-Limits angepasst: {normalized:.3f} (min={min_vol:.3f}, max={max_vol:.3f}, step={step:.5f})"
            )
        return normalized
