"""Wrapper um MetaTrader5, damit wir live Orders ausführen können."""
from __future__ import annotations

import math
import os
from typing import Any, Dict, List, Optional

import pandas as pd

try:
    import MetaTrader5 as mt5
except ImportError:  # pragma: no cover
    mt5 = None  # type: ignore


class MetaTrader5Adapter:
    """Minimaler Adapter für Zugang zu MT5 (oder Mock bei installiertem MT5)."""

    _MIN_STOP_POINTS = 10
    _DISTANCE_EPS = 1e-6
    _SKIPPED_INVALID_STOP = "skipped-invalid-stop"
    _UNSUPPORTED_FILLING_RETCODE = 10030
    RETCODE_DESCRIPTIONS = {
        10004: "TRADE_RETCODE_REQUOTE - Requote erhalten",
        10006: "TRADE_RETCODE_REJECT - Order vom Server abgelehnt",
        10009: "TRADE_RETCODE_DONE - Order erfolgreich",
        10010: "TRADE_RETCODE_DONE_PARTIAL - Order teilweise gefüllt",
        10011: "TRADE_RETCODE_ERROR - Allgemeiner Trading-Fehler",
        10014: "TRADE_RETCODE_INVALID_VOLUME - Volumen ungültig",
        10015: "TRADE_RETCODE_INVALID_PRICE - Preis ungültig",
        10016: "TRADE_RETCODE_INVALID_STOPS - Stops ungültig",
        10017: "TRADE_RETCODE_TRADE_DISABLED - Symbol nicht handelbar",
        10018: "TRADE_RETCODE_MARKET_CLOSED - Markt geschlossen",
        10019: "TRADE_RETCODE_NO_MONEY - Nicht genug Margin",
        10020: "TRADE_RETCODE_PRICE_CHANGED - Preis hat sich verändert",
        10021: "TRADE_RETCODE_PRICE_OFF - Kein aktueller Preis",
        10030: "TRADE_RETCODE_INVALID_FILL - Filling-Mode nicht erlaubt",
        10031: "TRADE_RETCODE_CONNECTION - Verbindungsproblem",
        10042: "TRADE_RETCODE_LONG_ONLY - Nur Long erlaubt",
        10043: "TRADE_RETCODE_SHORT_ONLY - Nur Short erlaubt",
        10044: "TRADE_RETCODE_CLOSE_ONLY - Nur Schliessen erlaubt",
    }

    def __init__(self, login: Optional[int] = None, password: Optional[str] = None, server: Optional[str] = None):
        self.login = login or int(os.environ.get("EW_MT5_LOGIN", "0"))
        self.password = password or os.environ.get("EW_MT5_PASSWORD", "")
        self.server = server or os.environ.get("EW_MT5_SERVER", "")
        self.connected = False
        self._mock = mt5 is None
        self._symbol_cache: dict[str, "Any"] = {}

    def _ensure_symbol_selected(self, symbol: str) -> None:
        if self._mock:
            return
        info = mt5.symbol_info(symbol)
        if info and getattr(info, "select", False):
            return
        if not mt5.symbol_select(symbol, True):
            raise RuntimeError(f"Symbol {symbol} konnte nicht aktiviert werden: {mt5.last_error()}")

    def connect(self) -> None:
        if self._mock:
            self.connected = True
            return
        if not mt5.initialize(login=self.login, password=self.password, server=self.server):
            raise ConnectionError(mt5.last_error())
        self.connected = True

    def disconnect(self) -> None:
        if self.connected and mt5 is not None:
            mt5.shutdown()
        self.connected = False

    def get_account_info(self) -> Optional[Dict[str, Any]]:
        if self._mock:
            return None
        try:
            info = mt5.account_info()
            if info is None:
                return None
            return {
                'login': info.login,
                'balance': float(info.balance),
                'equity': float(info.equity),
                'margin': float(info.margin),
                'free_margin': float(info.margin_free),
                'margin_level': float(info.margin_level) if info.margin_level is not None else None,
                'currency': info.currency,
                'company': info.company,
                'leverage': info.leverage,
            }
        except Exception:
            return None

    def _mock_rates(self, bars: int) -> List[dict]:
        now = pd.Timestamp.utcnow().floor("s")
        timestamps = [now - pd.Timedelta(minutes=30 * i) for i in range(bars)][::-1]
        return [
            {
                "time": int(ts.timestamp()),
                "open": 2000.0 + i,
                "high": 2000.5 + i,
                "low": 1999.5 + i,
                "close": 2000.2 + i,
                "tick_volume": 100,
            }
            for i, ts in enumerate(timestamps)
        ]

    def get_symbol_info(self, symbol: str) -> Optional[Any]:
        if self._mock:
            return None
        if symbol in self._symbol_cache:
            return self._symbol_cache[symbol]
        info = mt5.symbol_info(symbol)
        if info is not None:
            self._symbol_cache[symbol] = info
        return info

    def get_symbol_tick(self, symbol: str) -> Optional[Dict[str, float]]:
        if self._mock:
            return None
        try:
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                return None
            return {
                "bid": getattr(tick, "bid", None),
                "ask": getattr(tick, "ask", None),
                "last": getattr(tick, "last", None),
            }
        except Exception:
            return None


    def get_rates(self, symbol: str, timeframe: str, bars: int) -> List[dict]:
        if self._mock:
            return self._mock_rates(bars)
        try:
            frame = mt5.TIMEFRAME_H1 if timeframe == "H1" else mt5.TIMEFRAME_M30
            rates = mt5.copy_rates_from_pos(symbol, frame, 0, bars)
            if rates is None:
                raise RuntimeError("MT5 lieferte keine Kurse")
            df = pd.DataFrame(rates)
            return df.to_dict("records")
        except Exception as exc:
            if self._mock:
                return self._mock_rates(bars)
            print(f"[MT5] Fallback auf Mock-Raten wegen: {exc}")
            return self._mock_rates(bars)

    def _resolve_point(self, info: "Any") -> float:
        point = info.point if info and hasattr(info, "point") else 0.0
        return max(point, 1e-9)

    def _required_stop_distance(self, info: "Any") -> float:
        stops_level = getattr(info, "trade_stops_level", 0) or 0
        min_points = max(stops_level, self._MIN_STOP_POINTS)
        return self._resolve_point(info) * max(min_points, 1)

    def _normalize_price(self, info: "Any", value: float, direction: str, is_stop: bool) -> float:
        if info is None:
            return value
        tick = getattr(info, "trade_tick_size", None) or getattr(info, "point", None)
        digits = getattr(info, "digits", None)
        if tick and tick > 0:
            scaled = value / tick
            if is_stop:
                steps = math.floor(scaled + 1e-9) if direction == "UP" else math.ceil(scaled - 1e-9)
            else:
                steps = math.ceil(scaled - 1e-9) if direction == "UP" else math.floor(scaled + 1e-9)
            return steps * tick
        if digits is not None:
            precision = max(0, int(digits))
            rounded = round(value, precision)
            if is_stop:
                return min(rounded, value) if direction == "UP" else max(rounded, value)
            return max(rounded, value) if direction == "UP" else min(rounded, value)
        return value

    def _normalize_stop(self, info: "Any", value: float, direction: str) -> float:
        return self._normalize_price(info, value, direction, is_stop=True)

    def _normalize_tp(self, info: "Any", value: float, direction: str) -> float:
        return self._normalize_price(info, value, direction, is_stop=False)

    def _resolve_trade_price(self, symbol: str, info: "Any", op: int) -> float:
        if self._mock:
            return getattr(info, "last", 0.0) or 0.0
        tick = mt5.symbol_info_tick(symbol)
        price = None
        if tick is not None:
            if op == mt5.ORDER_TYPE_BUY:
                price = getattr(tick, "ask", None)
            else:
                price = getattr(tick, "bid", None)
        if price is None or price <= 0:
            self._ensure_symbol_selected(symbol)
            tick = mt5.symbol_info_tick(symbol)
            if tick is not None:
                price = getattr(tick, "ask", None) if op == mt5.ORDER_TYPE_BUY else getattr(tick, "bid", None)
        if (price is None or price <= 0) and info is not None:
            fallback = getattr(info, "ask", None) if op == mt5.ORDER_TYPE_BUY else getattr(info, "bid", None)
            price = fallback or getattr(info, "last", None)
        if price is None or price <= 0:
            raise RuntimeError(f"Kein gültiger Preis für {symbol} verfügbar")
        return price

    def _validate_stop_distance(
        self, price: float, sl: float, direction: str, info: "Any", min_distance: Optional[float] = None
    ) -> tuple[bool, float, float, str]:
        if min_distance is None:
            min_distance = self._required_stop_distance(info)
        if direction == "UP":
            distance = price - sl
            if distance <= 0:
                return False, distance, min_distance, "Stop liegt über dem Kaufpreis"
        else:
            distance = sl - price
            if distance <= 0:
                return False, distance, min_distance, "Stop liegt unter dem Verkaufspreis"
        if distance + self._DISTANCE_EPS < min_distance:
            return False, distance, min_distance, f"Abstand {distance:.6f} < Mindestabstand {min_distance:.6f}"
        return True, distance, min_distance, ""

    def _adjust_stop(self, price: float, sl: float, direction: str, min_distance: float) -> float:
        if direction == "UP":
            return min(sl, price - min_distance)
        return max(sl, price + min_distance)

    def _send_request(self, request: dict) -> dict:
        result = mt5.order_send(request)
        return dict(result._asdict()) if hasattr(result, "_asdict") else dict(result)

    def _filling_candidates(self, preferred: int) -> list[int]:
        if mt5 is None:
            return [preferred]
        candidates = [preferred]
        for mode in (mt5.ORDER_FILLING_FOK, mt5.ORDER_FILLING_RETURN):
            if mode not in candidates:
                candidates.append(mode)
        return candidates

    def _determine_filling(self, info: "Any") -> int:
        if info and hasattr(info, "order_filling_mode"):
            return info.order_filling_mode
        return mt5.ORDER_FILLING_IOC

    @classmethod
    def describe_retcode(cls, retcode: Optional[int]) -> str:
        if retcode is None:
            return "ohne Retcode"
        return cls.RETCODE_DESCRIPTIONS.get(retcode, f"Retcode {retcode}")

    def place_market_order(self, symbol: str, volume: float, direction: str, sl: float, tp: float) -> dict:
        if self._mock:
            return {"symbol": symbol, "volume": volume, "direction": direction, "sl": sl, "tp": tp, "status": "mock"}
        try:
            self._ensure_symbol_selected(symbol)
            info = mt5.symbol_info(symbol)
            if info is None:
                raise RuntimeError(f"Symbolinfo für {symbol} fehlt")
            op = mt5.ORDER_TYPE_BUY if direction == "UP" else mt5.ORDER_TYPE_SELL
            price = self._resolve_trade_price(symbol, info, op)
            min_distance = self._required_stop_distance(info)
            sl_candidate = self._adjust_stop(price, sl, direction, min_distance)
            sl_safe = self._normalize_stop(info, sl_candidate, direction)
            tp_safe = self._normalize_tp(info, tp, direction)
            valid_stop, distance, _, reason = self._validate_stop_distance(price, sl_safe, direction, info, min_distance)
            adjusted = sl_safe != sl
            if not valid_stop:
                sl_safe = self._adjust_stop(price, sl_safe, direction, min_distance)
                sl_safe = self._normalize_stop(info, sl_safe, direction)
                valid_stop, distance, _, reason = self._validate_stop_distance(price, sl_safe, direction, info, min_distance)
            if not valid_stop:
                return {
                    "symbol": symbol,
                    "volume": volume,
                    "direction": direction,
                    "sl": sl_safe,
                    "tp": tp_safe,
                    "status": self._SKIPPED_INVALID_STOP,
                    "reason": reason,
                    "price": price,
                    "required_distance": min_distance,
                    "actual_distance": distance,
                    "stop_adjusted": adjusted,
                    "adjusted_stop": sl_safe,
                }
            filling_mode = self._determine_filling(info)
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": volume,
                "type": op,
                "price": price,
                "sl": sl_safe,
                "tp": tp_safe,
                "deviation": 5,
                "type_filling": filling_mode,
            }
            candidates = self._filling_candidates(filling_mode)
            attempts: list[dict] = []
            final_payload: Optional[dict] = None
            for mode in candidates:
                request["type_filling"] = mode
                payload = self._send_request(request)
                payload["retcode_description"] = self.describe_retcode(payload.get("retcode"))
                payload["stop_adjusted"] = adjusted
                payload["adjusted_stop"] = sl_safe
                payload["adjusted_distance"] = distance
                attempts.append({"mode": mode, "retcode": payload.get("retcode"), "comment": payload.get("comment")})
                if payload.get("retcode") != self._UNSUPPORTED_FILLING_RETCODE:
                    payload["filling_mode_used"] = mode
                    payload["filling_attempts"] = attempts
                    final_payload = payload
                    break
                final_payload = payload
            if final_payload is None:
                return {
                    "symbol": symbol,
                    "volume": volume,
                    "direction": direction,
                    "sl": sl_safe,
                    "tp": tp_safe,
                    "status": "fallback",
                }
            if "filling_mode_used" not in final_payload:
                final_payload["filling_mode_used"] = candidates[-1]
            final_payload.setdefault("filling_attempts", attempts)
            final_payload["retcode_description"] = self.describe_retcode(final_payload.get("retcode"))
            return final_payload
        except Exception as exc:
            print(f"[MT5] Order fehlgeschlagen: {exc}")
            return {"symbol": symbol, "volume": volume, "direction": direction, "sl": sl, "tp": tp, "status": "fallback"}

    def place_limit_order(
        self,
        symbol: str,
        volume: float,
        direction: str,
        price: float,
        sl: float,
        tp: float,
        expiration: Optional[int] = None,
    ) -> dict:
        if self._mock:
            payload = {
                "symbol": symbol,
                "volume": volume,
                "direction": direction,
                "price": price,
                "sl": sl,
                "tp": tp,
                "status": "mock",
                "expiration": expiration,
            }
            return payload
        try:
            self._ensure_symbol_selected(symbol)
            info = mt5.symbol_info(symbol)
            if info is None:
                raise RuntimeError(f"Symbolinfo für {symbol} fehlt")
            op = mt5.ORDER_TYPE_BUY_LIMIT if direction == "UP" else mt5.ORDER_TYPE_SELL_LIMIT
            min_distance = self._required_stop_distance(info)
            sl_candidate = self._adjust_stop(price, sl, direction, min_distance)
            sl_safe = self._normalize_stop(info, sl_candidate, direction)
            tp_safe = self._normalize_tp(info, tp, direction)
            valid_stop, distance, _, reason = self._validate_stop_distance(price, sl_safe, direction, info, min_distance)
            adjusted = sl_safe != sl
            if not valid_stop:
                sl_safe = self._adjust_stop(price, sl_safe, direction, min_distance)
                sl_safe = self._normalize_stop(info, sl_safe, direction)
                valid_stop, distance, _, reason = self._validate_stop_distance(price, sl_safe, direction, info, min_distance)
            if not valid_stop:
                return {
                    "symbol": symbol,
                    "volume": volume,
                    "direction": direction,
                    "price": price,
                    "sl": sl_safe,
                    "tp": tp_safe,
                    "status": self._SKIPPED_INVALID_STOP,
                    "reason": reason,
                    "required_distance": min_distance,
                    "actual_distance": distance,
                    "stop_adjusted": adjusted,
                    "adjusted_stop": sl_safe,
                }
            filling_mode = self._determine_filling(info)
            request = {
                "action": mt5.TRADE_ACTION_PENDING,
                "symbol": symbol,
                "volume": volume,
                "type": op,
                "price": price,
                "sl": sl_safe,
                "tp": tp_safe,
                "deviation": 5,
                "type_filling": filling_mode,
            }
            request["type_time"] = mt5.ORDER_TIME_GTC
            if expiration is not None:
                request["type_time"] = mt5.ORDER_TIME_SPECIFIED
                request["expiration"] = expiration
            candidates = self._filling_candidates(filling_mode)
            attempts: list[dict] = []
            final_payload: Optional[dict] = None
            for mode in candidates:
                request["type_filling"] = mode
                payload = self._send_request(request)
                payload["retcode_description"] = self.describe_retcode(payload.get("retcode"))
                payload["stop_adjusted"] = adjusted
                payload["adjusted_stop"] = sl_safe
                payload["adjusted_distance"] = distance
                attempts.append({"mode": mode, "retcode": payload.get("retcode"), "comment": payload.get("comment")})
                if payload.get("retcode") != self._UNSUPPORTED_FILLING_RETCODE:
                    payload["filling_mode_used"] = mode
                    payload["filling_attempts"] = attempts
                    final_payload = payload
                    break
                final_payload = payload
            if final_payload is None:
                return {
                    "symbol": symbol,
                    "volume": volume,
                    "direction": direction,
                    "price": price,
                    "sl": sl_safe,
                    "tp": tp_safe,
                    "status": "fallback",
                }
            if "filling_mode_used" not in final_payload:
                final_payload["filling_mode_used"] = candidates[-1]
            final_payload.setdefault("filling_attempts", attempts)
            final_payload["retcode_description"] = self.describe_retcode(final_payload.get("retcode"))
            return final_payload
        except Exception as exc:
            print(f"[MT5] Pending Order fehlgeschlagen: {exc}")
            return {
                "symbol": symbol,
                "volume": volume,
                "direction": direction,
                "price": price,
                "sl": sl,
                "tp": tp,
                "status": "fallback",
            }
    def modify_position_sl_tp(self, ticket: int, symbol: str, sl: Optional[float], tp: Optional[float]) -> dict:
        try:
            request: dict[str, Any] = {
                "action": mt5.TRADE_ACTION_SLTP,
                "symbol": symbol,
                "position": ticket,
            }
            if sl is not None:
                request["sl"] = sl
            if tp is not None:
                request["tp"] = tp
            result = self._send_request(request)
            result["retcode_description"] = self.describe_retcode(result.get("retcode"))
            return result
        except Exception as exc:
            print(f"[MT5] SL/TP-Anpassung fehlgeschlagen: {exc}")
            return {"status": "fallback", "ticket": ticket, "sl": sl, "tp": tp}

    def get_positions(self, symbol: Optional[str] = None) -> List[dict]:
        if self._mock:
            return []
        try:
            positions = mt5.positions_get(symbol=symbol)
            if not positions:
                return []
            normalized: list[dict] = []
            for pos in positions:
                if hasattr(pos, "_asdict"):
                    normalized.append(pos._asdict())
                elif hasattr(pos, "__dict__"):
                    normalized.append({k: v for k, v in pos.__dict__.items() if not k.startswith("_")})
                else:
                    normalized.append({})
            return normalized
        except Exception:
            return []