"""Wrapper um MetaTrader5, damit wir live Orders ausfÃ¼hren kÃ¶nnen."""
from __future__ import annotations

import math
import os
from typing import Any, Dict, List, Optional

import pandas as pd

try:
    import MetaTrader5 as mt5
except ImportError:  # pragma: no cover
    mt5 = None  # type: ignore


class _MockSymbolInfo:
    """Schlanke Stub-Struktur fÃ¼r Demo-Modus ohne MT5."""

    def __init__(self) -> None:
        self.point = 0.0001
        self.trade_tick_size = 0.0001
        self.trade_tick_value = 1.0
        self.trade_contract_size = 1.0
        self.volume_min = 0.01
        self.volume_max = 10.0
        self.volume_step = 0.01
        self.digits = 5
        self.ask = 2000.05
        self.bid = 1999.95
        self.last = 2000.0
        self.trade_stops_level = 50


class MetaTrader5Adapter:
    """Minimaler Adapter fÃ¼r Zugang zu MT5 (oder Mock bei installiertem MT5)."""

    _MIN_STOP_POINTS = 10
    _DISTANCE_EPS = 1e-6
    _SKIPPED_INVALID_STOP = "skipped-invalid-stop"
    _UNSUPPORTED_FILLING_RETCODE = 10030
    RETCODE_DESCRIPTIONS = {
        10004: "TRADE_RETCODE_REQUOTE - Requote erhalten",
        10006: "TRADE_RETCODE_REJECT - Order vom Server abgelehnt",
        10009: "TRADE_RETCODE_DONE - Order erfolgreich",
        10010: "TRADE_RETCODE_DONE_PARTIAL - Order teilweise gefÃ¼llt",
        10011: "TRADE_RETCODE_ERROR - Allgemeiner Trading-Fehler",
        10014: "TRADE_RETCODE_INVALID_VOLUME - Volumen ungÃ¼ltig",
        10015: "TRADE_RETCODE_INVALID_PRICE - Preis ungÃ¼ltig",
        10016: "TRADE_RETCODE_INVALID_STOPS - Stops ungÃ¼ltig",
        10017: "TRADE_RETCODE_TRADE_DISABLED - Symbol nicht handelbar",
        10018: "TRADE_RETCODE_MARKET_CLOSED - Markt geschlossen",
        10019: "TRADE_RETCODE_NO_MONEY - Nicht genug Margin",
        10020: "TRADE_RETCODE_PRICE_CHANGED - Preis hat sich verÃ¤ndert",
        10021: "TRADE_RETCODE_PRICE_OFF - Kein aktueller Preis",
        10030: "TRADE_RETCODE_INVALID_FILL - Filling-Mode nicht erlaubt",
        10031: "TRADE_RETCODE_CONNECTION - Verbindungsproblem",
        10042: "TRADE_RETCODE_LONG_ONLY - Nur Long erlaubt",
        10043: "TRADE_RETCODE_SHORT_ONLY - Nur Short erlaubt",
        10044: "TRADE_RETCODE_CLOSE_ONLY - Nur Schliessen erlaubt",
    }

    def __init__(
        self,
        login: Optional[int] = None,
        password: Optional[str] = None,
        server: Optional[str] = None,
        *,
        force_mock: bool = False,
        mock_account_info: Optional[Dict[str, Any]] = None,
    ):
        self.login = login or int(os.environ.get("EW_MT5_LOGIN", "0"))
        self.password = password or os.environ.get("EW_MT5_PASSWORD", "")
        self.server = server or os.environ.get("EW_MT5_SERVER", "")
        self.connected = False
        if mt5 is None and not force_mock:
            raise RuntimeError(
                "MetaTrader5-Python-Paket nicht verfÃ¼gbar. Bitte installieren oder --demo nutzen."
            )
        self._mock = bool(force_mock)
        self._mock_account_info = dict(mock_account_info or {})
        self._symbol_cache: dict[str, "Any"] = {}
        self._mock_ticks: dict[str, dict[str, float]] = {}

    @property
    def is_mock(self) -> bool:
        return self._mock

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
            if self._mock_account_info:
                return dict(self._mock_account_info)
            return {
                "login": 0,
                "balance": 10_000.0,
                "equity": 10_000.0,
                "margin": 0.0,
                "free_margin": 10_000.0,
                "margin_level": 0.0,
                "currency": "DEMO",
                "company": "MockBroker",
                "leverage": 30,
            }
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

    def _mock_rates(self, symbol: str, bars: int) -> List[dict]:
        now = pd.Timestamp.utcnow().floor("s")
        timestamps = [now - pd.Timedelta(minutes=30 * i) for i in range(bars)][::-1]
        rates = [
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
        if rates:
            last_close = rates[-1]["close"]
            self._mock_ticks[symbol] = {"bid": last_close - 0.05, "ask": last_close + 0.05, "last": last_close}
        return rates

    def get_symbol_info(self, symbol: str, *, refresh: bool = False) -> Optional[Any]:
        """Fetches symbol info with optional cache refresh."""
        if self._mock:
            if symbol in self._symbol_cache and not refresh:
                return self._symbol_cache[symbol]
            info = _MockSymbolInfo()
            self._symbol_cache[symbol] = info
            return info
        if refresh:
            self._symbol_cache.pop(symbol, None)
        if symbol in self._symbol_cache:
            return self._symbol_cache[symbol]
        info = mt5.symbol_info(symbol)
        if info is not None:
            self._symbol_cache[symbol] = info
        return info

    def get_symbol_tick(self, symbol: str) -> Optional[Dict[str, float]]:
        if self._mock:
            if symbol not in self._mock_ticks:
                info = self.get_symbol_info(symbol)
                base = getattr(info, "last", None) or 2000.0
                self._mock_ticks[symbol] = {
                    "bid": base - 0.05,
                    "ask": base + 0.05,
                    "last": base,
                }
            return dict(self._mock_ticks.get(symbol, {}))
        try:
            self._ensure_symbol_selected(symbol)
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

    def get_orders_total(self) -> Optional[int]:
        """Returns total open/pending orders (mock -> 0)."""
        if self._mock:
            return 0
        try:
            orders = mt5.orders_get()
            return len(orders) if orders is not None else 0
        except Exception:
            return None

    def cancel_all_orders(self, symbol: Optional[str] = None) -> int:
        """Cancels all pending orders for the given symbol (or all symbols)."""
        if self._mock:
            return 0
        try:
            orders = mt5.orders_get(symbol=symbol)
            if not orders:
                return 0
            cancelled = 0
            for order in orders:
                ticket = getattr(order, "ticket", None)
                if ticket is None:
                    continue
                request = {"action": mt5.TRADE_ACTION_REMOVE, "order": ticket}
                result = self._send_request(request)
                retcode = result.get("retcode")
                if retcode == 10009:
                    cancelled += 1
            return cancelled
        except Exception as exc:
            print(f"[MT5] Cancel all orders fehlgeschlagen: {exc}")
            return 0


    def get_rates(self, symbol: str, timeframe: str, bars: int) -> List[dict]:
        if self._mock:
            return self._mock_rates(symbol, bars)
        try:
            tf = (timeframe or "").upper()
            if tf in {"H1", "1H"}:
                frame = mt5.TIMEFRAME_H1
            elif tf in {"M30", "30M"}:
                frame = mt5.TIMEFRAME_M30
            elif tf in {"D1", "1D", "DAILY"}:
                frame = mt5.TIMEFRAME_D1
            else:
                raise ValueError(f"Unsupported timeframe '{timeframe}' (expected H1/M30/D1)")
            rates = mt5.copy_rates_from_pos(symbol, frame, 0, bars)
            if rates is None:
                raise RuntimeError("MT5 lieferte keine Kurse")
            df = pd.DataFrame(rates)
            return df.to_dict("records")
        except Exception as exc:
            # Kein Fallback auf Mock in Live-Betrieb: lieber leer zurÃ¼ckgeben und Symbol skippen.
            if self._mock:
                return self._mock_rates(symbol, bars)
            print(f"[MT5] Keine Kurse fÃ¼r {symbol}: {exc}")
            return []

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
            raise RuntimeError(f"Kein gÃ¼ltiger Preis fÃ¼r {symbol} verfÃ¼gbar")
        return price

    def _validate_stop_distance(
        self, price: float, sl: float, direction: str, info: "Any", min_distance: Optional[float] = None
    ) -> tuple[bool, float, float, str]:
        if min_distance is None:
            min_distance = self._required_stop_distance(info)
        if direction == "UP":
            distance = price - sl
            if distance <= 0:
                return False, distance, min_distance, "Stop liegt Ã¼ber dem Kaufpreis"
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

    def _ceil_volume(self, info: "Any", volume: float) -> float:
        min_vol = getattr(info, "volume_min", 0.0) or 0.0
        max_vol = getattr(info, "volume_max", None)
        step = getattr(info, "volume_step", 0.01) or 0.01
        step = max(step, 1e-9)
        candidate = max(volume, min_vol)
        try:
            steps = math.ceil((candidate + 1e-9) / step)
            candidate = steps * step
        except Exception:
            candidate = candidate
        if max_vol is not None:
            candidate = min(candidate, max_vol)
        return float(round(candidate, 6))

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
                raise RuntimeError(f"Symbolinfo fÃ¼r {symbol} fehlt")
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
                while True:
                    payload = self._send_request(request)
                    payload["retcode_description"] = self.describe_retcode(payload.get("retcode"))
                    payload["stop_adjusted"] = adjusted
                    payload["adjusted_stop"] = sl_safe
                    payload["adjusted_distance"] = distance
                    attempts.append(
                        {
                            "mode": mode,
                            "retcode": payload.get("retcode"),
                            "comment": payload.get("comment"),
                            "volume": request.get("volume"),
                        }
                    )
                    ret = payload.get("retcode")
                    if ret == 10014:
                        new_vol = self._ceil_volume(info, float(request.get("volume", 0.0)) + 1e-9)
                        if new_vol <= float(request.get("volume", 0.0)):
                            break
                        request["volume"] = new_vol
                        continue
                    if ret != self._UNSUPPORTED_FILLING_RETCODE:
                        payload["filling_mode_used"] = mode
                        payload["filling_attempts"] = attempts
                        final_payload = payload
                        break
                    final_payload = payload
                    break
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
                raise RuntimeError(f"Symbolinfo fÃ¼r {symbol} fehlt")
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
            # Debug logging for SL/TP
            print(f"[MT5] Limit Order Request: {symbol} @ {price:.5f} SL={sl_safe:.5f} TP={tp_safe:.5f}")
            request["type_time"] = mt5.ORDER_TIME_GTC
            if expiration and expiration > 0:
                request["type_time"] = mt5.ORDER_TIME_SPECIFIED
                request["expiration"] = expiration
            candidates = self._filling_candidates(filling_mode)
            attempts: list[dict] = []
            final_payload: Optional[dict] = None
            for mode in candidates:
                request["type_filling"] = mode
                while True:
                    payload = self._send_request(request)
                    payload["retcode_description"] = self.describe_retcode(payload.get("retcode"))
                    payload["stop_adjusted"] = adjusted
                    payload["adjusted_stop"] = sl_safe
                    payload["adjusted_distance"] = distance
                    attempts.append(
                        {
                            "mode": mode,
                            "retcode": payload.get("retcode"),
                            "comment": payload.get("comment"),
                            "volume": request.get("volume"),
                        }
                    )
                    ret = payload.get("retcode")
                    if ret == 10014:
                        new_vol = self._ceil_volume(info, float(request.get("volume", 0.0)) + 1e-9)
                        if new_vol <= float(request.get("volume", 0.0)):
                            break
                        request["volume"] = new_vol
                        continue
                    if ret != self._UNSUPPORTED_FILLING_RETCODE:
                        payload["filling_mode_used"] = mode
                        payload["filling_attempts"] = attempts
                        final_payload = payload
                        break
                    final_payload = payload
                    break
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

    def _order_direction_from_type(self, order_type: Optional[int]) -> Optional[str]:
        if mt5 is None or order_type is None:
            return None
        if order_type in (
            mt5.ORDER_TYPE_BUY,
            mt5.ORDER_TYPE_BUY_LIMIT,
            mt5.ORDER_TYPE_BUY_STOP,
            mt5.ORDER_TYPE_BUY_STOP_LIMIT,
        ):
            return "UP"
        if order_type in (
            mt5.ORDER_TYPE_SELL,
            mt5.ORDER_TYPE_SELL_LIMIT,
            mt5.ORDER_TYPE_SELL_STOP,
            mt5.ORDER_TYPE_SELL_STOP_LIMIT,
        ):
            return "DOWN"
        return None

    def get_orders(self, symbol: Optional[str] = None) -> List[dict]:
        if self._mock:
            return []
        try:
            # MT5 returns None when symbol=None is passed explicitly, so only forward the filter when set.
            orders = mt5.orders_get(symbol=symbol) if symbol is not None else mt5.orders_get()
            if not orders:
                return []
            normalized: list[dict] = []
            for order in orders:
                if hasattr(order, "_asdict"):
                    data = order._asdict()
                elif hasattr(order, "__dict__"):
                    data = {k: v for k, v in order.__dict__.items() if not k.startswith("_")}
                else:
                    data = {}
                data["direction"] = self._order_direction_from_type(data.get("type"))
                normalized.append(data)
            return normalized
        except Exception:
            return []

    def get_positions(self, symbol: Optional[str] = None) -> List[dict]:
        if self._mock:
            return []
        try:
            # Passing symbol=None to MT5 yields None; call without filter to fetch all positions.
            positions = mt5.positions_get(symbol=symbol) if symbol is not None else mt5.positions_get()
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

    def close_position(self, ticket: int, symbol: str, volume: float, pos_type: int) -> dict:
        """Schließt eine Position durch Gegenorder.

        Args:
            ticket: Position ticket
            symbol: Symbol der Position
            volume: Volumen zum Schließen
            pos_type: 0 = BUY (schließen mit SELL), 1 = SELL (schließen mit BUY)

        Returns:
            Result dict mit retcode etc.
        """
        if self._mock:
            return {"retcode": 10009, "status": "mock", "ticket": ticket}
        try:
            self._ensure_symbol_selected(symbol)
            info = self.get_symbol_info(symbol, refresh=True)
            if info is None:
                return {"status": "error", "reason": "Symbol info not available"}
            tick = self.get_symbol_tick(symbol)
            if not tick:
                return {"status": "error", "reason": "No tick data"}
            # Gegenorder: BUY Position -> SELL, SELL Position -> BUY
            if pos_type == 0:  # BUY position
                order_type = mt5.ORDER_TYPE_SELL
                price = tick.get("bid", 0.0)
            else:  # SELL position
                order_type = mt5.ORDER_TYPE_BUY
                price = tick.get("ask", 0.0)
            request: dict[str, Any] = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": float(volume),
                "type": order_type,
                "position": ticket,
                "price": price,
                "deviation": 20,
                "magic": 234000,
                "comment": "momentum_exit",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            result = self._send_request(request)
            result["retcode_description"] = self.describe_retcode(result.get("retcode"))
            return result
        except Exception as exc:
            print(f"[MT5] Position close fehlgeschlagen: {exc}")
            return {"status": "fallback", "ticket": ticket}