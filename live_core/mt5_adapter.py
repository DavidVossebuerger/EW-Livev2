"""Wrapper um MetaTrader5, damit wir live Orders ausführen können."""
from __future__ import annotations

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
    _SKIPPED_INVALID_STOP = "skipped-invalid-stop"
    _UNSUPPORTED_FILLING_RETCODE = 10030
    RETCODE_DESCRIPTIONS = {
        10000: "TRADE_RETCODE_DONE - Order erfolgreich",
        10007: "TRADE_RETCODE_INVALID_STOPS - Stops zu nah am Preis",
        10009: "TRADE_RETCODE_MARKET_CLOSED - Markt geschlossen",
        10016: "TRADE_RETCODE_LONG_ONLY - Nur Long erlaubt",
        10017: "TRADE_RETCODE_SHORT_ONLY - Nur Short erlaubt",
        10018: "TRADE_RETCODE_TRADE_PROHIBITED - Trading verboten",
        10030: "TRADE_RETCODE_INVALID_FILLING - Filling-Mode nicht erlaubt",
        10044: "TRADE_RETCODE_TRADE_CONTEXT_BUSY - Context busy",
    }

    def __init__(self, login: Optional[int] = None, password: Optional[str] = None, server: Optional[str] = None):
        self.login = login or int(os.environ.get("EW_MT5_LOGIN", "0"))
        self.password = password or os.environ.get("EW_MT5_PASSWORD", "")
        self.server = server or os.environ.get("EW_MT5_SERVER", "")
        self.connected = False
        self._mock = mt5 is None
        self._symbol_cache: dict[str, "Any"] = {}

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
        if distance < min_distance:
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
            info = mt5.symbol_info(symbol)
            if info is None:
                raise RuntimeError(f"Symbolinfo für {symbol} fehlt")
            op = mt5.ORDER_TYPE_BUY if direction == "UP" else mt5.ORDER_TYPE_SELL
            tick = mt5.symbol_info_tick(symbol)
            price = tick.ask if op == mt5.ORDER_TYPE_BUY else tick.bid
            tp_safe = tp  # take profit typically does not require min distance
            min_distance = self._required_stop_distance(info)
            sl_safe = self._adjust_stop(price, sl, direction, min_distance)
            valid_stop, distance, _, reason = self._validate_stop_distance(price, sl_safe, direction, info, min_distance)
            adjusted = sl_safe != sl
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