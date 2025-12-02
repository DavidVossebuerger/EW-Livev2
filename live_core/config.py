"""Konfiguration und Hilfen für das Live-System."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


@dataclass
class LiveConfig:
    """Basis-Konfiguration, die sowohl CLI als auch Produktionsläufe verwendet."""

    symbol: str = "NAS100"
    timeframe: str = "H1"
    lookback_bars: int = 500
    risk_per_trade: float = 0.01
    lot_size_per_risk_point: float = 1.0
    account_balance: float = 100000.0
    min_lot: float = 0.01
    max_lot: float = 10.0
    max_open_trades: int = 1
    dynamic_dd_risk: bool = True
    dd_risk_steps: Tuple[Tuple[float, float], ...] = field(
        default_factory=lambda: ((-10.0, 0.75), (-20.0, 0.5), (-30.0, 0.35), (-40.0, 0.25))
    )
    use_vol_target: bool = False
    target_annual_vol: float = 0.25
    vol_window_trades: int = 40
    risk_per_trade_min: float = 0.002
    risk_per_trade_max: float = 0.02
    size_short_factor: float = 1.0
    allow_shorts: bool = False
    use_ml_filters: bool = False
    ml_probability_threshold: float = 0.65
    ml_threshold_shift: float = 0.0
    size_by_prob: bool = False
    prob_size_min: float = 0.7
    prob_size_max: float = 1.5
    order_cooldown_seconds: int = 1
    entry_buffer_atr: float = 0.25
    stop_loss_multiplier: float = 1.5
    take_profit_multiplier: float = 1.0
    min_stop_distance_pct: float = 0.0008
    trade_cooldown_minutes: int = 10
    max_trades_per_symbol_per_hour: int = 2
    min_profit_factor: float = 1.2
    max_gross_exposure_pct: float = 0.01
    atr_period: int = 14
    ema_fast: int = 34
    ema_slow: int = 144
    csv_history_path: Optional[str] = None
    config_path: Optional[str] = None
    symbols_file: Optional[str] = "Symbols.txt"
    verbose: bool = False
    ml_probability_path: Optional[str] = None
    ml_default_probability: float = 0.65
    primary_zz_pct: float = 0.012
    primary_zz_atr_mult: float = 0.9
    primary_min_imp_atr: float = 1.8
    h1_zz_pct: float = 0.0020
    h1_zz_atr_mult: float = 0.6
    h1_min_imp_atr: float = 1.6
    webhook_url: Optional[str] = None
    entry_zone_w3: Tuple[float, float] = (0.382, 0.786)
    entry_zone_w5: Tuple[float, float] = (0.236, 0.618)
    entry_zone_c: Tuple[float, float] = (0.382, 0.786)
    entry_window_h1: int = 96
    entry_window_m30: int = 192
    max_hold_h1: int = 192
    max_hold_m30: int = 384
    tp1: float = 1.272
    tp2: float = 1.618
    atr_mult_buffer: float = 0.2
    atr_pct_min: float = 0.05
    atr_pct_max: float = 2.5
    confirm_bars_h1: int = 6
    confirm_bars_m30: int = 12
    confirm_rules: Tuple[str, ...] = field(default_factory=lambda: ("break_prev_extreme",))
    allow_touch_if_no_confirm: bool = True
    use_w5: bool = False
    use_adx: bool = False
    adx_trend_threshold: float = 25.0
    use_ema_trend: bool = False
    require_price_above_ema_fast: bool = False

    def with_overrides(self, overrides: Dict[str, Any]) -> "LiveConfig":
        """Returniert eine neue Config, bei der Einträge aus overrides priorisiert werden."""
        data = {**self.__dict__, **{k: v for k, v in overrides.items() if v is not None}}
        return LiveConfig(**data)

    @classmethod
    def load_from_file(cls, path: Optional[str]) -> "LiveConfig":
        if path is None:
            return cls()
        try:
            with open(path, "r", encoding="utf-8") as fh:
                contents = json.load(fh)
        except FileNotFoundError:
            raise
        except Exception as exc:  # pragma: no cover
            raise ValueError(f"Konfigurationsdatei '{path}' konnte nicht geladen werden: {exc}")
        return cls(**contents)

    @classmethod
    def env_overrides(cls) -> Dict[str, Any]:
        overrides: Dict[str, Any] = {}
        for field_name in cls.__dataclass_fields__:
            env_key = f"EW_{field_name.upper()}"
            if env_key in os.environ:
                overrides[field_name] = os.environ[env_key]
        return overrides

    @classmethod
    def load_from_env(cls) -> "LiveConfig":
        return cls().with_overrides(cls.env_overrides())

    def dump(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v is not None}

    @property
    def working_dir(self) -> Path:
        return Path(self.csv_history_path).parent if self.csv_history_path else Path.cwd()