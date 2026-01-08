"""Konfiguration und Hilfen fÃ¼r das Live-System."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Set, Tuple

AGGRESSIVE_PROFILE_DEFAULTS: Dict[str, Any] = {
    "atr_period": 14,
    "atr_mult_buffer": 0.20,
    "primary_zz_pct": 0.012,
    "primary_zz_atr_mult": 0.9,
    "primary_min_imp_atr": 1.8,
    "h1_zz_pct": 0.0020,
    "h1_zz_atr_mult": 0.6,
    "h1_min_imp_atr": 1.6,
    "entry_zone_w3": (0.382, 0.786),
    "entry_zone_w5": (0.236, 0.618),
    "entry_zone_c": (0.382, 0.786),
    "entry_window_h1": 96,
    "entry_window_m30": 192,
    "max_hold_h1": 192,
    "max_hold_m30": 384,
    "tp1": 1.272,
    "tp2": 1.618,
    "ema_fast": 34,
    "ema_slow": 144,
    "use_ema_trend": False,
    "require_price_above_ema_fast": False,
    "use_daily_ema": False,
    "atr_pct_min": 0.05,
    "atr_pct_max": 2.5,
    "confirm_bars_h1": 6,
    "confirm_bars_m30": 12,
    "confirm_rules": ("break_prev_extreme", "ema_fast_cross"),
    "allow_touch_if_no_confirm": True,
    "use_w5": False,
    "use_adx": False,
    "adx_period": 14,
    "adx_trend_threshold": 25.0,
    "order_store_path": "logs/orders.db",
    "risk_per_trade": 0.04,
    "dynamic_dd_risk": True,
    "dd_risk_steps": ((-10.0, 0.75), (-20.0, 0.5), (-30.0, 0.35), (-40.0, 0.25)),
    "use_vol_target": False,
    "target_annual_vol": 0.25,
    "vol_window_trades": 40,
    "risk_per_trade_min": 0.04,
    "risk_per_trade_max": 0.04,
    "size_short_factor": 0.7,
    "allow_shorts": True,
    "use_ml_filters": True,
    "size_by_prob": False,
    "prob_size_min": 0.7,
    "prob_size_max": 1.5,
    "use_vola_gate": True,
    "vola_probability_threshold": 0.2,
    "vola_horizon_days": 2.0,
    "vola_lookback_bars": 400,
    "vola_min_samples": 80,
    "min_stop_atr_mult": 1.0,
    "min_stop_pct": 0.003,
    "signal_dedup_window_minutes": 120,
    "use_vola_forecast": True,
    "vola_forecast_window": 252,
    "ml_probability_threshold": 0.65,
    "ml_threshold_shift": 0.0,
    "dynamic_trend_scaling": True,
    "setup_size_factors": {"C": 1.0, "W3": 0.85, "W5": 0.8},
    "tf_size_factors": {"H1": 1.0, "M30": 0.9},
    "max_gross_exposure_pct": 0.05,
    # Top-down data acquisition (Backtest parity)
    "use_topdown_structure": True,
    "daily_lookback_bars": 800,
    "h1_lookback_bars": 800,
    "m30_lookback_bars": 1200,
}


@dataclass
class LiveConfig:
    """Basis-Konfiguration, die sowohl CLI als auch ProduktionslÃ¤ufe verwendet."""

    symbol: str = "NAS100"
    timeframe: str = "H1"
    lookback_bars: int = 500
    risk_per_trade: float = 0.01
    lot_size_per_risk_point: float = 1.0
    account_balance: float = 10000.0
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
    dynamic_trend_scaling: bool = True
    setup_size_factors: Dict[str, float] = field(default_factory=lambda: {"C": 1.0, "W3": 0.85, "W5": 0.8})
    tf_size_factors: Dict[str, float] = field(default_factory=lambda: {"H1": 1.0, "M30": 0.9})
    use_pending_orders: bool = True
    pending_order_expiry_minutes: int = 120
    order_cooldown_seconds: int = 1
    entry_buffer_atr: float = 0.25
    stop_loss_multiplier: float = 1.5
    take_profit_multiplier: float = 1.0
    min_stop_distance_pct: float = 0.003
    trade_cooldown_minutes: int = 30
    max_trades_per_symbol_per_hour: int = 2
    min_profit_factor: float = 1.2
    max_gross_exposure_pct: float = 0.0
    price_guard_margin: float = 0.0
    orders_soft_limit: int = 180
    use_vola_gate: bool = True
    vola_probability_threshold: float = 0.2
    vola_horizon_days: float = 2.0
    vola_lookback_bars: int = 400
    vola_min_samples: int = 80
    # Momentum-Exit (Backtest-Parität): Höhere TF, konsekutive abnehmende Bars
    use_momentum_exit: bool = True
    momentum_exit_bars: int = 3           # Konsekutive Bars mit schwächerem Momentum
    momentum_period: int = 14             # Lookback für Momentum-Berechnung
    momentum_exit_higher_tf: bool = True  # H1 für M30, Daily für H1
    exposure_basis: str = "margin"
    exposure_custom_factor: float = 1.0
    exposure_default_leverage: float = 30.0
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
    confirm_rules: Tuple[str, ...] = field(default_factory=lambda: ("break_prev_extreme", "ema_fast_cross"))
    allow_touch_if_no_confirm: bool = True
    use_w5: bool = False
    use_adx: bool = False
    adx_period: int = 14
    adx_trend_threshold: float = 25.0
    use_ema_trend: bool = False
    require_price_above_ema_fast: bool = False
    use_daily_ema: bool = False
    order_store_path: Optional[str] = "logs/orders.db"
    # --- Top-down parity (Backtest) ---
    use_topdown_structure: bool = True
    daily_lookback_bars: int = 800
    h1_lookback_bars: int = 800
    m30_lookback_bars: int = 1200
    # Minimum SL distance as multiple of ATR (prevents too tight stops)
    min_stop_atr_mult: float = 1.0
    # Fallback minimum SL distance as percentage of entry price
    min_stop_pct: float = 0.003
    # Signal deduplication window in minutes (same setup within window = duplicate)
    signal_dedup_window_minutes: int = 120
    # Volatility Forecast-based Position Sizing
    use_vola_forecast: bool = True
    vola_forecast_window: int = 252  # Training window in daily bars

    def __post_init__(self) -> None:
        if not hasattr(self, "_provided_fields"):
            object.__setattr__(self, "_provided_fields", set())

    def with_overrides(
        self,
        overrides: Dict[str, Any],
        provided_keys: Optional[Set[str]] = None,
        register_overrides_as_provided: bool = True,
    ) -> "LiveConfig":
        """Returniert eine neue Config, bei der EintrÃ¤ge aus overrides priorisiert werden."""
        base_data = {k: getattr(self, k) for k in self.__dataclass_fields__}
        allowed_fields = set(self.__dataclass_fields__)
        filtered_overrides = {
            k: v for k, v in overrides.items() if v is not None and k in allowed_fields
        }
        data = {**base_data, **filtered_overrides}
        result = LiveConfig(**data)
        provided = set(getattr(self, "_provided_fields", set()))
        if provided_keys is not None:
            provided |= provided_keys
        elif register_overrides_as_provided:
            provided |= set(filtered_overrides.keys())
        object.__setattr__(result, "_provided_fields", provided)
        return result

    @classmethod
    def load_from_file(cls, path: Optional[str]) -> "LiveConfig":
        base = cls()
        if path is None:
            return base.apply_aggressive_profile()
        try:
            with open(path, "r", encoding="utf-8") as fh:
                contents = json.load(fh)
        except FileNotFoundError:
            raise
        except Exception as exc:  # pragma: no cover
            raise ValueError(f"Konfigurationsdatei '{path}' konnte nicht geladen werden: {exc}")
        cfg = base.with_overrides(contents, provided_keys=set(contents.keys()))
        return cfg.apply_aggressive_profile()

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
        base = cls()
        overrides = cls.env_overrides()
        cfg = base.with_overrides(overrides, provided_keys=set(overrides.keys()))
        return cfg.apply_aggressive_profile()

    def dump(self) -> Dict[str, Any]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__ if getattr(self, k) is not None}

    @property
    def working_dir(self) -> Path:
        return Path(self.csv_history_path).parent if self.csv_history_path else Path.cwd()

    def apply_aggressive_profile(self) -> "LiveConfig":
        provided = set(getattr(self, "_provided_fields", set()))
        filtered = {k: v for k, v in AGGRESSIVE_PROFILE_DEFAULTS.items() if k not in provided}
        if not filtered:
            return self
        return self.with_overrides(filtered, register_overrides_as_provided=False)