"""
Advanced Risk Manager mit Differential ML SensitivitÃ¤ten.
Basiert auf "Differential ML with a Difference" (Glasserman & Karmarkar, 2025).
"""
from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from statistics import StatisticsError, mean, stdev
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

from .signals import EntrySignal, Dir
from .config import LiveConfig

# Import volatility forecaster (optional)
try:
    from volatility_backtest import VolatilityForecaster, create_vola_forecaster
    VOLA_FORECAST_AVAILABLE = True
except ImportError:
    VOLA_FORECAST_AVAILABLE = False

logger = logging.getLogger("ew.risk")


@dataclass
class RiskSensitivities:
    """SensitivitÃ¤ten eines Trading-Signals zu verschiedenen Parametern."""

    # HauptsensitivitÃ¤ten (analog zu Greeks bei Optionen)
    delta_atr: float  # SensitivitÃ¤t zu ATR-Ã„nderungen
    delta_volatility: float  # SensitivitÃ¤t zu VolatilitÃ¤tsÃ¤nderungen
    delta_price_momentum: float  # SensitivitÃ¤t zu Momentum
    delta_ema_distance: float  # SensitivitÃ¤t zu EMA-Distanz

    # Zweite Ordnung (analog zu Gamma)
    gamma_atr: Optional[float] = None

    # Konfidenz der SchÃ¤tzungen
    confidence_level: float = 0.0

    # Methode verwendet: "pathwise", "lrm", oder "hybrid"
    method: str = "lrm"


@dataclass
class RiskMetrics:
    """Aggregierte Risk-Metriken fÃ¼r Portfolio."""

    total_exposure: float
    risk_per_trade: Dict[str, float]
    portfolio_volatility: float
    var_95: float  # Value at Risk 95%
    expected_shortfall: float  # CVaR
    sharpe_estimate: float
    max_drawdown_current: float
    sensitivity_weighted_exposure: float
    timestamp: datetime


class RiskManager:
    """
    Advanced Risk Manager mit SensitivitÃ¤tsanalyse.

    Implementiert Konzepte aus Differential ML fÃ¼r Trading:
    - Likelihood Ratio Method (LRM) fÃ¼r unverzerrte SensitivitÃ¤ten
    - SensitivitÃ¤tsbasierte PositionsgrÃ¶ÃŸenanpassung
    - Portfolio-weites Risikomanagement
    - Echtzeit Risk Monitoring
    """

    # Konfigurations-Parameter
    MAX_SENSITIVITY_FACTOR = 2.0  # Max SensitivitÃ¤ts-Adjustment
    MIN_SENSITIVITY_FACTOR = 0.3  # Min SensitivitÃ¤ts-Adjustment
    SENSITIVITY_HISTORY_SIZE = 100

    def __init__(self, cfg: LiveConfig):
        self.cfg = cfg

        # SensitivitÃ¤ts-Historie fÃ¼r adaptive Kalibrierung
        self.sensitivity_history: Deque[RiskSensitivities] = deque(
            maxlen=self.SENSITIVITY_HISTORY_SIZE
        )

        # Tracking von realisierten vs. geschÃ¤tzten Risiken
        self.realized_pnl: Deque[float] = deque(maxlen=200)
        self.estimated_risk: Deque[float] = deque(maxlen=200)

        # Symbol-spezifische SensitivitÃ¤ts-Statistiken
        self.symbol_sensitivities: Dict[str, List[RiskSensitivities]] = {}

        # Portfolio Risk Tracking
        self.portfolio_metrics_history: Deque[RiskMetrics] = deque(maxlen=1000)

        # Volatility Forecast for position sizing
        self.vola_forecaster = None
        self.vola_forecast_ready = False
        self._vola_forecast_cache: Dict[str, Dict] = {}  # Cache per symbol
        self._vola_forecast_cache_time: Dict[str, datetime] = {}
        self._vola_cache_ttl_seconds = 3600  # 1 hour cache
        
        if getattr(cfg, 'use_vola_forecast', False) and VOLA_FORECAST_AVAILABLE:
            try:
                window = getattr(cfg, 'vola_forecast_window', 252)
                self.vola_forecaster = create_vola_forecaster(train_window=window)
                self.vola_forecast_ready = True
                logger.info(f"Volatility Forecaster initialized (window={window})")
            except Exception as e:
                logger.warning(f"Volatility Forecaster init failed: {e}")
        elif getattr(cfg, 'use_vola_forecast', False) and not VOLA_FORECAST_AVAILABLE:
            logger.warning("use_vola_forecast=True but volatility_backtest.py not available")

        logger.info(
            "Risk Manager initialisiert mit Differential ML SensitivitÃ¤ten"
        )

    # ------------------------------------------------------------------
    # Hauptfunktionen
    # ------------------------------------------------------------------
    def calculate_sensitivities(
        self,
        signal: EntrySignal,
        market_data: Dict[str, float],
        historical_outcomes: Optional[List[Dict]] = None,
    ) -> RiskSensitivities:
        """
        Berechnet SensitivitÃ¤ten mit Likelihood Ratio Method (LRM).

        LRM ist robust fÃ¼r "diskontinuierliche Payoffs" (Stop-Loss getroffen ja/nein),
        analog zu digitalen Optionen im Paper.
        """
        logger.debug(
            "Berechne SensitivitÃ¤ten fÃ¼r %s %s", signal.setup, signal.direction
        )

        # Extrahiere relevante Marktparameter (robust gegen None)
        atr = market_data.get("atr", 0.0)
        price_raw = market_data.get("price", getattr(signal, "entry_price", 0.0))
        try:
            price_current = float(price_raw) if price_raw is not None else 0.0
        except (TypeError, ValueError):
            price_current = 0.0

        def _safe_level(value: Optional[float], fallback: float) -> float:
            try:
                return float(value) if value is not None else fallback
            except (TypeError, ValueError):
                return fallback

        ema_fast = _safe_level(market_data.get("ema_fast"), price_current)
        ema_slow = _safe_level(market_data.get("ema_slow"), price_current)

        # Berechne VolatilitÃ¤tsmaÃŸe
        volatility = self._estimate_volatility(market_data)
        momentum = self._calculate_momentum(market_data)
        ema_distance = abs(ema_fast - ema_slow) / price_current if price_current else 0.0

        # LRM SensitivitÃ¤ten berechnen
        delta_atr = self._calculate_lrm_sensitivity_atr(
            signal, atr, historical_outcomes
        )

        delta_volatility = self._calculate_lrm_sensitivity_volatility(
            signal, volatility, historical_outcomes
        )

        delta_momentum = self._calculate_lrm_sensitivity_momentum(
            signal, momentum, historical_outcomes
        )

        delta_ema = self._calculate_lrm_sensitivity_ema(
            signal, ema_distance, historical_outcomes
        )

        # Optional: Gamma (zweite Ableitung) fÃ¼r KonvexitÃ¤tsrisiko
        gamma_atr = None
        if getattr(self.cfg, "use_ml_filters", False):
            gamma_atr = self._calculate_gamma_atr(signal, atr)

        # Konfidenz basierend auf DatenqualitÃ¤t
        confidence = self._estimate_confidence(historical_outcomes)

        sensitivities = RiskSensitivities(
            delta_atr=delta_atr,
            delta_volatility=delta_volatility,
            delta_price_momentum=delta_momentum,
            delta_ema_distance=delta_ema,
            gamma_atr=gamma_atr,
            confidence_level=confidence,
            method="lrm",
        )

        # Speichere fÃ¼r adaptive Kalibrierung
        self.sensitivity_history.append(sensitivities)

        # Store per symbol
        symbol = getattr(signal, "symbol", None)
        if symbol:
            self.symbol_sensitivities.setdefault(symbol, []).append(sensitivities)

        return sensitivities

    def adjust_position_size_for_sensitivities(
        self,
        base_volume: float,
        sensitivities: RiskSensitivities,
        market_uncertainty: Dict[str, float],
    ) -> float:
        """Moduliert PositionsgrÃ¶ÃŸe anhand SensitivitÃ¤ten und Unsicherheiten."""
        # Gewichtete Summe der SensitivitÃ¤ten (Betrag)
        weights = [
            abs(sensitivities.delta_atr),
            abs(sensitivities.delta_volatility),
            abs(sensitivities.delta_price_momentum),
            abs(sensitivities.delta_ema_distance),
        ]
        sens_score = float(np.nan_to_num(np.mean(weights))) if weights else 0.0

        # Unsicherheitsfaktor: hÃ¶heres Risiko -> kleinere GrÃ¶ÃŸe
        def _safe_float(value: Optional[float], default: float = 0.0) -> float:
            try:
                return float(value) if value is not None else default
            except (TypeError, ValueError):
                return default

        uncert = _safe_float(market_uncertainty.get("volatility"), 0.0)
        uncert += _safe_float(market_uncertainty.get("spread"), 0.0)
        uncert_factor = 1.0 / (1.0 + uncert)

        # Confidence steuert zurÃ¼ck Richtung 1.0
        confidence = max(0.0, min(1.0, sensitivities.confidence_level))
        conf_factor = 0.5 + 0.5 * confidence

        factor = sens_score * conf_factor * uncert_factor
        # Map factor in [MIN, MAX]
        factor = max(self.MIN_SENSITIVITY_FACTOR, min(self.MAX_SENSITIVITY_FACTOR, factor))
        adjusted = base_volume * factor
        
        # Apply volatility forecast sizing if available
        vola_mult = self._get_vola_size_multiplier(market_uncertainty.get("symbol"))
        if vola_mult != 1.0:
            logger.debug(f"Vola forecast multiplier: {vola_mult:.2f}")
            adjusted *= vola_mult
        
        return max(self.cfg.min_lot, min(adjusted, self.cfg.max_lot))
    
    def _get_vola_size_multiplier(self, symbol: Optional[str] = None) -> float:
        """Get position size multiplier from volatility forecast.
        
        Returns:
            Multiplier between 0.6 (high vola, size down) and 1.15 (low vola, size up)
        """
        if not self.vola_forecast_ready or self.vola_forecaster is None:
            return 1.0
        
        # Check cache first
        cache_key = symbol or "default"
        now = datetime.now(timezone.utc)
        if cache_key in self._vola_forecast_cache:
            cache_time = self._vola_forecast_cache_time.get(cache_key)
            if cache_time and (now - cache_time).total_seconds() < self._vola_cache_ttl_seconds:
                cached = self._vola_forecast_cache[cache_key]
                return cached.get('size_multiplier', 1.0)
        
        # No valid cache - we need daily data to forecast
        # In live system, this is called from execution with market_uncertainty dict
        # We'll use the cached result or default to 1.0
        # The actual forecast is done in update_vola_forecast() called from cycle
        return self._vola_forecast_cache.get(cache_key, {}).get('size_multiplier', 1.0)
    
    def update_vola_forecast(self, symbol: str, daily_df) -> Optional[Dict]:
        """Update volatility forecast using daily data.
        
        Called from cycle.py when new daily data is available.
        
        Args:
            symbol: Trading symbol
            daily_df: DataFrame with daily OHLC data
            
        Returns:
            Forecast dict with 'size_multiplier', 'regime', 'forecast', or None
        """
        if not self.vola_forecast_ready or self.vola_forecaster is None:
            return None
        
        try:
            import pandas as pd
            
            # Prepare data if not already prepared
            if 'log_returns' not in daily_df.columns:
                daily_df = self.vola_forecaster.prepare_data(daily_df.copy())
            
            # Get forecast for latest bar
            idx = len(daily_df) - 1
            window = getattr(self.cfg, 'vola_forecast_window', 252)
            
            if idx < window:
                logger.debug(f"Not enough data for vola forecast: {idx} < {window}")
                return None
            
            result = self.vola_forecaster.forecast_volatility(daily_df, idx)
            
            if result:
                # Cache the result
                self._vola_forecast_cache[symbol] = result
                self._vola_forecast_cache_time[symbol] = datetime.now(timezone.utc)
                
                logger.info(
                    f"[{symbol}] Vola forecast: regime={result.get('regime')}, "
                    f"mult={result.get('size_multiplier', 1.0):.2f}"
                )
                return result
                
        except Exception as e:
            logger.warning(f"Vola forecast update failed: {e}")
        
        return None

    def compute_risk_metrics(
        self,
        open_positions: List[Dict[str, float]],
        recent_returns: Optional[List[float]],
        equity: float,
    ) -> RiskMetrics:
        """Aggregiert Portfoliorisiko-Metriken auf Basis aktueller Positionen."""
        exposure = sum(float(p.get("volume", 0.0)) * float(p.get("price", 0.0)) for p in open_positions)
        risk_per_trade: Dict[str, float] = {}
        for pos in open_positions:
            sym = str(pos.get("symbol", "")).upper()
            risk_per_trade[sym] = float(pos.get("risk", pos.get("volume", 0.0)))

        port_vol = self._realized_volatility(recent_returns) if recent_returns else 0.0
        var_95, es = self._var_es(recent_returns) if recent_returns else (0.0, 0.0)
        sharpe = self._sharpe_estimate(recent_returns) if recent_returns else 0.0
        mdd = self._max_drawdown_from_returns(recent_returns) if recent_returns else 0.0

        # Sensitivity weighted exposure approximiert
        sens_weighted = exposure
        ts = datetime.now(timezone.utc)
        metrics = RiskMetrics(
            total_exposure=exposure,
            risk_per_trade=risk_per_trade,
            portfolio_volatility=port_vol,
            var_95=var_95,
            expected_shortfall=es,
            sharpe_estimate=sharpe,
            max_drawdown_current=mdd,
            sensitivity_weighted_exposure=sens_weighted,
            timestamp=ts,
        )
        self.portfolio_metrics_history.append(metrics)
        return metrics

    def log_risk_state(self, metrics: RiskMetrics) -> None:
        logger.info(
            "RiskState exposure=%.2f var95=%.4f es=%.4f sharpe=%.3f mdd=%.2f",
            metrics.total_exposure,
            metrics.var_95,
            metrics.expected_shortfall,
            metrics.sharpe_estimate,
            metrics.max_drawdown_current,
        )

    def track_trade_outcome(
        self,
        symbol: str,
        pnl: float,
        risk_amount: float,
        success: bool,
        market_snapshot: Optional[Dict[str, float]] = None,
    ) -> None:
        """Persistiert Outcomes fÃ¼r spÃ¤tere SensitivitÃ¤ts-SchÃ¤tzungen."""
        self.realized_pnl.append(pnl)
        self.estimated_risk.append(risk_amount)
        if market_snapshot is not None:
            snap = dict(market_snapshot)
            snap["success"] = success
            snap.setdefault("symbol", symbol)
            # Speichere rudimentÃ¤r in history for adaptive sensitivity
            sens = RiskSensitivities(
                delta_atr=snap.get("atr", 0.0),
                delta_volatility=snap.get("volatility", 0.0),
                delta_price_momentum=snap.get("momentum", 0.0),
                delta_ema_distance=snap.get("ema_distance", 0.0),
                gamma_atr=None,
                confidence_level=0.0,
                method="history",
            )
            self.symbol_sensitivities.setdefault(symbol, []).append(sens)

    # ------------------------------------------------------------------
    # Hilfsfunktionen
    # ------------------------------------------------------------------
    def _estimate_volatility(self, market_data: Dict[str, float]) -> float:
        # Bevorzugt bereits vorberechnete Werte; toleriert None
        if "volatility" in market_data:
            raw = market_data.get("volatility")
            try:
                return float(raw) if raw is not None else 0.0
            except (TypeError, ValueError):
                return 0.0
        window = market_data.get("returns_window")
        if window is None:
            return 0.0
        arr = np.asarray(window, dtype=float)
        if arr.size == 0:
            return 0.0
        return float(np.nanstd(arr))

    def _calculate_momentum(self, market_data: Dict[str, float]) -> float:
        hist = market_data.get("price_history")
        if hist is None:
            return 0.0
        arr = np.asarray(hist, dtype=float)
        if arr.size < 2:
            return 0.0
        return float((arr[-1] - arr[0]) / max(1e-9, arr[0]))

    def _estimate_confidence(self, historical_outcomes: Optional[List[Dict]]) -> float:
        if not historical_outcomes:
            return 0.0
        successes = sum(1 for d in historical_outcomes if d.get("success"))
        total = len(historical_outcomes)
        base = successes / total if total else 0.0
        # Penalize short histories
        penalty = 0.2 if total < 30 else 0.0
        return max(0.0, min(1.0, base - penalty))

    def _calculate_lrm_sensitivity_atr(
        self,
        signal: EntrySignal,
        atr_current: float,
        historical_outcomes: Optional[List[Dict]],
    ) -> float:
        if not historical_outcomes or len(historical_outcomes) < 10:
            # Fallback auf heuristische SchÃ¤tzung
            return -0.5  # Negative Korrelation: hÃ¶here ATR = hÃ¶heres Risiko

        atr_values = [d.get("atr", 0.0) for d in historical_outcomes]
        success_flags = [d.get("success", False) for d in historical_outcomes]

        try:
            atr_mean = mean(atr_values)
            atr_std = stdev(atr_values)
        except StatisticsError:
            return -0.5

        if atr_std < 1e-9:
            return 0.0

        score = (atr_current - atr_mean) / (atr_std ** 2)
        success_rate = sum(success_flags) / len(success_flags)
        delta_lrm = success_rate * score
        delta_lrm = max(-2.0, min(2.0, delta_lrm))

        logger.debug(
            "LRM ATR Sensitivity: %.4f (ATR=%.5f, mu=%.5f, sigma=%.5f)",
            delta_lrm,
            atr_current,
            atr_mean,
            atr_std,
        )
        return delta_lrm

    def _calculate_lrm_sensitivity_volatility(
        self,
        signal: EntrySignal,
        volatility: float,
        historical_outcomes: Optional[List[Dict]],
    ) -> float:
        if not historical_outcomes or len(historical_outcomes) < 10:
            return -0.3

        vol_values = [d.get("volatility", 0.0) for d in historical_outcomes]
        success_flags = [d.get("success", False) for d in historical_outcomes]

        try:
            vol_mean = mean(vol_values)
            vol_std = stdev(vol_values)
        except StatisticsError:
            return -0.3

        if vol_std < 1e-9:
            return 0.0

        score = (volatility - vol_mean) / (vol_std ** 2)
        success_rate = sum(success_flags) / len(success_flags)
        delta_lrm = success_rate * score
        return max(-2.0, min(2.0, delta_lrm))

    def _calculate_lrm_sensitivity_momentum(
        self,
        signal: EntrySignal,
        momentum: float,
        historical_outcomes: Optional[List[Dict]],
    ) -> float:
        if not historical_outcomes or len(historical_outcomes) < 10:
            return 0.4 if signal.direction == Dir.UP else -0.4

        momentum_values = [d.get("momentum", 0.0) for d in historical_outcomes]
        success_flags = [d.get("success", False) for d in historical_outcomes]

        try:
            mom_mean = mean(momentum_values)
            mom_std = stdev(momentum_values)
        except StatisticsError:
            return 0.4 if signal.direction == Dir.UP else -0.4

        if mom_std < 1e-9:
            return 0.0

        score = (momentum - mom_mean) / (mom_std ** 2)
        success_rate = sum(success_flags) / len(success_flags)
        delta_lrm = success_rate * score
        return max(-2.0, min(2.0, delta_lrm))

    def _calculate_lrm_sensitivity_ema(
        self,
        signal: EntrySignal,
        ema_distance: float,
        historical_outcomes: Optional[List[Dict]],
    ) -> float:
        if not historical_outcomes or len(historical_outcomes) < 10:
            return 0.2

        ema_values = [d.get("ema_distance", 0.0) for d in historical_outcomes]
        success_flags = [d.get("success", False) for d in historical_outcomes]

        try:
            ema_mean = mean(ema_values)
            ema_std = stdev(ema_values)
        except StatisticsError:
            return 0.2

        if ema_std < 1e-9:
            return 0.0

        score = (ema_distance - ema_mean) / (ema_std ** 2)
        success_rate = sum(success_flags) / len(success_flags)

        delta_lrm = success_rate * score
        return max(-2.0, min(2.0, delta_lrm))

    def _calculate_gamma_atr(
        self,
        signal: EntrySignal,
        atr: float,
    ) -> float:
        epsilon = atr * 0.01 if atr else 0.01
        gamma_approx = -0.1  # placeholder: konservative Short-Gamma-Annahme
        return gamma_approx / max(1e-6, epsilon)

    # ------------------------------------------------------------------
    # Risiko-Aggregation und Statistiken
    # ------------------------------------------------------------------
    def _realized_volatility(self, returns: List[float]) -> float:
        if not returns:
            return 0.0
        arr = np.asarray(returns, dtype=float)
        return float(np.nanstd(arr))

    def _sharpe_estimate(self, returns: List[float]) -> float:
        if not returns:
            return 0.0
        arr = np.asarray(returns, dtype=float)
        mu = float(np.nanmean(arr))
        sigma = float(np.nanstd(arr))
        return (mu / sigma) * math.sqrt(len(arr)) if sigma > 0 else 0.0

    def _var_es(self, returns: List[float]) -> Tuple[float, float]:
        if not returns:
            return 0.0, 0.0
        arr = np.sort(np.asarray(returns, dtype=float))
        idx = int(0.05 * len(arr))
        idx = min(max(idx, 0), len(arr) - 1)
        var = float(arr[idx])
        tail = arr[: idx + 1]
        es = float(np.mean(tail)) if tail.size > 0 else var
        return var, es

    def _max_drawdown_from_returns(self, returns: List[float]) -> float:
        if not returns:
            return 0.0
        equity = 1.0
        peak = 1.0
        max_dd = 0.0
        for r in returns:
            equity *= (1.0 + r)
            peak = max(peak, equity)
            dd = (equity - peak) / peak
            max_dd = min(max_dd, dd)
        return max_dd * 100.0  # Prozent


__all__ = [
    "RiskManager",
    "RiskSensitivities",
    "RiskMetrics",
]