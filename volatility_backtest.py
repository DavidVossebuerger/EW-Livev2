"""
Simplified Volatility Forecasting for Backtesting

Walk-forward volatility prediction using GARCH and HAR models.
Used to adjust position sizing based on expected volatility regime.

Key principle: Higher forecasted volatility -> smaller positions (risk-adjusted)
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings("ignore")

# Try to import arch for GARCH, use fallback if not available
try:
    from arch import arch_model
    ARCH_AVAILABLE = True
except ImportError:
    ARCH_AVAILABLE = False
    print("[VolaForecast] arch not available, using simplified model")


class VolatilityForecaster:
    """
    Walk-forward volatility forecaster for backtesting.
    
    Uses rolling window training to avoid lookahead bias.
    Models: GARCH(1,1) + HAR (Heterogeneous Autoregressive)
    
    Output: volatility_score in range [0, 1]
        0 = very low volatility -> can size up
        0.5 = normal volatility -> standard sizing
        1 = very high volatility -> size down for protection
    """
    
    def __init__(self, 
                 train_window: int = 252,  # 1 year of daily data for training
                 har_weights: Tuple[float, float, float] = (0.6, 0.3, 0.1),  # day, week, month
                 garch_weight: float = 0.4,  # weight for GARCH in ensemble
                 har_weight: float = 0.6):   # weight for HAR in ensemble
        """
        Args:
            train_window: Number of bars to use for model training
            har_weights: Weights for (daily, weekly, monthly) in HAR model
            garch_weight: Weight for GARCH model in final ensemble
            har_weight: Weight for HAR model in final ensemble
        """
        self.train_window = train_window
        self.har_weights = har_weights
        self.garch_weight = garch_weight
        self.har_weight = har_weight
        
        # Caches to avoid retraining every bar
        self._last_train_idx = -1
        self._garch_model = None
        self._har_model = None
        self._har_last_values = None
        
        # Historical volatility stats for percentile calculation
        self._vol_history = []
        self._vol_percentile_lookback = 60  # days for percentile calculation
        
    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare dataframe with volatility features.
        
        Args:
            df: DataFrame with 'open', 'high', 'low', 'close' columns
        
        Returns:
            DataFrame with added volatility columns
        """
        df = df.copy()
        
        # Preserve existing ATR column if present (different cases)
        existing_atr = None
        for col in df.columns:
            if col.lower() == 'atr':
                existing_atr = df[col].copy()
                break
        
        # Ensure column names are lowercase for internal processing
        col_mapping = {c: c.lower() for c in df.columns}
        df.columns = [c.lower() for c in df.columns]
        
        # Log returns (in %)
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1)) * 100
        df['log_returns'] = df['log_returns'].fillna(0)
        
        # Garman-Klass-Yang-Zhang volatility estimator (more efficient than close-to-close)
        log_hl = np.log(df['high'] / df['low'])
        log_co = np.log(df['close'] / df['open'])
        
        gkyz = (0.511 * (log_hl ** 2) - 
                0.019 * (log_co * (2 * log_hl - log_co)) + 
                0.383 * (log_co ** 2))
        
        df['gkyz_vol'] = np.sqrt(gkyz.clip(lower=0)) * 100  # Daily vol in %
        
        # HAR components: daily, weekly (5d), monthly (22d) averages
        df['rv_d'] = df['gkyz_vol']
        df['rv_w'] = df['gkyz_vol'].rolling(5, min_periods=1).mean()
        df['rv_m'] = df['gkyz_vol'].rolling(22, min_periods=1).mean()
        
        # Calculate ATR for volatility reference
        df['atr_vola'] = self._calculate_atr(df, period=14)
        df['atr_pct'] = (df['atr_vola'] / df['close']) * 100
        
        # Restore existing ATR column (needed by backtester)
        if existing_atr is not None:
            df['ATR'] = existing_atr.values
        
        return df
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range."""
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(period, min_periods=1).mean()
    
    def _fit_garch(self, returns: np.ndarray) -> Optional[float]:
        """
        Fit GARCH(1,1) model and return 1-step ahead forecast.
        
        Args:
            returns: Array of log returns
            
        Returns:
            Forecasted daily volatility (%), or None if failed
        """
        if not ARCH_AVAILABLE or len(returns) < 50:
            return None
        
        try:
            # GARCH(1,1) - simple and robust
            model = arch_model(returns, vol='Garch', p=1, q=1, rescale=True)
            result = model.fit(disp='off', show_warning=False)
            
            # Forecast 1 step ahead
            forecast = result.forecast(horizon=1)
            variance = forecast.variance.iloc[-1, 0]
            
            # Return daily volatility (sqrt of variance)
            return np.sqrt(variance)
            
        except Exception as e:
            return None
    
    def _fit_har(self, df: pd.DataFrame, idx: int) -> Optional[float]:
        """
        Fit HAR model and return 1-step ahead forecast.
        
        HAR uses lagged daily, weekly, monthly volatility to predict next day.
        
        Args:
            df: DataFrame with rv_d, rv_w, rv_m columns
            idx: Current index (forecast for idx+1)
            
        Returns:
            Forecasted daily volatility (%), or None if failed
        """
        if idx < self.train_window:
            return None
        
        try:
            # Training data: [idx - train_window : idx]
            train_start = idx - self.train_window
            train_data = df.iloc[train_start:idx][['rv_d', 'rv_w', 'rv_m']].dropna()
            
            if len(train_data) < 30:
                return None
            
            # X: features at time t, y: realized vol at time t+1
            X = train_data[['rv_d', 'rv_w', 'rv_m']].iloc[:-1].values
            y = train_data['rv_d'].iloc[1:].values
            
            # Fit simple linear regression
            model = LinearRegression()
            model.fit(X, y)
            
            # Forecast using latest values
            last_values = df.iloc[idx][['rv_d', 'rv_w', 'rv_m']].values.reshape(1, -1)
            forecast = model.predict(last_values)[0]
            
            return max(0, forecast)  # Volatility cannot be negative
            
        except Exception as e:
            return None
    
    def _simple_forecast(self, df: pd.DataFrame, idx: int) -> float:
        """
        Simple fallback forecast using weighted average of recent volatility.
        
        Args:
            df: DataFrame with gkyz_vol column
            idx: Current index
            
        Returns:
            Forecasted daily volatility (%)
        """
        # Weighted average: more weight on recent observations
        lookback = min(22, idx)
        if lookback < 5:
            return df.iloc[idx]['gkyz_vol']
        
        recent_vol = df.iloc[idx-lookback:idx]['gkyz_vol'].values
        
        # Exponential weights
        weights = np.exp(np.linspace(0, 2, len(recent_vol)))
        weights /= weights.sum()
        
        return np.average(recent_vol, weights=weights)
    
    def forecast_volatility(self, df: pd.DataFrame, idx: int, 
                           retrain_frequency: int = 20) -> Dict:
        """
        Generate volatility forecast at a specific index (walk-forward).
        
        Args:
            df: Prepared DataFrame (from prepare_data)
            idx: Current bar index (we forecast for idx+1)
            retrain_frequency: How often to retrain models (in bars)
            
        Returns:
            Dict with:
                'forecast': forecasted daily vol (%)
                'percentile': where this forecast ranks historically (0-100)
                'regime': 'low', 'normal', or 'high'
                'size_multiplier': suggested position size multiplier (0.5 - 1.5)
        """
        if idx < 30:
            # Not enough data, return neutral
            return {
                'forecast': df.iloc[idx]['gkyz_vol'] if 'gkyz_vol' in df.columns else 1.0,
                'percentile': 50.0,
                'regime': 'normal',
                'size_multiplier': 1.0
            }
        
        # Check if we need to retrain models
        need_retrain = (idx - self._last_train_idx) >= retrain_frequency
        
        forecasts = []
        weights = []
        
        # GARCH forecast
        if need_retrain or self._garch_model is None:
            returns = df.iloc[max(0, idx-self.train_window):idx]['log_returns'].dropna().values
            garch_forecast = self._fit_garch(returns)
        else:
            garch_forecast = self._fit_garch(
                df.iloc[max(0, idx-self.train_window):idx]['log_returns'].dropna().values
            )
        
        if garch_forecast is not None and np.isfinite(garch_forecast):
            forecasts.append(garch_forecast)
            weights.append(self.garch_weight)
        
        # HAR forecast
        har_forecast = self._fit_har(df, idx)
        if har_forecast is not None and np.isfinite(har_forecast):
            forecasts.append(har_forecast)
            weights.append(self.har_weight)
        
        # Fallback if no models worked
        if not forecasts:
            ensemble_forecast = self._simple_forecast(df, idx)
        else:
            # Weighted ensemble
            weights = np.array(weights)
            weights /= weights.sum()
            ensemble_forecast = np.average(forecasts, weights=weights)
        
        # Update training index
        if need_retrain:
            self._last_train_idx = idx
        
        # Calculate percentile (where does this forecast rank?)
        self._vol_history.append(ensemble_forecast)
        
        lookback = min(self._vol_percentile_lookback, len(self._vol_history))
        recent_history = self._vol_history[-lookback:]
        percentile = (np.array(recent_history) < ensemble_forecast).sum() / len(recent_history) * 100
        
        # Determine regime and size multiplier
        if percentile < 25:
            regime = 'low'
            # Low volatility -> can size up slightly (but not too much)
            size_multiplier = 1.15
        elif percentile < 75:
            regime = 'normal'
            size_multiplier = 1.0
        else:
            regime = 'high'
            # High volatility -> size down for protection
            # Scale: 75th percentile = 0.9x, 100th percentile = 0.6x
            size_multiplier = max(0.6, 1.0 - (percentile - 75) / 100)
        
        return {
            'forecast': ensemble_forecast,
            'percentile': percentile,
            'regime': regime,
            'size_multiplier': size_multiplier
        }
    
    def get_position_size_multiplier(self, df: pd.DataFrame, idx: int) -> float:
        """
        Convenience method: get position size multiplier at given index.
        
        Args:
            df: Prepared DataFrame
            idx: Current bar index
            
        Returns:
            Size multiplier (0.6 - 1.15)
        """
        result = self.forecast_volatility(df, idx)
        return result['size_multiplier']
    
    def batch_forecast(self, df: pd.DataFrame, 
                      start_idx: int = None, 
                      end_idx: int = None) -> pd.DataFrame:
        """
        Generate forecasts for a range of indices (for analysis).
        
        Args:
            df: Prepared DataFrame
            start_idx: Starting index (default: train_window)
            end_idx: Ending index (default: len(df))
            
        Returns:
            DataFrame with forecasts for each index
        """
        if start_idx is None:
            start_idx = self.train_window
        if end_idx is None:
            end_idx = len(df)
        
        results = []
        
        for idx in range(start_idx, end_idx):
            forecast = self.forecast_volatility(df, idx)
            forecast['idx'] = idx
            forecast['date'] = df.iloc[idx]['date'] if 'date' in df.columns else idx
            forecast['actual_vol'] = df.iloc[idx]['gkyz_vol']
            results.append(forecast)
        
        return pd.DataFrame(results)


# Convenience function for backtester integration
def create_vola_forecaster(train_window: int = 252) -> VolatilityForecaster:
    """Create a volatility forecaster with default settings."""
    return VolatilityForecaster(
        train_window=train_window,
        garch_weight=0.4,
        har_weight=0.6
    )


if __name__ == "__main__":
    # Test with sample data
    import yfinance as yf
    
    print("Testing Volatility Forecaster...")
    
    # Load sample data
    data = yf.download("QQQ", start="2020-01-01", end="2024-12-31", progress=False)
    data = data.reset_index()
    data.columns = [c.lower() if isinstance(c, str) else c[0].lower() for c in data.columns]
    
    # Initialize forecaster
    forecaster = VolatilityForecaster(train_window=252)
    
    # Prepare data
    df = forecaster.prepare_data(data)
    
    # Test single forecast
    test_idx = 300
    result = forecaster.forecast_volatility(df, test_idx)
    print(f"\nForecast at idx {test_idx}:")
    print(f"  Predicted Vol: {result['forecast']:.2f}%")
    print(f"  Percentile: {result['percentile']:.1f}")
    print(f"  Regime: {result['regime']}")
    print(f"  Size Multiplier: {result['size_multiplier']:.2f}")
    
    # Batch forecast for analysis
    print("\nRunning batch forecast...")
    forecasts_df = forecaster.batch_forecast(df, start_idx=300, end_idx=400)
    
    # Simple accuracy check
    correlation = forecasts_df['forecast'].corr(forecasts_df['actual_vol'])
    print(f"Forecast-Actual Correlation: {correlation:.3f}")
    
    # Regime distribution
    print("\nRegime Distribution:")
    print(forecasts_df['regime'].value_counts(normalize=True))
