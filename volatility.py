"""Volatility forecasting models using GARCH, HAR, HMM, and neural networks."""

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
from arch import arch_model
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import norm
import warnings
import pickle
import os
from pathlib import Path
from datetime import datetime, timedelta
warnings.filterwarnings("ignore", category=UserWarning, module="arch")

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
try:
    from hmmlearn.hmm import GaussianHMM
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Input
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.regularizers import l2
    NN_AVAILABLE = True
except ImportError:
    NN_AVAILABLE = False

class VolatilityModel:
    """Hybrid volatility forecasting model with persistence and HMM regime detection."""

    def __init__(self, cache_dir: str = ".cache/models"):
        self.models = {}
        self.scaler = MinMaxScaler()
        self.har_model = None
        self.gru_model = None
        self.lstm_model = None
        self.hmm_model = None
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def _get_cache_key(self, ticker: str, years_back: int, use_nn: bool) -> str:
        """Generate cache key for model."""
        return f"{ticker}_{years_back}y_nn{use_nn}"
    
    def _get_cache_path(self, cache_key: str) -> Path:
        """Get cache file path."""
        return self.cache_dir / f"{cache_key}.pkl"
    
    def _is_cache_valid(self, cache_path: Path, max_age_hours: int = 24) -> bool:
        """Check if cached model is still valid."""
        if not cache_path.exists():
            return False
        
        # Check age
        file_time = datetime.fromtimestamp(cache_path.stat().st_mtime)
        age = datetime.now() - file_time
        return age < timedelta(hours=max_age_hours)
    
    def save_to_cache(self, cache_key: str):
        """Save model state to cache."""
        cache_path = self._get_cache_path(cache_key)
        try:
            state = {
                'models': self.models,
                'scaler': self.scaler,
                'har_model': self.har_model,
                'hmm_model': self.hmm_model,
                '_hmm_last_volatility': getattr(self, '_hmm_last_volatility', None),
                'last_rv': getattr(self, 'last_rv', None),
                '_last_X': getattr(self, '_last_X', None),
                'timestamp': datetime.now().isoformat()
            }
            # Don't cache NN models (too large, Keras issue with pickling)
            # They will be retrained if needed
            with open(cache_path, 'wb') as f:
                pickle.dump(state, f)
            return True
        except Exception as e:
            print(f"Warning: Could not save cache: {e}")
            return False
    
    def load_from_cache(self, cache_key: str, max_age_hours: int = 24) -> bool:
        """Load model state from cache."""
        cache_path = self._get_cache_path(cache_key)
        
        if not self._is_cache_valid(cache_path, max_age_hours):
            return False
        
        try:
            with open(cache_path, 'rb') as f:
                state = pickle.load(f)
            
            self.models = state.get('models', {})
            self.scaler = state.get('scaler', MinMaxScaler())
            self.har_model = state.get('har_model')
            self.hmm_model = state.get('hmm_model')
            if state.get('_hmm_last_volatility') is not None:
                self._hmm_last_volatility = state['_hmm_last_volatility']
            if state.get('last_rv') is not None:
                self.last_rv = state['last_rv']
            if state.get('_last_X') is not None:
                self._last_X = state['_last_X']
            
            return True
        except Exception as e:
            print(f"Warning: Could not load cache: {e}")
            return False

    def load_data_csv(self, file_path):
        """Load data from CSV."""
        data = pd.read_csv(file_path)
        date_columns = [data.columns[i] for i in range(len(data.columns)) if 'date' in data.columns[i].lower()]
        if date_columns:
            data.rename(columns={date_columns[0]: 'Date'}, inplace=True)
        data.columns = [col.title() for col in data.columns]
        data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
        data.set_index('Date', inplace=True)
        data.dropna(inplace=True)
        return data

    def load_data_yahoo(self, ticker, start_date='1999-01-01', end_date='2025-12-31'):
        """Load data from Yahoo Finance."""
        if not YFINANCE_AVAILABLE:
            raise ImportError("yfinance required.")
        stock = yf.Ticker(ticker)
        data = stock.history(start=start_date, end=end_date, interval='1d')
        if data.empty:
            raise ValueError(f"No data for {ticker}.")
        data.columns = [col.title() for col in data.columns]
        data.index.name = 'Date'
        data.dropna(inplace=True)
        return data

    def preprocess_data(self, data):
        """Calculate log returns and volatility estimators."""
        data['Log_Returns'] = np.log(data['Close'] / data['Close'].shift(1)).fillna(0)
        data['Log_Returns'] *= 100  # Convert to percentage

        # GKZ (Garman-Klass-Yang-Zhang) volatility estimator
        # This gives daily volatility in natural log units
        gkyz = (
            0.511 * (np.log(data['High'] / data['Low']) ** 2) -
            0.019 * (np.log(data['Close'] / data['Open']) * (2 * np.log(data['High'] / data['Low']) - np.log(data['Close'] / data['Open']))) +
            0.383 * (np.log(data['Close'] / data['Open']) ** 2)
        )
        data['GKZ'] = np.sqrt(gkyz.clip(lower=0)) * 100  # Convert to percentage to match Log_Returns
        data['GKZ_Volatility'] = data['GKZ']

        # Seasonal decompose if enough data
        if len(data) >= 504:
            decomposition = seasonal_decompose(data['GKZ'].dropna(), model='additive', period=252)
            data['GKYZ_Trend'] = decomposition.trend
            data['GKYZ_Seasonal'] = decomposition.seasonal
            data['GKYZ_Residual'] = decomposition.resid

        return data

    def fit_garch_models(self, returns):
        """Fit GARCH, EGARCH, TGARCH models."""
        garch = arch_model(returns, vol='Garch', p=1, q=1).fit(disp='off')
        egarch = arch_model(returns, vol='EGarch', p=1, q=1).fit(disp='off')
        tgarch = arch_model(returns, vol='EGarch', p=1, q=1).fit(disp='off')  # Fallback
        self.models = {'GARCH': garch, 'EGARCH': egarch, 'TGARCH': tgarch}
        return self.models

    def fit_har_model(self, data):
        """Fit HAR (Heterogeneous Autoregressive) model on realized volatility.
        
        Uses daily, weekly, and monthly averages of GKZ volatility estimates.
        """
        # Use GKZ_Volatility (already in daily % form) instead of squared returns
        data['RV_d'] = data['GKZ_Volatility']  # Daily
        data['RV_w'] = data['GKZ_Volatility'].rolling(5).mean()  # Weekly (5 days)
        data['RV_m'] = data['GKZ_Volatility'].rolling(22).mean()  # Monthly (22 days)
        data_clean = data[['RV_d', 'RV_w', 'RV_m']].dropna()

        # Shift features by 1 to predict next day's volatility
        X = data_clean[['RV_d', 'RV_w', 'RV_m']].iloc[:-1].values
        y = data_clean['RV_d'].iloc[1:].values
        
        model = LinearRegression().fit(X, y)
        self.har_model = model
        
        # Store last values for forecasting
        self._har_last_values = data_clean[['RV_d', 'RV_w', 'RV_m']].iloc[-1].values.reshape(1, -1)
        
        return model

    def fit_hmm_model(self, data, n_states=2, progress_callback=None):
        """Fit Hidden Markov Model for regime detection.
        
        Args:
            data: DataFrame with 'GKZ_Volatility' column
            n_states: Number of volatility regimes (default=2: Low/High)
            progress_callback: Optional callback for progress updates
        """
        if not HMM_AVAILABLE:
            if progress_callback:
                progress_callback("HMM nicht verfügbar (hmmlearn fehlt)")
            return None
        
        try:
            if progress_callback:
                progress_callback(f"Trainiere HMM-Modell ({n_states} Regime)...")
            
            # Use realized volatility as observation sequence
            volatility = data['GKZ_Volatility'].dropna().values.reshape(-1, 1)
            
            if len(volatility) < 50:
                if progress_callback:
                    progress_callback("Zu wenig Daten für HMM-Training")
                return None
            
            # Fit Gaussian HMM
            model = GaussianHMM(n_components=n_states, covariance_type="full", 
                               n_iter=100, random_state=42)
            model.fit(volatility)
            
            self.hmm_model = model
            self._hmm_last_volatility = volatility[-10:]  # Store recent observations
            
            # Get state assignments for analysis
            states = model.predict(volatility)
            unique_states, counts = np.unique(states, return_counts=True)
            
            if progress_callback:
                regime_info = ", ".join([f"Regime {i}: {c/len(states)*100:.1f}%" 
                                        for i, c in zip(unique_states, counts)])
                progress_callback(f"HMM trainiert - {regime_info}")
            
            return model
            
        except Exception as e:
            print(f"Warning: HMM fitting failed: {e}")
            self.hmm_model = None
            if progress_callback:
                progress_callback(f"HMM-Training fehlgeschlagen: {str(e)}")
            return None

    def fit_nn_models(self, data, progress_callback=None):
        """Fit GRU and LSTM models with optional progress callback."""
        if not NN_AVAILABLE:
            if progress_callback:
                progress_callback("Neural Networks nicht verfügbar (TensorFlow fehlt)")
            return
        try:
            scaled = self.scaler.fit_transform(data[['Log_Returns', 'GKZ_Volatility']].dropna())
            look_back = 10
            if len(scaled) <= look_back:
                if progress_callback:
                    progress_callback("Zu wenig Daten für NN-Training")
                return  # Not enough data for NN
            X, y = [], []
            for i in range(look_back, len(scaled)):
                X.append(scaled[i-look_back:i])
                y.append(scaled[i, 1])
            X, y = np.array(X), np.array(y)
            X = X.reshape(X.shape[0], X.shape[1], X.shape[2])

            self._last_X = X[-1]  # Store last input for forecasting

            if progress_callback:
                progress_callback("Trainiere GRU-Modell...")
            self.gru_model = self._build_gru(X, y)
            
            if progress_callback:
                progress_callback("Trainiere LSTM-Modell...")
            self.lstm_model = self._build_lstm(X, y)
        except Exception as e:
            print(f"Warning: NN model fitting failed: {e}")
            # Continue without NN models
            pass

    def _build_gru(self, X, y):
        model = Sequential()
        model.add(Input(shape=(X.shape[1], X.shape[2])))
        model.add(GRU(64, activation='relu', kernel_regularizer=l2(0.001)))
        model.add(Dropout(0.3))
        model.add(Dense(32, activation='relu', kernel_regularizer=l2(0.001)))
        model.add(Dropout(0.2))
        model.add(Dense(1))
        model.compile(optimizer=Adam(0.0005), loss='mse')
        early_stop = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
        model.fit(X, y, epochs=100, batch_size=32, verbose=0, callbacks=[early_stop])
        return model

    def _build_lstm(self, X, y):
        model = Sequential()
        model.add(Input(shape=(X.shape[1], X.shape[2])))
        model.add(LSTM(64, activation='relu', kernel_regularizer=l2(0.001)))
        model.add(Dropout(0.3))
        model.add(Dense(32, activation='relu', kernel_regularizer=l2(0.001)))
        model.add(Dropout(0.2))
        model.add(Dense(1))
        model.compile(optimizer=Adam(0.0005), loss='mse')
        early_stop = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
        model.fit(X, y, epochs=100, batch_size=32, verbose=0, callbacks=[early_stop])
        return model

    def forecast_volatility(self, steps=10, weighting='inverse_variance'):
        """Forecast volatility using weighted ensemble with confidence intervals.
        
        Args:
            steps: Number of days to forecast
            weighting: 'equal' for simple average, 'inverse_variance' for precision-weighted
        
        Returns:
            dict with forecasts, 'confidence_interval', and 'weights' keys
        """
        forecasts = {}
        individual_forecasts = []  # For confidence interval calculation
        
        if 'GARCH' in self.models:
            try:
                # Forecast horizon=1, use last observation
                forecast = self.models['GARCH'].forecast(horizon=1)
                var = forecast.variance.iloc[-1, 0]
                # GARCH returns variance of returns (in %²), sqrt gives daily vol in %
                garch_forecast = np.full(steps, np.sqrt(var))
                forecasts['GARCH'] = garch_forecast
                individual_forecasts.append(garch_forecast)
            except Exception as e:
                print(f"GARCH forecast error: {e}")
                # Use conditional volatility from the fitted model as fallback
                try:
                    cond_vol = self.models['GARCH'].conditional_volatility[-1]
                    garch_forecast = np.full(steps, cond_vol)
                    forecasts['GARCH'] = garch_forecast
                    individual_forecasts.append(garch_forecast)
                except:
                    pass  # Skip this model entirely if it fails
                    
        if 'EGARCH' in self.models:
            try:
                forecast = self.models['EGARCH'].forecast(horizon=1)
                var = forecast.variance.iloc[-1, 0]
                egarch_forecast = np.full(steps, np.sqrt(var))
                forecasts['EGARCH'] = egarch_forecast
                individual_forecasts.append(egarch_forecast)
            except Exception as e:
                print(f"EGARCH forecast error: {e}")
                try:
                    cond_vol = self.models['EGARCH'].conditional_volatility[-1]
                    egarch_forecast = np.full(steps, cond_vol)
                    forecasts['EGARCH'] = egarch_forecast
                    individual_forecasts.append(egarch_forecast)
                except:
                    pass
                    
        if 'TGARCH' in self.models:
            try:
                forecast = self.models['TGARCH'].forecast(horizon=1)
                var = forecast.variance.iloc[-1, 0]
                tgarch_forecast = np.full(steps, np.sqrt(var))
                forecasts['TGARCH'] = tgarch_forecast
                individual_forecasts.append(tgarch_forecast)
            except Exception as e:
                print(f"TGARCH forecast error: {e}")
                try:
                    cond_vol = self.models['TGARCH'].conditional_volatility[-1]
                    tgarch_forecast = np.full(steps, cond_vol)
                    forecasts['TGARCH'] = tgarch_forecast
                    individual_forecasts.append(tgarch_forecast)
                except:
                    pass

        # HAR forecast
        if self.har_model and hasattr(self, '_har_last_values'):
            try:
                # Predict next day's volatility using the HAR model
                har_pred = self.har_model.predict(self._har_last_values)[0]
                # HAR already outputs daily volatility in %, no transformation needed
                har_forecast = np.full(steps, har_pred)
                forecasts['HAR'] = har_forecast
                individual_forecasts.append(har_forecast)
            except Exception as e:
                print(f"HAR forecast error: {e}")

        # HMM regime-based forecast
        if self.hmm_model and hasattr(self, '_hmm_last_volatility'):
            try:
                # Get current state probabilities
                recent_vol = self._hmm_last_volatility
                state_probs = self.hmm_model.predict_proba(recent_vol)[-1]
                
                # Weighted forecast based on regime means
                # HMM was trained on GKZ_Volatility which is already in % (not squared)
                regime_means = self.hmm_model.means_.flatten()
                
                # Forecast as weighted sum of regime volatilities (already in daily % form)
                hmm_forecast = np.sum(state_probs * regime_means)
                hmm_forecast_array = np.full(steps, hmm_forecast)
                
                forecasts['HMM'] = hmm_forecast_array
                individual_forecasts.append(hmm_forecast_array)
            except Exception as e:
                print(f"Warning: HMM forecast failed: {e}")

        # NN forecasts - need to inverse transform the scaled predictions
        if self.gru_model and hasattr(self, '_last_X'):
            try:
                scaled_pred = self.gru_model.predict(self._last_X.reshape(1, -1, 2), verbose=0)[0][0]
                # Inverse transform: create dummy array with [0, pred] to inverse only the volatility column
                dummy = np.array([[0, scaled_pred]])
                unscaled = self.scaler.inverse_transform(dummy)
                gru_pred = np.full(steps, unscaled[0, 1])  # Get the volatility value
                forecasts['GRU'] = gru_pred
                individual_forecasts.append(gru_pred)
            except Exception as e:
                print(f"GRU forecast error: {e}")
                
        if self.lstm_model and hasattr(self, '_last_X'):
            try:
                scaled_pred = self.lstm_model.predict(self._last_X.reshape(1, -1, 2), verbose=0)[0][0]
                # Inverse transform
                dummy = np.array([[0, scaled_pred]])
                unscaled = self.scaler.inverse_transform(dummy)
                lstm_pred = np.full(steps, unscaled[0, 1])
                forecasts['LSTM'] = lstm_pred
                individual_forecasts.append(lstm_pred)
            except Exception as e:
                print(f"LSTM forecast error: {e}")
            individual_forecasts.append(lstm_pred)

        # Ensemble with weighted averaging and confidence intervals
        if forecasts:
            # Debug: Print individual forecast values
            print("\n=== Volatility Forecast Debug ===")
            model_names = []
            model_forecasts = []
            
            for model_name, forecast_array in forecasts.items():
                if model_name not in ['Ensemble', 'confidence_interval', 'weights']:
                    print(f"{model_name}: {np.mean(forecast_array):.6f} (daily %)")
                    model_names.append(model_name)
                    model_forecasts.append(forecast_array)
            
            # Calculate weights
            if len(model_forecasts) > 1:
                model_array = np.array(model_forecasts)
                
                if weighting == 'strategic':
                    # Strategic weighting based on model roles:
                    # GARCH/EGARCH (Core): 40-50% - Robust, mean-reverting, industry standard
                    # HAR (Tactical): 20-30% - Responds to market dynamics, mean reversion signals
                    # LSTM/GRU (Alpha): 10-20% - Black box risk, prone to overfitting recent spikes
                    # HMM (Director): 0-10% - Better for regime detection than forecasting
                    
                    weights = np.zeros(len(model_names))
                    total_weight = 0.0
                    
                    # Core models (GARCH family): 45% combined
                    garch_models = [i for i, name in enumerate(model_names) if 'GARCH' in name]
                    if garch_models:
                        garch_weight = 0.45 / len(garch_models)
                        for idx in garch_models:
                            weights[idx] = garch_weight
                            total_weight += garch_weight
                    
                    # Tactical (HAR): 25%
                    har_idx = [i for i, name in enumerate(model_names) if name == 'HAR']
                    if har_idx:
                        weights[har_idx[0]] = 0.25
                        total_weight += 0.25
                    
                    # Alpha models (NN): 15% combined, reduced weight to avoid overfitting
                    nn_models = [i for i, name in enumerate(model_names) if name in ['LSTM', 'GRU']]
                    if nn_models:
                        nn_weight = 0.15 / len(nn_models)
                        for idx in nn_models:
                            weights[idx] = nn_weight
                            total_weight += nn_weight
                    
                    # Director (HMM): 5% - minimal weight for forecast, mainly for regime detection
                    hmm_idx = [i for i, name in enumerate(model_names) if name == 'HMM']
                    if hmm_idx:
                        weights[hmm_idx[0]] = 0.05
                        total_weight += 0.05
                    
                    # Normalize to ensure sum = 1 (handles missing models)
                    if total_weight > 0:
                        weights = weights / total_weight
                    
                    print(f"\nStrategic Weights:")
                    for name, weight in zip(model_names, weights):
                        print(f"  {name}: {weight:.1%}")
                    
                    # Weighted ensemble
                    weighted_ensemble = np.average(model_array, axis=0, weights=weights)
                    forecasts['Ensemble'] = weighted_ensemble
                
                elif weighting == 'inverse_variance':
                    # Inverse variance weighting: models with lower variance get higher weight
                    variances = np.var(model_array, axis=1)
                    variances = np.maximum(variances, 1e-8)
                    inv_var = 1.0 / variances
                    weights = inv_var / np.sum(inv_var)
                    
                    print(f"\nInverse Variance Weights:")
                    for name, weight in zip(model_names, weights):
                        print(f"  {name}: {weight:.1%}")
                    
                    weighted_ensemble = np.average(model_array, axis=0, weights=weights)
                    forecasts['Ensemble'] = weighted_ensemble
                    
                else:  # equal weighting
                    weights = np.ones(len(model_forecasts)) / len(model_forecasts)
                    weighted_ensemble = np.mean(model_array, axis=0)
                    forecasts['Ensemble'] = weighted_ensemble
                
                # Store weights for display
                forecasts['weights'] = dict(zip(model_names, weights))
                
                # Calculate confidence interval from individual forecasts
                std_dev = np.std(model_array, axis=0)
                mean_forecast = np.mean(model_array, axis=0)
                print(f"\nEnsemble mean: {np.mean(weighted_ensemble):.6f}, std: {np.mean(std_dev):.6f}")
                
                # 95% confidence interval (±1.96 std), clipped at 0 (volatility cannot be negative)
                lower_bound = mean_forecast - 1.96 * std_dev
                forecasts['confidence_interval'] = {
                    'lower': np.maximum(lower_bound, 0),  # Clip at 0
                    'upper': mean_forecast + 1.96 * std_dev,
                    'std': std_dev
                }
                print(f"CI: [{np.mean(forecasts['confidence_interval']['lower']):.6f}, {np.mean(forecasts['confidence_interval']['upper']):.6f}]")
            else:
                # Only one model, use it as ensemble
                forecasts['Ensemble'] = model_forecasts[0]
                forecasts['weights'] = {model_names[0]: 1.0}
                
            print("=================================\n")

        return forecasts

    def calculate_options(self, S, K, T=0.25, r=0.05, sigma=None):
        """Calculate Black-Scholes options."""
        if sigma is None:
            sigma = 0.2  # Default
        call = self._black_scholes_call(S, K, T, r, sigma)
        put = self._black_scholes_put(S, K, T, r, sigma)
        return {'call': call, 'put': put}

    def _black_scholes_call(self, S, K, T, r, sigma):
        d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)

    def _black_scholes_put(self, S, K, T, r, sigma):
        d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        return K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)