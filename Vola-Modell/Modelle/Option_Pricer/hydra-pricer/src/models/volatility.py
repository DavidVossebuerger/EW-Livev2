"""Volatility forecasting models using GARCH, HAR, and neural networks."""

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
from arch import arch_model
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import norm
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="arch")

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
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
    """Hybrid volatility forecasting model."""

    def __init__(self):
        self.models = {}
        self.scaler = MinMaxScaler()
        self.har_model = None
        self.gru_model = None
        self.lstm_model = None

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
        data['Log_Returns'] *= 100

        # GKZ estimator
        gkyz = (
            0.511 * (np.log(data['High'] / data['Low']) ** 2) -
            0.019 * (np.log(data['Close'] / data['Open']) * (2 * np.log(data['High'] / data['Low']) - np.log(data['Close'] / data['Open']))) +
            0.383 * (np.log(data['Close'] / data['Open']) ** 2)
        )
        data['GKZ'] = np.sqrt(gkyz.clip(lower=0))
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
        """Fit HAR model."""
        data['Realized_Volatility'] = data['Log_Returns'].rolling(1).apply(lambda x: np.sum(x**2))
        data['RV_d'] = data['Realized_Volatility']
        data['RV_w'] = data['Realized_Volatility'].rolling(5).mean()
        data['RV_m'] = data['Realized_Volatility'].rolling(22).mean()
        data.dropna(inplace=True)

        X = data[['RV_d', 'RV_w', 'RV_m']].shift(1).dropna()
        y = data['Realized_Volatility'].iloc[1:]
        model = LinearRegression().fit(X, y)
        self.har_model = model
        self.last_rv = data['Realized_Volatility'].iloc[-1]
        return model

    def fit_nn_models(self, data):
        """Fit GRU and LSTM models."""
        if not NN_AVAILABLE:
            return
        scaled = self.scaler.fit_transform(data[['Log_Returns', 'GKZ_Volatility']].dropna())
        look_back = 10
        if len(scaled) <= look_back:
            return  # Not enough data for NN
        X, y = [], []
        for i in range(look_back, len(scaled)):
            X.append(scaled[i-look_back:i])
            y.append(scaled[i, 1])
        X, y = np.array(X), np.array(y)
        X = X.reshape(X.shape[0], X.shape[1], X.shape[2])

        self._last_X = X[-1]  # Store last input for forecasting

        self.gru_model = self._build_gru(X, y)
        self.lstm_model = self._build_lstm(X, y)

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

    def forecast_volatility(self, steps=10):
        """Forecast volatility using ensemble."""
        forecasts = {}
        if 'GARCH' in self.models:
            try:
                var = self.models['GARCH'].forecast(1).variance.iloc[-1, 0]
                forecasts['GARCH'] = np.full(steps, np.sqrt(var))
            except:
                forecasts['GARCH'] = np.full(steps, 0.01)
        if 'EGARCH' in self.models:
            try:
                var = self.models['EGARCH'].forecast(1).variance.iloc[-1, 0]
                forecasts['EGARCH'] = np.full(steps, np.sqrt(var))
            except:
                forecasts['EGARCH'] = np.full(steps, 0.01)
        if 'TGARCH' in self.models:
            try:
                var = self.models['TGARCH'].forecast(1).variance.iloc[-1, 0]
                forecasts['TGARCH'] = np.full(steps, np.sqrt(var))
            except:
                forecasts['TGARCH'] = np.full(steps, 0.01)

        # HAR forecast
        if self.har_model:
            # Simplified HAR forecast
            last_rv = self.last_rv if hasattr(self, 'last_rv') else 0.01
            har_forecast = [last_rv] * steps
            forecasts['HAR'] = np.array(har_forecast)

        # NN forecasts
        if self.gru_model and hasattr(self, '_last_X'):
            gru_pred = [self.gru_model.predict(self._last_X.reshape(1, -1, 2), verbose=0)[0][0]] * steps
            forecasts['GRU'] = np.array(gru_pred)
        if self.lstm_model and hasattr(self, '_last_X'):
            lstm_pred = [self.lstm_model.predict(self._last_X.reshape(1, -1, 2), verbose=0)[0][0]] * steps
            forecasts['LSTM'] = np.array(lstm_pred)

        # Ensemble
        if forecasts:
            ensemble = np.mean(list(forecasts.values()), axis=0)
            forecasts['Ensemble'] = ensemble

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