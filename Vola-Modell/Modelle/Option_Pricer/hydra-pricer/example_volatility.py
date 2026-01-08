"""Example usage of VolatilityModel in the hydra-pricer."""

from src.models import VolatilityModel

# Initialize the model
vol_model = VolatilityModel()

# Load data (example with CSV)
# data = vol_model.load_data_csv('path/to/data.csv')
# Or from Yahoo Finance
data = vol_model.load_data_yahoo('AAPL', '2020-01-01', '2023-01-01')

# Preprocess data
data = vol_model.preprocess_data(data)

# Fit models
vol_model.fit_garch_models(data['Log_Returns'])
vol_model.fit_har_model(data)
vol_model.fit_nn_models(data)

# Forecast volatility
forecasts = vol_model.forecast_volatility(steps=5)
print("Volatility forecasts:", forecasts)

# Use in options pricing
S = data['Close'].iloc[-1]  # Current price
K = S * 1.05  # Strike price
options = vol_model.calculate_options(S, K, sigma=forecasts['Ensemble'][0])
print("Call price:", options['call'])
print("Put price:", options['put'])