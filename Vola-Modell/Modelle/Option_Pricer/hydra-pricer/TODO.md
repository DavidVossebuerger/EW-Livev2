# HYDRA PRICER TODO

## Phase 1 – Core Pricing Engine
- [ ] Implement closed-form Black-Scholes call/put pricing (`src/models/black_scholes.py`)
- [ ] Add analytical Greeks (delta, gamma, vega, theta, rho) plus `get_all_greeks()`
- [ ] Build Cox-Ross-Rubinstein binomial model with early-exercise logic
- [ ] Cross-check European binomial outputs against BSM values

## Phase 2 – Advanced Analytics & Risk Engine
- [ ] Wire `GreeksResult` helpers to BSM implementation
- [ ] Provide finite-difference Greeks for the binomial tree
- [ ] Implement Monte-Carlo GBM path generator (vectorized)
- [ ] Support payoffs: European, Asian average price, Lookback max/min
- [ ] Return price, stderr und 95%-Konfidenzintervall

## Phase 3 – Data & Dashboard
- [ ] Integrate yfinance data fetcher with caching & robust error handling
- [ ] Add historical volatility estimator & risk-free rate proxy
- [ ] Build Streamlit Sidebar/Input-Controls inkl. Auto-Parameter
- [ ] Implement Plotly tabs: Pricing & Greeks, Payoff, Greeks-Grid, Vol Surface
- [ ] Provide status banner + loading spinner UX flows

## Qualität & Infrastruktur
- [ ] Ergänze Helper-Funktionen (e.g. validation, date utilities)
- [ ] Schreiben umfassende pytest-Suite (BSM, Binomial, MC, Greeks)
- [ ] Put-Call-Parity Testcases
- [ ] Monte-Carlo Konvergenztests + Seeds
- [ ] README um Screenshots, Deployment-Hinweise und Architektur erweitern
- [ ] CI/CD Workflow (GitHub Actions) für Format, Lint, Tests
- [ ] Deployment auf Streamlit Community Cloud + Link in README
