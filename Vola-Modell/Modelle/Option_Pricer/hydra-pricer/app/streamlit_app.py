"""Streamlit dashboard entry point for HYDRA PRICER."""
from __future__ import annotations

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings

import sys
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# Suppress TensorFlow/Keras warnings further
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('keras').setLevel(logging.ERROR)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # Ensure "src" package is importable when running via Streamlit
    sys.path.append(str(ROOT))

from src.analytics.backtesting import delta_hedge_backtest
from src.analytics.calibration import implied_volatility
from src.data import get_historical_volatility, get_price_history, get_risk_free_rate, get_stock_price
from src.greeks import as_dict, compute_binomial_greeks, compute_bsm_greeks
from src.models import BinomialModel, BlackScholesModel, MonteCarloModel, VolatilityModel

PLOTLY_CONFIG = {
    "displaylogo": False,
    "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"],
    "responsive": True,
}

GREEK_EXPLANATIONS = {
    "delta": "Delta misst die Preisänderung pro 1 USD Spotbewegung und ist das Fundament jeder Hedge-Entscheidung.",
    "gamma": "Gamma beschreibt, wie schnell sich Delta bei Spotbewegungen ändert – wichtig für große Sprünge.",
    "vega": "Vega zeigt die Preissensitivität gegenüber Veränderungen der impliziten Volatilität.",
    "theta": "Theta entspricht dem täglichen Zeitwertverfall. Negative Werte bedeuten, dass die Option täglich an Wert verliert.",
    "rho": "Rho misst die Zinssensitivität und gewinnt bei langen Laufzeiten an Bedeutung.",
}

FOCUS_GUIDE = {
    "Preis": "Der Optionspreis bündelt alle Parameter. Unterschiede zeigen direkt, wie stark ein Szenario wirkt.",
    "Delta": GREEK_EXPLANATIONS["delta"],
    "Gamma": GREEK_EXPLANATIONS["gamma"],
    "Vega": GREEK_EXPLANATIONS["vega"],
    "Theta": GREEK_EXPLANATIONS["theta"],
    "Rho": GREEK_EXPLANATIONS["rho"],
}


def _render_plotly_chart(fig: go.Figure, *, key: str | None = None) -> None:
    """Uniform Plotly rendering with config-based layout."""

    st.plotly_chart(fig, config=PLOTLY_CONFIG, key=key)


SCENARIO_PRESETS = {
    "Bullish Momentum": {
        "spot_shift": 0.08,
        "sigma_mult": 0.9,
        "strike_shift": 0.0,
        "maturity_shift_days": -15,
        "rate_shift": 0.0,
        "description": "Spot steigt nach einem Breakout, Volatilität fällt leicht durch erwartete Trendfortsetzung.",
    },
    "Bearish Hedge": {
        "spot_shift": -0.07,
        "sigma_mult": 1.2,
        "strike_shift": 0.02,
        "maturity_shift_days": 30,
        "rate_shift": 0.0,
        "description": "Absicherungsszenario mit fallendem Spot, höherer impliziter Volatilität und längerer Laufzeit.",
    },
    "Volatility Spike": {
        "spot_shift": 0.0,
        "sigma_mult": 1.6,
        "strike_shift": 0.0,
        "maturity_shift_days": 0,
        "rate_shift": 0.0,
        "description": "Event-Risiko treibt die implizite Volatilität nach oben bei unverändertem Spot.",
    },
}


@st.cache_data(ttl=300)
def _cached_market_data(ticker: str) -> float:
    spot = get_stock_price(ticker)
    return spot


@st.cache_data(ttl=3600)  # Cache for 1 hour
def _cached_volatility_forecast(ticker: str, maturity_days: int, version=4) -> float:
    """Get forecasted volatility using the hybrid model for the given maturity."""
    vol_model = VolatilityModel()
    try:
        data = vol_model.load_data_yahoo(ticker, '2020-01-01', '2023-01-01')
        data = vol_model.preprocess_data(data)
        vol_model.fit_garch_models(data['Log_Returns'])
        vol_model.fit_har_model(data)
        vol_model.fit_nn_models(data)
        steps = min(int(maturity_days), 252)  # Cap at 1 year for performance
        forecasts = vol_model.forecast_volatility(steps=steps)
        # Use ensemble forecast, average over the period, annualize
        ensemble_forecast = forecasts.get('Ensemble', np.full(steps, 0.2))
        avg_vol = np.mean(ensemble_forecast)
        # Annualize: since data is daily, multiply by sqrt(252), then /100 for decimal
        annualized_vol = avg_vol * np.sqrt(252) / 100
        return annualized_vol
    except Exception as e:
        print(f"Error in volatility forecast for {ticker}: {e}")
        import traceback
        traceback.print_exc()
        return 0.2  # Fallback to 20%


@st.cache_data(ttl=300)
def _cached_risk_free_rate() -> float:
    return get_risk_free_rate()


def _payoff_key(payoff_family: str, option_type: str) -> str:
    family = payoff_family.lower()
    opt = option_type.lower()
    return {
        ("european", "call"): "european_call",
        ("european", "put"): "european_put",
        ("asian", "call"): "asian_call",
        ("asian", "put"): "asian_put",
        ("lookback", "call"): "lookback_call",
        ("lookback", "put"): "lookback_put",
    }[(family, opt)]


def _compute_break_even(option_type: str, strike: float, premium: float) -> float:
    if option_type == "call":
        return strike + premium
    return max(strike - premium, 0.0)


def _build_payoff_curve(
    option_type: str, strike: float, premium: float, prices: np.ndarray
) -> np.ndarray:
    if option_type == "call":
        return np.maximum(prices - strike, 0.0) - premium
    return np.maximum(strike - prices, 0.0) - premium


def _greeks_plot_data(spot: float, strike: float, maturity: float, rate: float, sigma: float, option_type: str) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    s_range = np.linspace(max(1.0, spot * 0.5), spot * 1.5, 80)
    v_start = max(0.05, sigma * 0.5)
    v_end = max(v_start + 0.1, sigma * 1.5)
    v_range = np.arange(v_start, v_end + 1e-9, 0.1)
    if v_range.size < 2:
        v_range = np.linspace(v_start, v_end, 40)
    t_range = np.linspace(max(0.05, maturity * 0.2), maturity * 1.5, 80)

    delta_vals, gamma_vals = [], []
    for s_val in s_range:
        model = BlackScholesModel(spot=s_val, strike=strike, maturity=maturity, rate=rate, volatility=sigma)
        delta_vals.append(model.delta(option_type))
        gamma_vals.append(model.gamma())

    vega_vals = []
    for vol in v_range:
        model = BlackScholesModel(spot=spot, strike=strike, maturity=maturity, rate=rate, volatility=vol)
        vega_vals.append(model.vega())

    theta_vals = []
    for t_val in t_range:
        model = BlackScholesModel(spot=spot, strike=strike, maturity=t_val, rate=rate, volatility=sigma)
        theta_vals.append(model.theta(option_type))

    return {
        "delta": (s_range, np.array(delta_vals)),
        "gamma": (s_range, np.array(gamma_vals)),
        "vega": (v_range, np.array(vega_vals)),
        "theta": (t_range, np.array(theta_vals)),
    }


@st.cache_data(ttl=600)
def _dummy_vol_surface(base_sigma: float, spot: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    strikes = np.linspace(0.8 * spot, 1.2 * spot, 20)
    maturity_days = np.arange(10, 370, 10)
    ks, ts = np.meshgrid(strikes, maturity_days / 365)
    surface = base_sigma * (1 + 0.25 * (ks / spot - 1)) * (1 - 0.15 * (ts - ts.mean()))
    surface = np.clip(surface, 0.05, 1.5)
    return strikes, maturity_days, surface


def _comparison_table(rows: Iterable[Dict[str, float]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame()
    df = df.set_index("Modell")
    return df


def main() -> None:
    st.set_page_config(page_title="HYDRA PRICER", layout="wide", page_icon="💹")
    st.title("HYDRA PRICER – Institutional Options Analytics")
    st.caption("Real-time pricing, Greeks und Visualisierungen für institutionelle Workflows")

    st.markdown(
        """
        <style>
        .metric-container {background-color: #0f172a; padding: 12px 16px; border-radius: 12px; color: #f8fafc;}
        .metric-container h3 {color: #cbd5f5; font-size: 0.95rem; margin-bottom: 4px;}
        .metric-container span {font-size: 1.6rem; font-weight: 600;}
        </style>
        """,
        unsafe_allow_html=True,
    )

    sidebar = st.sidebar
    sidebar.header("Input-Parameter")

    ticker_input = sidebar.text_input(
        "Ticker-Symbol",
        value="AAPL",
        help="Börsenkürzel des Underlyings (z. B. AAPL, MSFT).",
        key="ticker_input",
    )
    ticker = ticker_input.upper().strip()
    option_type_label = sidebar.radio(
        "Option-Typ",
        ("Call", "Put"),
        help="Call profitiert von steigenden Kursen, Put von fallenden Kursen.",
        key="option_type_radio",
    )
    option_type = option_type_label.lower()
    strike = sidebar.number_input(
        "Strike K",
        min_value=0.01,
        value=150.0,
        step=1.0,
        help="Ausübungspreis der Option. Höherer Strike senkt Call-Preis und erhöht Put-Preis.",
        key="strike_input",
    )
    maturity_days = sidebar.slider(
        "Restlaufzeit T in Tagen",
        min_value=1,
        max_value=365,
        value=90,
        help="Restliche Zeit bis zur Fälligkeit in Tagen.",
        key="maturity_days_slider",
    )
    maturity = maturity_days / 365

    # Volatilität wird automatisch aus dem hybriden Modell prognostiziert
    rate_mode = sidebar.radio(
        "Risikofreier Zinssatz r",
        ("Auto", "Manuell"),
        help="Proxy aus US-Treasuries oder eigener Wert.",
        key="rate_mode_radio",
    )
    manual_rate = sidebar.slider(
        "r (manuell)",
        min_value=0.0,
        max_value=0.1,
        value=0.02,
        step=0.001,
        help="Jährliche Verzinsung des risikofreien Assets.",
        key="manual_rate_slider",
    )

    model_choice = sidebar.selectbox("Pricing-Modell", ("Black-Scholes", "Binomial", "Monte Carlo"))

    with sidebar.expander("Binomial Einstellungen", expanded=model_choice == "Binomial"):
        binom_steps = st.slider(
            "Zeitschritte",
            min_value=25,
            max_value=2000,
            value=500,
            step=25,
            help="Mehr Schritte erhöhen die Genauigkeit, aber auch die Rechenzeit.",
            key="binom_steps_slider",
        )
        binom_exercise = st.radio(
            "Ausübungsart",
            ("European", "American"),
            horizontal=True,
            help="American erlaubt vorzeitige Ausübung.",
            key="binom_exercise_radio",
        )

    with sidebar.expander("Monte-Carlo Einstellungen", expanded=model_choice == "Monte Carlo"):
        mc_payoff_family = st.selectbox(
            "Payoff-Struktur",
            ("European", "Asian", "Lookback"),
            help="Pfadabhängige Payoffs erweitern den Anwendungsbereich.",
            key="mc_payoff_select",
        )
        mc_sims = st.slider(
            "Anzahl Simulationen",
            min_value=2_000,
            max_value=100_000,
            value=20_000,
            step=2_000,
            help="Mehr Simulationen stabilisieren den Preis, kosten aber Rechenzeit.",
            key="mc_sims_slider",
        )
        mc_steps = st.slider(
            "Zeitschritte pro Pfad",
            min_value=50,
            max_value=365,
            value=252,
            step=5,
            help="Zeitdiskretisierung pro Pfad. Höher = feinere Pfade.",
            key="mc_steps_slider",
        )
        mc_seed = st.number_input(
            "Zufalls-Seed",
            min_value=0,
            value=123,
            step=1,
            help="Fixiere Seed für reproduzierbare Resultate.",
            key="mc_seed_input",
        )

    market_spot = None
    if ticker:
        try:
            with st.spinner("Lade Marktdaten..."):
                market_spot = _cached_market_data(ticker)
        except Exception as exc:  # pragma: no cover - network interaction
            st.warning(f"Marktdaten konnten nicht geladen werden ({exc}). Manuelle Werte verwenden.")

    spot_fallback = sidebar.number_input(
        "Spot (Fallback)",
        min_value=0.01,
        value=150.0,
        step=1.0,
        help="Manueller Spot-Wert, falls kein Live-Preis geladen werden kann.",
        key="spot_fallback_input",
    )
    spot = market_spot if market_spot is not None else spot_fallback

    sigma = _cached_volatility_forecast(ticker, maturity_days, version=4)
    if sigma <= 0:
        st.warning("Volatilität wurde <= 0 gesetzt – fallback auf 0.2")
        sigma = 0.2

    if rate_mode == "Auto":
        try:
            rate = _cached_risk_free_rate()
        except Exception as exc:  # pragma: no cover
            st.warning(f"Risikofreier Zinssatz konnte nicht geladen werden ({exc}).")
            rate = manual_rate
    else:
        rate = manual_rate

    base_model = BlackScholesModel(spot=spot, strike=strike, maturity=maturity, rate=rate, volatility=sigma)
    bsm_price = base_model.call_price() if option_type == "call" else base_model.put_price()
    g_table = as_dict(compute_bsm_greeks(base_model, option_type=option_type))

    results = [{"Modell": "Black-Scholes", "Preis": bsm_price}]

    # Binomial price
    try:
        binom_model = BinomialModel(
            spot=spot,
            strike=strike,
            maturity=maturity,
            rate=rate,
            volatility=sigma,
            steps=binom_steps,
            option_type=option_type,
            exercise=binom_exercise.lower(),
        )
        binom_price = binom_model.price()
        results.append({"Modell": f"Binomial ({binom_exercise})", "Preis": binom_price})
    except Exception as exc:  # pragma: no cover - invalid combos
        binom_price = None
        st.error(f"Binomial-Modell fehlgeschlagen: {exc}")

    # Monte Carlo price
    mc_summary = None
    try:
        payoff_key = _payoff_key(mc_payoff_family, option_type)
        mc_model = MonteCarloModel(
            spot=spot,
            strike=strike,
            maturity=maturity,
            rate=rate,
            volatility=sigma,
            n_simulations=mc_sims,
            n_steps=mc_steps,
            payoff=payoff_key,
            rng_seed=mc_seed,
        )
        mc_summary = mc_model.summarize()
        mc_price = float(mc_summary["price"])
        results.append({"Modell": f"Monte Carlo ({mc_payoff_family})", "Preis": mc_price})
    except Exception as exc:  # pragma: no cover
        mc_model = None
        mc_price = None
        st.error(f"Monte-Carlo-Modell fehlgeschlagen: {exc}")

    selected_price = {
        "Black-Scholes": bsm_price,
        "Binomial": binom_price,
        "Monte Carlo": mc_price,
    }.get(model_choice)

    if selected_price is None:
        selected_price = bsm_price

    if model_choice == "Binomial" and binom_price is not None:
        greeks_source = compute_binomial_greeks(binom_model)
        g_table = as_dict(greeks_source)
    elif model_choice == "Monte Carlo":
        st.info("Monte-Carlo-Greeks werden aktuell analytisch über Black-Scholes approximiert.")

    comparison_df = _comparison_table(results)

    pricing_tab, payoff_tab, greeks_tab, surface_tab, calibration_tab, backtest_tab, education_tab = st.tabs(
        [
            "Pricing & Greeks",
            "Payoff-Diagramm",
            "Greeks-Visualisierung",
            "Volatility Surface",
            "Calibration",
            "Backtesting",
            "Education Mode",
        ]
    )

    with pricing_tab:
        cols = st.columns(3)
        cols[0].markdown(
            f"<div class='metric-container'><h3>Aktueller Spot</h3><span>{spot:,.2f}</span></div>",
            unsafe_allow_html=True,
        )
        cols[1].markdown(
            f"<div class='metric-container'><h3>Optionpreis ({model_choice})</h3><span>{selected_price:,.2f}</span></div>",
            unsafe_allow_html=True,
        )
        cols[2].markdown(
            f"<div class='metric-container'><h3>Volatilität σ</h3><span>{sigma:.2%}</span></div>",
            unsafe_allow_html=True,
        )

        greek_df = pd.DataFrame([g_table]).T.rename(columns={0: "Wert"})
        st.subheader("Greeks")
        st.dataframe(greek_df.style.format({"Wert": "{:.4f}"}), width="stretch")

        st.subheader("Modellvergleich")
        if not comparison_df.empty:
            st.dataframe(comparison_df.style.format({"Preis": "{:.4f}"}), width="stretch")
        else:
            st.write("Keine Vergleichsdaten verfügbar.")

        if model_choice == "Monte Carlo" and mc_summary is not None:
            st.write(
                f"Monte-Carlo-Schätzung ± 1.96σ: {mc_summary['conf_interval'][0]:.2f} bis {mc_summary['conf_interval'][1]:.2f}"
            )

    with payoff_tab:
        st.subheader("Payoff bei Laufzeitende")
        price_grid = np.linspace(max(0.0, spot * 0.2), spot * 2.0, 200)
        payoff_values = _build_payoff_curve(option_type, strike, selected_price, price_grid)
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(x=price_grid, y=payoff_values, mode="lines", name="Payoff", line=dict(color="#10b981", width=3))
        )
        break_even = _compute_break_even(option_type, strike, selected_price)
        fig.add_vline(x=break_even, line=dict(color="#f97316", dash="dash"), annotation_text="Break-even")
        fig.update_layout(
            xaxis_title="Aktienkurs bei Fälligkeit",
            yaxis_title="Gewinn / Verlust",
            template="plotly_dark",
            height=450,
        )
        _render_plotly_chart(fig, key="payoff_chart")

    with greeks_tab:
        st.subheader("Sensitivitätsprofile")
        data = _greeks_plot_data(spot, strike, maturity, rate, sigma, option_type)
        subplots = make_subplots(rows=2, cols=2, subplot_titles=("Delta vs. Spot", "Gamma vs. Spot", "Vega vs. σ", "Theta vs. T"))
        subplots.add_trace(go.Scatter(x=data["delta"][0], y=data["delta"][1], name="Delta"), row=1, col=1)
        subplots.add_trace(go.Scatter(x=data["gamma"][0], y=data["gamma"][1], name="Gamma"), row=1, col=2)
        subplots.add_trace(go.Scatter(x=data["vega"][0], y=data["vega"][1], name="Vega"), row=2, col=1)
        subplots.add_trace(go.Scatter(x=data["theta"][0], y=data["theta"][1], name="Theta"), row=2, col=2)
        subplots.update_xaxes(title_text="Spot", row=1, col=1)
        subplots.update_yaxes(title_text="Delta", row=1, col=1)
        subplots.update_xaxes(title_text="Spot", row=1, col=2)
        subplots.update_yaxes(title_text="Gamma", row=1, col=2)
        subplots.update_xaxes(title_text="Volatilität σ", dtick=0.1, row=2, col=1)
        subplots.update_yaxes(title_text="Vega", row=2, col=1)
        subplots.update_xaxes(title_text="Restlaufzeit T (Jahre)", row=2, col=2)
        subplots.update_yaxes(title_text="Theta", row=2, col=2)
        subplots.update_layout(template="plotly_dark", height=600, legend=dict(orientation="h", yanchor="bottom", y=-0.15))
        _render_plotly_chart(subplots, key="greeks_sensitivity_chart")

    with surface_tab:
        st.subheader("Dummy Volatility Surface")
        surface_strikes, surface_maturity_days, surface = _dummy_vol_surface(sigma, spot)
        heatmap = go.Figure(
            data=go.Heatmap(
                z=surface,
                x=surface_strikes,
                y=surface_maturity_days,
                colorscale="Viridis",
                colorbar=dict(title="σ"),
            )
        )
        heatmap.update_layout(
            xaxis_title="Strike",
            yaxis_title="Restlaufzeit (Tage)",
            template="plotly_dark",
            height=500,
        )
        _render_plotly_chart(heatmap, key="vol_surface_chart")

    with calibration_tab:
        st.subheader("Implied Volatility Calibration")
        st.caption("Kalibriere σ anhand eines beobachteten Marktpreises mit Newton-Raphson.")

        default_market_price = float(selected_price) if selected_price is not None else float(bsm_price)
        left_col, right_col = st.columns(2)
        with left_col:
            market_price_input = st.number_input(
                "Beobachteter Marktpreis",
                min_value=0.0,
                value=round(default_market_price, 4),
                step=0.1,
                help="Marktpreis der Option, gegen den kalibriert wird.",
            )
            initial_guess = st.slider(
                "Startwert σ",
                min_value=0.01,
                max_value=2.0,
                value=float(sigma),
                step=0.01,
            )
        with right_col:
            tol_input = st.number_input(
                "Toleranz",
                min_value=1e-8,
                max_value=1e-2,
                value=1e-6,
                step=1e-6,
                format="%.1e",
                help="Absolutes Abbruchkriterium für den Preisfehler.",
            )
            max_iter = st.number_input(
                "Maximale Iterationen",
                min_value=10,
                max_value=2000,
                value=200,
                step=10,
            )

        if st.button("Kalibriere σ", type="primary"):
            try:
                calib_result = implied_volatility(
                    option_price=float(market_price_input),
                    spot=float(spot),
                    strike=float(strike),
                    maturity=float(maturity),
                    rate=float(rate),
                    option_type=option_type,
                    initial_guess=float(initial_guess),
                    tol=float(tol_input),
                    max_iter=int(max_iter),
                )

                delta_vs_input = calib_result.implied_vol - sigma
                metrics = st.columns(3)
                metrics[0].metric("Implied σ", f"{calib_result.implied_vol:.2%}", f"{delta_vs_input:.2%}")
                metrics[1].metric("Iterationen", calib_result.iterations)
                metrics[2].metric("Konvergenz", "Ja" if calib_result.converged else "Nein")

                if calib_result.converged:
                    st.success("Kalibrierung konvergent.")
                else:
                    st.warning("Kalibrierung hat das Toleranzkriterium nicht erreicht – Ergebnis mit Vorsicht verwenden.")

                st.markdown(
                    "**Hinweis:** Delta zeigt die Abweichung zum aktuell genutzten σ. Nutze diesen Wert, um Modelle oder Oberflächen zu aktualisieren."
                )
            except Exception as exc:  # pragma: no cover - user input edge cases
                st.error(f"Kalibrierung fehlgeschlagen: {exc}")

    with backtest_tab:
        st.subheader("Delta-Hedging Backtest")
        st.caption("Simuliere tägliche Neugewichtungen einer Delta-Hedge-Strategie auf Basis historischer Preise.")

        bt_cols = st.columns(2)
        bt_window = bt_cols[0].slider("Historienfenster (Tage)", min_value=30, max_value=252, value=120, step=10)
        bt_maturity_days = bt_cols[0].slider(
            "Restlaufzeit im Hedge (Tage)",
            min_value=5,
            max_value=365,
            value=min(maturity_days, 180),
            step=5,
        )
        bt_vol = bt_cols[1].slider(
            "Volatilität für Hedge σ",
            min_value=0.05,
            max_value=1.5,
            value=float(sigma),
            step=0.01,
        )
        bt_rate = bt_cols[1].number_input(
            "Risikofreier Satz r (Backtest)",
            min_value=0.0,
            max_value=0.2,
            value=float(rate),
            step=0.001,
            format="%.3f",
        )

        if st.button("Backtest starten", type="primary"):
            chosen_ticker = ticker or "AAPL"
            try:
                with st.spinner("Lade Preisverlauf..."):
                    price_history = get_price_history(chosen_ticker, window_days=int(bt_window))
                result = delta_hedge_backtest(
                    prices=price_history,
                    strike=float(strike),
                    rate=float(bt_rate),
                    volatility=float(bt_vol),
                    maturity_days=int(bt_maturity_days),
                    option_type=option_type,
                )
            except Exception as exc:  # pragma: no cover - network/input issues
                st.error(f"Backtest konnte nicht ausgeführt werden: {exc}")
            else:
                pnl_delta = result.pnl
                history = result.history.copy()
                metrics = st.columns(3)
                metrics[0].metric("PNL", f"{pnl_delta:,.2f}")
                metrics[1].metric("Finales Portfolio", f"{result.final_portfolio_value:,.2f}")
                metrics[2].metric("Hedge-Schritte", len(history))

                perf_fig = go.Figure()
                perf_fig.add_trace(
                    go.Scatter(x=history.index, y=history["spot"], name="Spot", line=dict(color="#10b981", width=3))
                )
                perf_fig.add_trace(
                    go.Scatter(
                        x=history.index,
                        y=history["option_value"],
                        name="Optionwert",
                        line=dict(color="#6366f1", width=2, dash="dot"),
                        yaxis="y2",
                    )
                )
                perf_fig.update_layout(
                    template="plotly_dark",
                    height=450,
                    yaxis=dict(title="Spot"),
                    yaxis2=dict(title="Optionwert", overlaying="y", side="right"),
                    legend=dict(orientation="h", yanchor="bottom", y=-0.2),
                )
                _render_plotly_chart(perf_fig, key="backtest_perf_chart")

                delta_fig = go.Figure()
                delta_fig.add_trace(go.Scatter(x=history.index, y=history["delta"], name="Delta"))
                delta_fig.add_trace(
                    go.Scatter(x=history.index, y=history["shares"], name="Hedge-Position", line=dict(dash="dot"))
                )
                delta_fig.update_layout(template="plotly_dark", height=350, legend=dict(orientation="h"))
                _render_plotly_chart(delta_fig, key="backtest_delta_chart")

                st.markdown("Letzte 10 Beobachtungen")
                st.dataframe(history.tail(10), width="stretch")

    with education_tab:
        st.subheader("Education Mode")
        st.caption("Vergleiche Presets, lies praxisnahe Erklärungen und übertrage Szenarien direkt auf die Eingabemaske.")
        st.markdown(
            "- **Spot** verschiebt die Auszahlung, weil sich das Underlying-Level ändert.\n"
            "- **σ** steht für Unsicherheit und dominiert Vega.\n"
            "- **Restlaufzeit & r** treiben Theta- und Rho-Effekte."
        )

        scenario_name = st.selectbox("Szenario-Preset", list(SCENARIO_PRESETS.keys()), key="scenario_select")
        scenario = SCENARIO_PRESETS[scenario_name]
        scenario_spot = max(0.01, spot * (1 + scenario["spot_shift"]))
        scenario_sigma = max(0.05, sigma * scenario["sigma_mult"])
        scenario_strike = max(0.01, strike * (1 + scenario["strike_shift"]))
        scenario_maturity_days = int(np.clip(maturity_days + scenario["maturity_shift_days"], 5, 365))
        scenario_maturity = scenario_maturity_days / 365
        scenario_rate = max(0.0, rate + scenario["rate_shift"])

        scenario_model = BlackScholesModel(
            spot=scenario_spot,
            strike=scenario_strike,
            maturity=scenario_maturity,
            rate=scenario_rate,
            volatility=scenario_sigma,
        )
        scenario_price = scenario_model.call_price() if option_type == "call" else scenario_model.put_price()
        scenario_greeks = as_dict(compute_bsm_greeks(scenario_model, option_type=option_type))

        base_price = float(selected_price) if selected_price is not None else float(bsm_price)
        base_greeks = g_table
        delta_price = scenario_price - base_price

        education_tabs = st.tabs(["Szenario-Analyse", "Greek-Lexikon", "Playbook & Glossar"])

        with education_tabs[0]:
            st.info(scenario["description"])
            metric_cols = st.columns(3)
            metric_cols[0].metric("Szenario-Preis", f"{scenario_price:,.2f}", f"{delta_price:,.2f}")
            metric_cols[1].metric("Szenario-σ", f"{scenario_sigma:.2%}", f"{scenario_sigma - sigma:.2%}")
            metric_cols[2].metric(
                "Laufzeit",
                f"{scenario_maturity_days} Tage",
                f"{scenario_maturity_days - maturity_days} Tage",
            )

            param_df = pd.DataFrame(
                {
                    "Parameter": ["Spot", "Strike", "σ", "Restlaufzeit", "r"],
                    "Baseline": [
                        f"{spot:,.2f}",
                        f"{strike:,.2f}",
                        f"{sigma:.2%}",
                        f"{maturity_days} Tage",
                        f"{rate:.2%}",
                    ],
                    scenario_name: [
                        f"{scenario_spot:,.2f}",
                        f"{scenario_strike:,.2f}",
                        f"{scenario_sigma:.2%}",
                        f"{scenario_maturity_days} Tage",
                        f"{scenario_rate:.2%}",
                    ],
                }
            ).set_index("Parameter")
            st.dataframe(param_df, width="stretch")

            focus_choice = st.radio(
                "Welche Kennzahl möchtest du vertiefen?",
                ("Preis", "Delta", "Gamma", "Vega", "Theta", "Rho"),
                horizontal=True,
                key="education_focus",
            )
            if focus_choice == "Preis":
                focus_value = scenario_price
                focus_delta = delta_price
                baseline_value = base_price
                value_fmt = f"{focus_value:,.2f}"
                delta_fmt = f"{focus_delta:+.2f}"
                baseline_fmt = f"Baseline: {baseline_value:,.2f}"
            else:
                greek_key = focus_choice.lower()
                focus_value = scenario_greeks.get(greek_key, 0.0)
                baseline_value = base_greeks.get(greek_key, 0.0)
                focus_delta = focus_value - baseline_value
                value_fmt = f"{focus_value:.4f}"
                delta_fmt = f"{focus_delta:+.4f}"
                baseline_fmt = f"Baseline: {baseline_value:.4f}"
            st.metric(f"{focus_choice} im Szenario", value_fmt, delta_fmt)
            st.success(f"{FOCUS_GUIDE[focus_choice]} {baseline_fmt}")

            scenario_payoff_prices = np.linspace(max(0.0, scenario_spot * 0.4), scenario_spot * 1.8, 150)
            base_payoff = _build_payoff_curve(option_type, strike, base_price, scenario_payoff_prices)
            scen_payoff = _build_payoff_curve(option_type, scenario_strike, scenario_price, scenario_payoff_prices)
            payoff_fig = go.Figure()
            payoff_fig.add_trace(go.Scatter(x=scenario_payoff_prices, y=base_payoff, name="Baseline Payoff"))
            payoff_fig.add_trace(
                go.Scatter(
                    x=scenario_payoff_prices,
                    y=scen_payoff,
                    name=f"{scenario_name} Payoff",
                    line=dict(dash="dash"),
                )
            )
            payoff_fig.update_layout(
                template="plotly_dark",
                height=420,
                xaxis_title="Spot bei Fälligkeit",
                yaxis_title="Gewinn / Verlust",
            )
            _render_plotly_chart(payoff_fig, key="education_payoff_chart")

            if st.button("Preset auf Sidebar anwenden", key="apply_preset_btn"):
                st.session_state["strike_input"] = float(scenario_strike)
                st.session_state["maturity_days_slider"] = scenario_maturity_days
                st.session_state["manual_sigma_slider"] = float(scenario_sigma)
                st.session_state["spot_fallback_input"] = float(scenario_spot)
                st.session_state["vol_mode_radio"] = "Manuell"
                st.success("Preset wurde übernommen – passe jetzt weitere Parameter an.")

        with education_tabs[1]:
            st.markdown("### Greek-Lexikon")
            greek_cols = st.columns(3)
            for idx, greek_key in enumerate(GREEK_EXPLANATIONS.keys()):
                scen_value = scenario_greeks.get(greek_key, 0.0)
                base_value = base_greeks.get(greek_key, 0.0)
                delta_value = scen_value - base_value
                col = greek_cols[idx % 3]
                col.metric(greek_key.capitalize(), f"{scen_value:.4f}", f"{delta_value:+.4f}")
            st.markdown("---")
            for greek_key, explanation in GREEK_EXPLANATIONS.items():
                base_val = base_greeks.get(greek_key, 0.0)
                scen_val = scenario_greeks.get(greek_key, 0.0)
                st.markdown(
                    f"**{greek_key.capitalize()}** – {explanation}  \n"
                    f"Baseline: {base_val:.4f} · Szenario: {scen_val:.4f} · Δ {scen_val - base_val:+.4f}"
                )

            if st.checkbox("Zeige Mathe-Details", key="show_math_details"):
                st.code(
                    """
Delta = N(d1)
Gamma = n(d1) / (S * sigma * sqrt(T))
Vega  = S * n(d1) * sqrt(T)
Theta ~= -(S * n(d1) * sigma) / (2 * sqrt(T)) - r * K * exp(-r * T) * N(d2)
                    """.strip()
                )

        with education_tabs[2]:
            st.markdown("### Playbook & Glossar")
            st.markdown(
                "1. **Setze das Szenario** – wähle Preset oder passe Parameter manuell an.\n"
                "2. **Validiere Preis & Greeks** – vergleiche Baseline vs. Szenario.\n"
                "3. **Leite Maßnahmen ab** – z. B. Hedge-Adjustments oder Roll-Strategien.\n"
                "4. **Backtest & Kalibriere** – springe zu den anderen Tabs, um das Setup zu verproben."
            )
            st.code(
                "price = BSM(spot, strike, maturity, rate, sigma);\n"
                "delta = ∂price/∂spot;\n"
                "greeks = {delta, gamma, vega, theta, rho}"
            )
            with st.expander("Praxis-Tipps", expanded=True):
                st.markdown(
                    "- **Delta-Drift** → Passe Hedge-Schwellen an, wenn Gamma hoch ist.\n"
                    "- **Vega-Schock** → Nutze Spreads oder Kalender, um Vega-Risiko zu glätten.\n"
                    "- **Theta-Fokus** → Bei positiven Theta (Short Premium) enges Monitoring der Vega-Exponierung."
                )
            st.warning(
                "Education Mode liefert keine Handelsempfehlung, sondern erklärt Wirkzusammenhänge. Kombiniere die Insights mit Backtesting und Kalibrierung."
            )


if __name__ == "__main__":
    main()
