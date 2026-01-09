import math
import os
import argparse
import datetime
from enum import Enum
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import csv

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Unterdrückt Tkinter GUI (verhindert Thread-Lösch-Fehler)
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import TwoSlopeNorm
import yfinance as yf

from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.inspection import permutation_importance

# Volatility Forecaster (optional)
try:
    from volatility_backtest import VolatilityForecaster, create_vola_forecaster
    VOLA_FORECAST_AVAILABLE = True
except ImportError:
    VOLA_FORECAST_AVAILABLE = False

plt.style.use('seaborn-v0_8-darkgrid')

# --------------------------------------------------------------------------------------
# Konfiguration (wird in main() mit CLI-Parametern überschrieben)
# --------------------------------------------------------------------------------------
PROFILE = "aggressive"  # "balanced" | "aggressive"

# Ergebnis Verzeichnisse
BASE_RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'Ergebnisse')
RESULTS_PDF_DIR = os.path.join(BASE_RESULTS_DIR, 'PDF')
RESULTS_CSV_DIR = os.path.join(BASE_RESULTS_DIR, 'CSV')

PROFILES = {
    "balanced": dict(
        SYMBOL="QQQ",                     # wird in main() überschrieben, Default Nasdaq-ETF
        DAILY_PERIOD="10y",
        H1_PERIOD="730d",
        M30_PERIOD="60d",
        START_CAPITAL=100_000.0,
        RISK_PER_TRADE=0.01,
    # Dynamische Risiko / Risikosteuerung
    DYNAMIC_DD_RISK=True,               # Passe Risiko bei Drawdowns an
    DD_RISK_STEPS=[(-10,0.75),(-20,0.5),(-30,0.35),(-40,0.25)],  # (Drawdown %, Multiplikator)
    USE_VOL_TARGET=False,               # Annualisierte Vol-Zielsteuerung aktiv?
    TARGET_ANNUAL_VOL=0.25,             # Ziel-Vol (25%)
    VOL_WINDOW_TRADES=40,               # Rolling Fenster über Trades
    RISK_PER_TRADE_MIN=0.002,           # Untere Klammer für dynamisches Risiko
    RISK_PER_TRADE_MAX=0.02,            # Obere Klammer
    MAX_DRAWDOWN_STOP=-60.0,            # Stop Trading wenn Equity DD < -60%
    OPTIMIZE_ML_THRESHOLD=True,         # Finde bestes Probability-Threshold
    ATR_PERIOD=14,
    ATR_MULT_BUFFER=0.20,
        PRIMARY_ZZ_PCT=0.012, PRIMARY_ZZ_ATR_MULT=0.90, PRIMARY_MIN_IMP_ATR=1.8,
        H1_ZZ_PCT=0.0020,  H1_ZZ_ATR_MULT=0.60,  H1_MIN_IMP_ATR=1.6,
        ENTRY_ZONE_W3=(0.382, 0.786),
        ENTRY_ZONE_W5=(0.236, 0.618),
        ENTRY_ZONE_C=(0.382, 0.786),
        ENTRY_WINDOW_H1=96,
        ENTRY_WINDOW_M30=192,
        MAX_HOLD_H1=192,
        MAX_HOLD_M30=384,
        TP1=1.272, TP2=1.618,
        EMA_FAST=34, EMA_SLOW=144,
        USE_EMA_TREND=False,        # wie gewünscht ausgeschaltet
        REQUIRE_PRICE_ABOVE_EMA_FAST=False,
        USE_DAILY_EMA=False,
        ATR_PCT_MIN=0.05, ATR_PCT_MAX=2.50,
        REQUIRE_CONFIRM=True,
        CONFIRM_BARS_H1=6, CONFIRM_BARS_M30=12,
        CONFIRM_RULES=("break_prev_extreme","ema_fast_cross"),
        ALLOW_TOUCH_IF_NO_CONFIRM=True,
        USE_ML=True, TRAIN_FRAC=0.6, CALIBRATE_PROBS=True,
        SIZE_BY_PROB=True, PROB_SIZE_MIN=0.7, PROB_SIZE_MAX=1.5,
        ML_MIN_PASS_RATE=0.30,
        ML_MIN_PASS_RATE_TEST=0.25,
        USE_W5=False,
        SIZE_SHORT_FACTOR=0.7,
        SHOW_INTERMEDIATE=True,
        LABEL_GAP_DAYS=25,
        PLOT_TRADE_SAMPLE=220,
        EQUITY_LOG_THRESHOLD=5.0,
        WAVE_MIN_PCT=0.08,
        WAVE_MIN_DURATION_DAYS=40,
        WAVE_LABEL_GAP_DAYS=60,
        MAX_PORTFOLIO_DD=-1e9
    ),
    "aggressive": dict(
        SYMBOL="QQQ",
        DAILY_PERIOD="10y",
        H1_PERIOD="730d",
        M30_PERIOD="60d",
        START_CAPITAL=100_000.0,
        RISK_PER_TRADE=0.01,
        ATR_PERIOD=14,
        ATR_MULT_BUFFER=0.20,
        PRIMARY_ZZ_PCT=0.012, PRIMARY_ZZ_ATR_MULT=0.90, PRIMARY_MIN_IMP_ATR=1.8,
        H1_ZZ_PCT=0.0020,  H1_ZZ_ATR_MULT=0.60, H1_MIN_IMP_ATR=1.6,
        ENTRY_ZONE_W3=(0.382, 0.786),
        ENTRY_ZONE_W5=(0.236, 0.618),
        ENTRY_ZONE_C=(0.382, 0.786),
        ENTRY_WINDOW_H1=96,
        ENTRY_WINDOW_M30=192,
        MAX_HOLD_H1=192,
        MAX_HOLD_M30=384,
        TP1=1.272, TP2=1.618,
        EMA_FAST=34, EMA_SLOW=144,
        USE_EMA_TREND=False,
        REQUIRE_PRICE_ABOVE_EMA_FAST=False,
        USE_DAILY_EMA=False,
        ATR_PCT_MIN=0.05, ATR_PCT_MAX=2.50,
        REQUIRE_CONFIRM=True,
        CONFIRM_BARS_H1=6, CONFIRM_BARS_M30=12,
        CONFIRM_RULES=("break_prev_extreme", "ema_fast_cross"),
        ALLOW_TOUCH_IF_NO_CONFIRM=True,
        USE_ML=True, TRAIN_FRAC=0.6, CALIBRATE_PROBS=True,
        SIZE_BY_PROB=True, PROB_SIZE_MIN=0.7, PROB_SIZE_MAX=1.5,
        ML_MIN_PASS_RATE=0.30,
        ML_MIN_PASS_RATE_TEST=0.25,
        USE_W5=False,
        SIZE_SHORT_FACTOR=0.7,
        SHOW_INTERMEDIATE=True,
        LABEL_GAP_DAYS=25,
        PLOT_TRADE_SAMPLE=220,
        EQUITY_LOG_THRESHOLD=5.0,
        WAVE_MIN_PCT=0.08,
        WAVE_MIN_DURATION_DAYS=40,
        WAVE_LABEL_GAP_DAYS=60,
    MAX_PORTFOLIO_DD=-1e9,
    REPORT_THEME="light",
    REPORT_EQUITY_LOG=False,
    REPORT_TABLE_FONT_SIZE=11,
    REPORT_SHOW_ML=True,
    REPORT_MAX_KPI_COLUMNS=2,
    REPORT_INCLUDE_DISTRIBUTIONS=True,
    REPORT_INCLUDE_SETUP_STATS=True,
    REPORT_INCLUDE_STRUCTURE=True,
    ),
    "adaptive": dict(
        SYMBOL="QQQ",
        DAILY_PERIOD="10y",
        H1_PERIOD="730d",
        M30_PERIOD="60d",
        START_CAPITAL=100_000.0,
        RISK_PER_TRADE=0.008,              # 0.8%
        DYNAMIC_DD_RISK=True,
        DD_RISK_STEPS=[(-5,0.85),(-10,0.65),(-15,0.50),(-20,0.38),(-25,0.30)],
        USE_VOL_TARGET=True,
        TARGET_ANNUAL_VOL=0.25,
        VOL_WINDOW_TRADES=35,
        RISK_PER_TRADE_MIN=0.002,
        RISK_PER_TRADE_MAX=0.012,
        MAX_DRAWDOWN_STOP=-40.0,
        OPTIMIZE_ML_THRESHOLD=True,
        ATR_PERIOD=14,
        ATR_MULT_BUFFER=0.20,
        TP1_LONG=1.272, TP2_LONG=1.618,
        TP1_SHORT=1.272, TP2_SHORT=1.618,
        ATR_MULT_BUFFER_LONG=0.20,
        ATR_MULT_BUFFER_SHORT=0.20,
        ENTRY_ZONE_W3_LONG=(0.382, 0.786),
        ENTRY_ZONE_W3_SHORT=(0.50, 0.786),
        ENTRY_ZONE_W5_LONG=(0.236, 0.618),
        ENTRY_ZONE_W5_SHORT=(0.382, 0.618),
        ENTRY_ZONE_C_LONG=(0.382, 0.786),
        ENTRY_ZONE_C_SHORT=(0.50, 0.786),
        PRIMARY_ZZ_PCT=0.012, PRIMARY_ZZ_ATR_MULT=0.90, PRIMARY_MIN_IMP_ATR=1.9,
        H1_ZZ_PCT=0.0022,  H1_ZZ_ATR_MULT=0.60, H1_MIN_IMP_ATR=1.7,
        ENTRY_ZONE_W3=(0.382, 0.786),
        ENTRY_ZONE_W5=(0.236, 0.618),
        ENTRY_ZONE_C=(0.382, 0.786),
        ENTRY_WINDOW_H1=96,
        ENTRY_WINDOW_M30=192,
        MAX_HOLD_H1=192,
        MAX_HOLD_M30=384,
        TP1=1.272, TP2=1.618,
        EMA_FAST=34, EMA_SLOW=144,
        USE_EMA_TREND=True,
        REQUIRE_PRICE_ABOVE_EMA_FAST=False,
        USE_DAILY_EMA=False,   # deaktiviert (geringer Zusatznutzen)
        ATR_PCT_MIN=0.05, ATR_PCT_MAX=2.30,
        ADX_TREND_THRESHOLD=25,
        USE_ADX=False,         # ADX aus (keine Filterwirkung festgestellt)
        REQUIRE_CONFIRM=True,
        CONFIRM_BARS_H1=6, CONFIRM_BARS_M30=12,
        CONFIRM_RULES=("break_prev_extreme", "ema_fast_cross"),
        ALLOW_TOUCH_IF_NO_CONFIRM=True,
        USE_ML=True, TRAIN_FRAC=0.6, CALIBRATE_PROBS=True,
        SIZE_BY_PROB=True, PROB_SIZE_MIN=0.8, PROB_SIZE_MAX=1.35,
        ML_MIN_PASS_RATE=0.30,
        ML_MIN_PASS_RATE_TEST=0.25,
        USE_W5=False,
        SIZE_SHORT_FACTOR=0.6,
        COMMISSION_PER_TRADE=0.0,
        SLIPPAGE_PCT=0.0,
        SHOW_INTERMEDIATE=True,
        LABEL_GAP_DAYS=25,
        PLOT_TRADE_SAMPLE=220,
        EQUITY_LOG_THRESHOLD=5.0,
        WAVE_MIN_PCT=0.08,
        WAVE_MIN_DURATION_DAYS=40,
        WAVE_LABEL_GAP_DAYS=60,
        MAX_PORTFOLIO_DD=-1e9
    ),
    "target77": dict(
        SYMBOL="QQQ",
        DAILY_PERIOD="10y",
        H1_PERIOD="730d",
        M30_PERIOD="60d",
        START_CAPITAL=100_000.0,
        RISK_PER_TRADE=0.012,
        OPTIMIZE_ML_THRESHOLD=True,
        USE_ML=True, TRAIN_FRAC=0.6, CALIBRATE_PROBS=True,
        SIZE_BY_PROB=True, PROB_SIZE_MIN=0.8, PROB_SIZE_MAX=1.40,
        ML_MIN_PASS_RATE=0.30, ML_MIN_PASS_RATE_TEST=0.25,
        USE_EMA_TREND=False, USE_DAILY_EMA=True, USE_ADX=False,
        REQUIRE_CONFIRM=False,
        TP1=1.272, TP2=1.618,
        ATR_PERIOD=14, ATR_MULT_BUFFER=0.20,
        ENTRY_ZONE_W3=(0.382, 0.786), ENTRY_ZONE_W5=(0.236,0.618), ENTRY_ZONE_C=(0.382,0.786),
        ENTRY_WINDOW_H1=96, ENTRY_WINDOW_M30=192,
        MAX_HOLD_H1=192, MAX_HOLD_M30=384,
        EMA_FAST=34, EMA_SLOW=144,
        ATR_PCT_MIN=0.05, ATR_PCT_MAX=2.40,
        SIZE_SHORT_FACTOR=0.7,
        SHOW_INTERMEDIATE=True, LABEL_GAP_DAYS=25, PLOT_TRADE_SAMPLE=220,
        EQUITY_LOG_THRESHOLD=5.0, WAVE_MIN_PCT=0.08, WAVE_MIN_DURATION_DAYS=40, WAVE_LABEL_GAP_DAYS=60,
        TARGET_WINRATE_MIN=0.70, TARGET_WINRATE_MAX=0.80, TARGET_CAGR_MIN=120.0, TARGET_MAX_DD=-5.0,
        AUTO_RISK_SCALE=True,
    THR_OFFSET_W3=-0.02, THR_OFFSET_C=-0.04, THR_OFFSET_W5=-0.04, THR_OFFSET_OTHER=-0.04,
    APPLY_THRESHOLD_SHIFT=-0.12,
        MAX_PORTFOLIO_DD=-1e9
    ),
}

# Wird in main() gesetzt (damit wir Symbol/Profil einfach überschreiben können)
CFG: Dict = {}
RISK_FREE_RATE = 2.0  # % p.a.

# --------------------------------------------------------------------------------------
# CLI / Main (wurde durch Refactor offenbar entfernt) – neu hinzugefügt
# --------------------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Elliott Wave Backtest + ML + Deep Eval",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "Beispiele:\n"
            "  Standard (aggressive, QQQ)\n"
            "    python Ur-EW-Code.py\n\n"
            "  Mehrere Symbole + ML aus + manuelle Schwellen\n"
            "    python Ur-EW-Code.py -s QQQ,GC=F,EURUSD=X --no-ml --thr-shift -0.1\n\n"
            "  CSV Daten statt yfinance + Gebühren/Slippage\n"
            "    python Ur-EW-Code.py -s QQQ --csv --fee 0.001 --slippage 0.0005\n\n"
            "  Gewinne quartalsweise realisieren + monatlicher Payout\n"
            "    python Ur-EW-Code.py --realisieren --monthly-payout 5000\n\n"
            "  Deep Eval Ziel-Winrate 0.77 und beste Offsets anwenden\n"
            "    python Ur-EW-Code.py --deep-eval --deep-eval-target-wr 0.77 --deep-eval-apply-best --deep-eval-rerun-best\n"
        )
    )
    # Basis
    p.add_argument("--symbol","-s", default=os.getenv("EW_SYMBOL","QQQ"), help="Ticker oder mehrere mit Komma (z.B. QQQ,GC=F,EURUSD=X)")
    p.add_argument("--profile","-p", choices=list(PROFILES.keys()), default=os.getenv("EW_PROFILE", PROFILE), help="Profil (balanced/aggressive/adaptive)")
    p.add_argument("--no-ml", action="store_true", help="ML Filter deaktivieren")
    p.add_argument("--show-intermediate", action="store_true", help="Zwischenstruktur im Report zeigen")
    p.add_argument("--verbose", action="store_true", help="Mehr Debug-Ausgaben")
    # Daten / Kosten / Runtime
    p.add_argument("--csv","-c", action="store_true", help="CSV Daten nutzen (siehe load_data Dateinamen)")
    p.add_argument("--realisieren","-r", action="store_true", help="Quartalsweise Gewinne realisieren (Reset auf Startkapital + Equity Überschuss)")
    p.add_argument("--fee", type=float, default=0.0, help="Gebühr pro Trade (abs oder %% wenn <1) z.B. 0.001=0.1%%")
    p.add_argument("--slippage", type=float, default=0.0, help="Slippage pro Trade (abs oder %% wenn <1)")
    p.add_argument("--slow", type=float, default=0.0, help="Verzögerung Orderausführung in Sekunden (max 30)")
    p.add_argument("--monthly-payout", type=float, default=0.0, help="Monatlicher Auszahlungsbetrag vom Konto")
    # Filter Toggles
    p.add_argument("--no-ema-trend", action="store_true", help="EMA Trendfilter abschalten")
    p.add_argument("--no-daily-ema", action="store_true", help="Daily EMA Filter abschalten")
    p.add_argument("--no-adx", action="store_true", help="ADX Filter abschalten")
    p.add_argument("--no-confirm", action="store_true", help="Entry Bestätigungslogik abschalten")
    # FTMO/Prop Optionen
    p.add_argument("--ftmo", action="store_true", help="FTMO Auszahlung: Nur 80%% der Gewinne werden dem Konto gutgeschrieben (20%% an FTMO)")
    p.add_argument("--ftmo-challenge", type=float, default=None, help="FTMO/Prop Challenge Account-Größe (z.B. 100000 für 100k)")
    p.add_argument("--ftmo_challenge100k", action="store_true", help="Alias für --ftmo-challenge 100000")
    p.add_argument("--ftmo_challenge1mio", action="store_true", help="Alias für --ftmo-challenge 1000000")
    # Firm Presets & Rule Overrides
    p.add_argument("--firm", choices=["ftmo", "ftmo-aggressive", "apex", "fundednext"], default=None,
                   help="Voreinstellungen für Prop-Firmenregeln: ftmo (5/10), ftmo-aggressive (10/20), apex (Trailing DD), fundednext (kein Zeitlimit)")
    p.add_argument("--rules-daily-loss", type=float, default=None, help="Täglicher Max-Verlust in %% (z.B. 5 für 5%%)")
    p.add_argument("--rules-max-loss", type=float, default=None, help="Maximaler Verlust in %% (z.B. 10 für 10%%)")
    p.add_argument("--rules-profit-target", type=float, default=None, help="Profitziel in %% (z.B. 10 für 10%%)")
    p.add_argument("--rules-min-days", type=int, default=None, help="Minimale Anzahl Trading-Tage")
    p.add_argument("--rules-time-limit", type=int, default=None, help="Zeitlimit in Tagen (0/negativ = kein Limit)")
    p.add_argument("--rules-trailing", action="store_true", help="Trailing Max-Drawdown statt statischem Max-Drawdown anwenden")
    # Counterfactual / Varianten
    p.add_argument("--counterfactuals", action="store_true", help="Einfache Was-wäre-wenn Varianten")
    p.add_argument("--deep-counterfactuals", action="store_true", help="Erweiterte Varianten (Risk/Filter Sweeps)")
    p.add_argument("--full-grid-cf", action="store_true", help="Kompletter Varianten-Grid (sehr langsam)")
    # Deep Evaluation / Threshold Suche
    p.add_argument("--deep-eval", action="store_true", help="Tiefere Threshold/Offset Raster Suche aktivieren")
    p.add_argument("--deep-eval-wide", action="store_true", help="Breiteres Raster für Schwellen")
    p.add_argument("--deep-eval-target-wr", type=float, default=None, help="Ziel-Winrate (z.B. 0.77)")
    p.add_argument("--deep-eval-apply-best", action="store_true", help="Beste Offsets in CFG anwenden")
    p.add_argument("--deep-eval-rerun-best", action="store_true", help="Nach Anwendung erneut mit gespeicherten Wahrscheinlichkeiten rerunnen")
    p.add_argument("--deep-eval-make-final", action="store_true", help="Finalen Report mit angewendeten Offsets erzeugen")
    p.add_argument("--deep-eval-save-csv", action="store_true", help="Ergebnisraster als CSV speichern")
    p.add_argument("--deep-eval-min-trades", type=int, default=300, help="Minimale Trades für Gültigkeit")
    p.add_argument("--deep-eval-max-dd", type=float, default=-5.0, help="Max Drawdown Constraint (z.B. -5.0)")
    p.add_argument("--secondary-threshold", type=float, default=None, help="Sekundärer Schwellenwert (optional)")
    # Neutral Mode – removes overfitting-prone optimizations
    p.add_argument("--neutral", action="store_true", help="Neutraler Backtest: fester Threshold, kein Risk-Scaling, keine Offsets, kein SIZE_BY_PROB")
    # Momentum Filter (like live system)
    p.add_argument("--momentum", action="store_true", help="Momentum-Filter aktivieren: Trade nur wenn Momentum mit Richtung übereinstimmt")
    p.add_argument("--momentum-period", type=int, default=14, help="Momentum Lookback Periode (default: 14 Bars)")
    p.add_argument("--momentum-threshold", type=float, default=0.0, help="Minimum Momentum Threshold (default: 0.0 = nur Richtung)")
    # Momentum Exit (early exit when momentum weakens on higher TF)
    p.add_argument("--momentum-exit", type=int, default=0, help="Momentum Exit: Schließe nach N abnehmenden Momentum-Bars auf H1 (0=aus, 2-3 empfohlen)")
    # ADX / Regime Filter
    p.add_argument("--use-adx", action="store_true", help="ADX-Filter aktivieren (filtert Trades bei ADX < Threshold)")
    p.add_argument("--adx-threshold", type=int, default=25, help="ADX Threshold für Trendstärke (default: 25)")
    # Volatility Regime Analysis
    p.add_argument("--analyze-regimes", action="store_true", help="Performance nach Volatilitäts-Regimes analysieren")
    p.add_argument("--vola-min", type=int, default=0, help="Min ATR-Percentile für Trades (0-100, z.B. 25 = keine Low-Vola)")
    p.add_argument("--vola-max", type=int, default=100, help="Max ATR-Percentile für Trades (0-100, z.B. 75 = keine High-Vola)")
    # Volatility Forecast Sizing (GARCH+HAR based)
    p.add_argument("--vola-forecast", action="store_true", help="Vola-Prognose für Position Sizing aktivieren")
    p.add_argument("--vola-forecast-window", type=int, default=252, help="Trainings-Fenster für Vola-Modell (default: 252 Tage)")
    # Minimum Profit Factor (wie im Live System)
    p.add_argument("--min-pf", type=float, default=0.0, help="Min Profit Factor (TP/SL Distanz), z.B. 1.2 = nur Trades mit RR >= 1.2:1 (wie Live System)")
    # Local Data (statt yfinance)
    p.add_argument("--local-data", type=str, default=None, help="Pfad zu lokalen Daten (Basis-Name, z.B. 'daten/xauusd' lädt _daily.csv, _h1.csv, _m30.csv)")
    # Manuelle Schwellen / Offsets
    p.add_argument("--thr-shift", type=float, default=None, help="Globaler Threshold Shift (additiv)")
    p.add_argument("--thr-off-w3", type=float, default=None, help="Offset W3")
    p.add_argument("--thr-off-c", type=float, default=None, help="Offset C")
    p.add_argument("--thr-off-w5", type=float, default=None, help="Offset W5")
    p.add_argument("--thr-off-other", type=float, default=None, help="Offset andere")
    p.add_argument(
        "--portfolio-file","-P",
        type=str,
        default=None,
        help=(
            "Portfolio CSV (Format: symbol,<gewicht>). Zweite Spalte beliebig. Dezimal oder Prozent (25 / 25%%). "
            "Auto-Delimiter ',' / ';'. Wird Dateiname allein angegeben und nicht gefunden, wird in ./Portfolios/ gesucht. "
            "Bsp: -P Portfolios/dax.csv  oder  -P dax.csv"
        )
    )
    # Portfolio Report Optionen
    p.add_argument("--portfolio-corr-values", action="store_true", help="Korrelations-Heatmap mit Zahlen beschriften")
    p.add_argument("--portfolio-trim-pnl", type=float, default=None, help="Winsorize Trade PnL auf gegebenes Quantil (z.B. 0.99)")
    p.add_argument("--portfolio-max-contrib", type=int, default=25, help="Max Anzahl Einzelbeiträge in Tabelle/Barchart (Rest aggregiert)")
    return p.parse_args()

def _apply_manual_offsets(cfg, args):
    if args.thr_shift is not None: cfg['APPLY_THRESHOLD_SHIFT']=args.thr_shift
    if args.thr_off_w3 is not None: cfg['THR_OFFSET_W3']=args.thr_off_w3
    if args.thr_off_c is not None: cfg['THR_OFFSET_C']=args.thr_off_c
    if args.thr_off_w5 is not None: cfg['THR_OFFSET_W5']=args.thr_off_w5
    if args.thr_off_other is not None: cfg['THR_OFFSET_OTHER']=args.thr_off_other

def sanitize_symbol(sym: str) -> str:
    """Ersetze Dateisystem-kritische Zeichen in Symbolen für Dateinamen.
    Beispiel: '^GSPC' -> 'GSPC', 'EURUSD=X' -> 'EURUSD_X'
    """
    if not sym:
        return 'UNKNOWN'
    # Entferne führendes '^'
    s = sym.strip()
    if s.startswith('^'):
        s = s[1:]
    repl = {':':'-', '/':'-', '\\':'-', '*':'x', '?':'', '"':'', '<':'', '>':'', '|':'-', '=':'_', ' ':'_', ';':'-'}
    out = []
    for ch in s:
        out.append(repl.get(ch, ch))
    cleaned = ''.join(out)
    # Doppelte Unterstriche reduzieren
    while '__' in cleaned:
        cleaned = cleaned.replace('__','_')
    return cleaned.strip('_') or 'SYMBOL'

def load_portfolio_file(path: str):
    """Liest Portfolio CSV mit Spalten: symbol, weight/allocation.
    Unterstützt:
      - Dezimalgewichte (0.25 / 0.4 usw.)
      - Prozent Angaben (25 / 40 oder 25% / 40%) – Werte >1 oder mit '%'
        werden als Prozent interpretiert und durch 100 geteilt.
      - Trennzeichen ';' werden entfernt (falls aus Excel exportiert)
    Kommentare (# ...) und Leerzeilen werden ignoriert.
    Rückgabe: DataFrame mit Spalten symbol, weight (normalisiert auf 1.0)
    """
    import pandas as _pd, re as _re
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(f"Portfolio Datei nicht gefunden: {path}")
    with open(path,'r',encoding='utf-8') as fh:
        raw=fh.read()
    lines_raw = raw.splitlines()
    # Entferne BOM / unsichtbare Zeichen aus Header
    if lines_raw:
        lines_raw[0] = lines_raw[0].replace('\ufeff','').strip()
    # Bestimme ob Semikolon benutzt werden soll: wenn irgendeine Datenzeile ein Semikolon enthält UND die erste "richtige" Datenzeile nur genau 1 Semikolon (Symbol;Zahl) hat.
    data_lines = [l for l in lines_raw[1:] if l.strip() and not l.strip().startswith('#')]
    first_data = data_lines[0] if data_lines else ''
    # Heuristik: wenn viele Linien genau ein Semikolon enthalten und Komma im Zahlenteil (Dezimal-Komma), dann Semikolon Trenner
    semicolon_simple_rows = sum(1 for l in data_lines if l.count(';')==1 and _re.search(r';\s*\d', l))
    use_semicolon = semicolon_simple_rows >= max(1, int(0.5*len(data_lines)))
    cleaned = []
    for ln in lines_raw:
        s = ln.strip()
        if not s or s == ';' or s.startswith('#'):
            continue
        # Header Mischform: "symbol,weight;" -> vereinheitlichen zu "symbol;weight"
        if s.lower().startswith('symbol') and ('weight' in s.lower() or 'alloc' in s.lower()):
            # ersetze jegliche Komma oder Semikolon Sequenzen zwischen symbol und weight durch EIN Semikolon
            s = _re.sub(r'symbol\s*[,;]\s*(weight|allocation|alloc|percent)[,;]*', r'symbol;\1', s, flags=_re.IGNORECASE)
            s = s.rstrip(';')
        # Bereinigung gemischter Delimiter in Datenzeilen
        if use_semicolon:
            # Varianten: "SAP.DE; 10,03" oder "SAP.DE , 10,03" -> vereinheitliche
            s = _re.sub(r',\s*;',';', s)
            s = _re.sub(r';\s*,',';', s)
            s = _re.sub(r'\s+,\s+', ';', s)  # SYMBOL , 10,03 -> SYMBOL;10,03
            # Falls Komma direkt nach Symbol (falsch) aber Semikolon fehlt und genau ein Komma + Dezimal-Komma existiert -> ersetze erstes Komma durch Semikolon
            if s.count(';')==0 and s.count(',')==1 and _re.search(r'^[A-Za-z0-9_.=^]+,\s*\d', s):
                s = s.replace(',', ';', 1)
        cleaned.append(s.rstrip(';'))
    if os.getenv('EW_DEBUG_PORTFOLIO'):
        try:
            print(f"[DEBUG-PORTFOLIO] Header Raw: {lines_raw[0] if lines_raw else '(none)'}")
            print(f"[DEBUG-PORTFOLIO] use_semicolon={use_semicolon} cleaned_lines={len(cleaned)}")
            for i,l in enumerate(cleaned[:6]):
                print(f"[DEBUG-PORTFOLIO] L{i}: {l}")
        except Exception:
            pass
    # EARLY MANUAL PARSE FALLBACK: Wenn erste Zeile wie 'symbol,weight;' und folgende viele 'XYZ.DE; 1,23' -> direkt manuell parsen
    header_line = cleaned[0].lower() if cleaned else ''
    pattern_manual = header_line.startswith('symbol') and ('weight' in header_line)
    data_like = sum(1 for l in cleaned[1:15] if ';' in l and _re.search(r';\s*\d+[,.]\d+', l))
    if pattern_manual and data_like>=3:
        rows=[]
        for l in cleaned[1:]:
            if ';' not in l: continue
            sym_part, val_part = l.split(';',1)
            sym_part=sym_part.strip()
            val_part=val_part.strip()
            if not sym_part or not _re.search(r'[A-Za-z]', sym_part):
                continue
            rows.append((sym_part, val_part))
        if rows:
            df=_pd.DataFrame(rows, columns=['symbol','weight'])
            # Simuliere normalen Verlauf ab hier
            df.columns=['symbol','weight']
            weight_col='weight'
            # Weiter unten normale Nachbearbeitung -> springe zum Ende des Parsing-Abschnitts
            # (wir setzen use_manual Flag und gehen später nicht erneut durch read_csv)
            # -> Weiterverarbeitung passiert nach Spaltenerkennung
            # Konvertierung erfolgt später
        else:
            df=None
    from io import StringIO
    csv_str = '\n'.join(cleaned)
    # Versuche zuerst mit gewünschtem Separator, bei Fehler fallback autodetect
    df=None
    # Direkte manuelle Parse-Variante falls erste Zeile genau 2 Tokens enthält 'symbol;weight'
    header_tokens = cleaned[0].lower().split(';') if cleaned else []
    if len(header_tokens)==2 and header_tokens[0]=='symbol' and 'weight' in header_tokens[1]:
        rows=[]
        for line in cleaned[1:]:
            if ';' not in line:
                continue
            sym,val=line.split(';',1)
            sym=sym.strip(); val=val.strip()
            if not sym or not _re.search(r'[A-Za-z]', sym):
                continue
            rows.append((sym,val))
        if rows:
            df=_pd.DataFrame(rows, columns=['symbol','weight'])
    if df is None:
        try:
            if use_semicolon:
                df = _pd.read_csv(StringIO(csv_str), sep=';')
            else:
                df = _pd.read_csv(StringIO(csv_str))
        except Exception:
            csv_str2 = '\n'.join(l.replace(' ;', ';').replace('; ', ';') for l in cleaned)
            try:
                df = _pd.read_csv(StringIO(csv_str2), sep=';' if use_semicolon else ',')
            except Exception:
                rows = []
                for l in cleaned:
                    if ';' in l:
                        left, right = l.split(';',1)
                    elif ',' in l:
                        left, right = l.split(',',1)
                    else:
                        continue
                    left = left.strip(); right = right.strip()
                    if not _re.search(r'[A-Za-z]', left):
                        continue
                    rows.append((left, right))
                if not rows:
                    raise ValueError("Portfolio CSV Parsing fehlgeschlagen (kein Fallback möglich)")
                df = _pd.DataFrame(rows, columns=['symbol','weight'])
    df.columns=[c.lower().strip() for c in df.columns]
    # Spezialfall: Header kam als eine kombinierte Spalte 'symbol,weight' rein
    if len(df.columns)==1 and ('symbol' in df.columns[0]) and (',' in df.columns[0]):
        # Versuche manuell zu splitten
        col_parts=[p.strip() for p in df.columns[0].split(',') if p.strip()]
        if len(col_parts)==2 and col_parts[0]=='symbol':
            # split rows on first semicolon or comma
            rows=[]
            for _,row in df.iterrows():
                raw_line=''.join(str(x) for x in row.values)
                raw_line=raw_line.strip().rstrip(';')
                if not raw_line: continue
                if ';' in raw_line:
                    left,right=raw_line.split(';',1)
                elif ',' in raw_line:
                    left,right=raw_line.split(',',1)
                else:
                    continue
                left=left.strip(); right=right.strip()
                if left.lower()=='symbol' and right.lower() in {'weight','alloc','allocation','percent','percentage','w'}:
                    continue
                rows.append((left,right))
            if rows:
                df=_pd.DataFrame(rows, columns=['symbol','weight'])
                df.columns=['symbol','weight']
    # Weiterer Spezialfall: zwei Spalten, aber erste heißt 'symbol,weight' und zweite irgendwas -> rename
    if len(df.columns)==2 and df.columns[0].startswith('symbol') and 'weight' in df.columns[0] and 'symbol' not in df.columns:
        df = df.rename(columns={df.columns[0]:'symbol', df.columns[1]:'weight'})
    # Diagnose-Ausgabe (einmalig)
    try:
        if 'EW_DEBUG_PORTFOLIO' in os.environ:
            print(f"[DEBUG-PORTFOLIO] Columns nach Einlesen: {df.columns.tolist()} (rows={len(df)})")
    except Exception:
        pass
    weight_col=None
    for cand in ("weight","allocation","alloc","w","percent","percentage"):
        if cand in df.columns:
            weight_col=cand; break
    # Fallback: falls nur 2 Spalten und zweite nicht erkannt -> nutze zweite Spalte
    if weight_col is None and 'symbol' in df.columns and len(df.columns)==2:
        other=[c for c in df.columns if c!='symbol'][0]
        weight_col=other
    if weight_col is None:
        # Letzter Versuch: genau 2 Spalten? Dann zweite nehmen.
        if len(df.columns)==2:
            weight_col = df.columns[1]
        else:
            raise ValueError("Portfolio CSV benötigt Spalte 'weight' / 'allocation' / 'percent' oder beliebige zweite Spalte mit Zahlen")
    if 'symbol' not in df.columns:
        raise ValueError("Portfolio CSV benötigt Spalte 'symbol'")
    df=df[['symbol',weight_col]].copy().dropna()
    # Normalisieren: entferne % Zeichen und Kommas
    def _parse_w(x):
        if isinstance(x,str):
            xs=x.strip().replace(',','.')
            pct=False
            if xs.endswith('%'):
                pct=True; xs=xs[:-1]
            try:
                val=float(xs)
            except ValueError:
                return None
            # Werte >1 oder mit erkanntem pct Flag -> Prozent
            if pct or val>1.0:
                val=val/100.0
            return val
        elif isinstance(x,(int,float)):
            val=float(x)
            if val>1.0: # wahrscheinlich Prozent
                val=val/100.0
            return val
        return None
    # Falls alle Werte >1 (z.B. alle Prozent ohne %) und max <=100 -> später skalieren
    raw_vals = df[weight_col].astype(str).str.strip().tolist()
    df[weight_col]=df[weight_col].apply(_parse_w)
    if df[weight_col].notna().all():
        max_v = df[weight_col].max()
        min_v = df[weight_col].min()
        # Heuristik: Wenn Werte wie 10.03 (nach Komma->Punkt) interpretiert wurden und max <=100 -> Prozent (teilen durch 100)
        if max_v > 1.0 and max_v <= 100.0:
            df[weight_col] = df[weight_col] / 100.0
        # Falls versehentlich schon geteilt und jetzt alles sehr klein (<0.02) aber Original enthielt viele Kommas => kein zweites Mal teilen
        # (Keine Aktion nötig, nur Kommentar für Verständnis)
    df=df[df[weight_col].notna() & (df[weight_col]>0)]
    if df.empty:
        raise ValueError("Keine gültigen Gewichte nach Parsing")
    total=df[weight_col].sum()
    if total<=0:
        raise ValueError("Summe der Gewichte ist 0")
    df['weight']=df[weight_col]/total
    out = df[['symbol','weight']].copy()
    # Logging
    try:
        symbols_preview = ', '.join(out['symbol'].head(8).tolist())
        print(f"[Portfolio-Datei] {len(out)} Symbole geladen aus {os.path.basename(path)}: {symbols_preview}{' ...' if len(out)>8 else ''}")
    except Exception:
        pass
    return out

def build_equity_series(trades, start_capital: float):
    import pandas as _pd
    if not trades:
        return _pd.Series([start_capital], index=[_pd.Timestamp.utcnow()])
    df = _pd.DataFrame([t.__dict__ for t in trades])
    df['time_out'] = _pd.to_datetime(df['time_out'])
    df = df.sort_values('time_out')
    df['cum_pnl'] = df['pnl'].cumsum()
    df['equity'] = start_capital + df['cum_pnl']
    eq = df.set_index('time_out')['equity'].resample('1D').last().ffill()
    return eq

def ftmo_evaluate_attempts(daily_equity: 'pd.Series',
                           challenge_capital: float,
                           daily_loss_pct: float = 0.05,
                           max_loss_pct: float = 0.10,
                           profit_target_pct: float = 0.10,
                           min_trading_days: int = 4,
                           time_limit_days: int = 30,
                           trade_dates: 'Optional[set]' = None,
                           trailing_max_loss: bool = False):
    """Simulate sequential FTMO attempts over a daily equity curve.

    Assumptions:
    - Uses daily close-to-close returns to approximate PnL (intraday drawdowns not modeled).
    - Daily loss and max loss thresholds are based on challenge_capital.
    - A new attempt starts the next day after success/failure, with equity reset to challenge_capital.
    - min_trading_days counts distinct days with at least one trade (if trade_dates provided),
      otherwise counts elapsed days.
    Returns: (attempts:int, passed:int)
    """
    import pandas as _pd
    if daily_equity is None or len(daily_equity) < 2:
        return 0, 0
    daily_equity = daily_equity.dropna()
    if len(daily_equity) < 2:
        return 0, 0
    returns = daily_equity.pct_change().fillna(0.0)
    dates = list(returns.index)
    attempts = 0
    passed = 0
    i = 0
    while i < len(dates):
        # Start a new attempt at day i
        attempt_equity = float(challenge_capital)
        attempt_start_equity = attempt_equity
        high_water = attempt_equity  # for trailing DD
        start_date = _pd.Timestamp(dates[i]).normalize()
        days_used = 0
        traded_days = 0
        # iterate days up to time limit or series end
        # interpret time_limit_days<=0 as no limit
        _limit = int(time_limit_days) if (time_limit_days is not None and time_limit_days > 0) else 10**9
        while i < len(dates) and days_used < _limit:
            d = _pd.Timestamp(dates[i]).normalize()
            day_start = attempt_equity
            r = float(returns.iloc[i])
            day_end = day_start * (1.0 + r)
            # daily loss rule
            if daily_loss_pct and daily_loss_pct > 0:
                if (day_end - day_start) <= -daily_loss_pct * challenge_capital:
                    attempts += 1
                    i += 1
                    break
            attempt_equity = day_end
            # update high-water mark for trailing DD
            if trailing_max_loss:
                if attempt_equity > high_water:
                    high_water = attempt_equity
            # track trading day
            if trade_dates is not None:
                if d in trade_dates:
                    traded_days += 1
            else:
                traded_days += 1
            days_used += 1
            # max loss rule
            if max_loss_pct and max_loss_pct > 0:
                if trailing_max_loss:
                    # Fail if equity falls below high-water minus allowed trailing dd
                    if attempt_equity <= (high_water - max_loss_pct * challenge_capital):
                        attempts += 1
                        i += 1
                        break
                else:
                    if (attempt_equity - attempt_start_equity) <= -max_loss_pct * challenge_capital:
                        attempts += 1
                        i += 1
                        break
            # profit target check
            if (attempt_equity - attempt_start_equity) >= profit_target_pct * challenge_capital:
                attempts += 1
                if traded_days >= min_trading_days:
                    passed += 1
                i += 1
                break
            # advance to next day
            i += 1
        else:
            # time limit reached or series ended
            if days_used > 0:
                attempts += 1
                if (attempt_equity - attempt_start_equity) >= profit_target_pct * challenge_capital and traded_days >= min_trading_days:
                    passed += 1
            # If no days used, advance to avoid infinite loop
            if days_used == 0:
                i += 1
    return attempts, passed

def ftmo_tune_risk(daily_df: 'pd.DataFrame', h1_df: 'pd.DataFrame', m30_df: 'pd.DataFrame', base_cfg: dict,
                   allowed_max_loss_pct: float = 0.10, target_utilization: float = 0.95,
                   iters: int = 5) -> float:
    """Binary-search a multiplier for RISK_PER_TRADE so that backtest max_dd approaches
    -allowed_max_loss_pct * 100 (not exceeding it), targeting a utilization fraction.
    Returns the tuned absolute RISK_PER_TRADE value, or the original if tuning fails.
    """
    import copy as _copy
    try:
        base_risk = float(base_cfg.get('RISK_PER_TRADE', 0.01))
        if base_risk <= 0:
            return base_risk
        low, high = 0.25, 5.0
        target_dd = -allowed_max_loss_pct * 100.0 * float(target_utilization)
        tuned = base_risk
        for _ in range(max(1, iters)):
            mid = (low + high) / 2.0
            test_cfg = _copy.deepcopy(base_cfg)
            test_cfg['RISK_PER_TRADE'] = base_risk * mid
            # fix dynamic risk to keep consistent scaling during tuning
            dyn_backup = test_cfg.get('DYNAMIC_DD_RISK', False)
            test_cfg['DYNAMIC_DD_RISK'] = False
            # run backtest
            global CFG
            CFG = test_cfg
            bt = Backtester(daily_df, h1_df, m30_df)
            m = bt.run()
            if not m or 'max_dd' not in m:
                break
            dd = float(m['max_dd'])  # negative number
            # If dd is less negative than target (|dd| < |target|), we can scale up risk
            if dd > target_dd:
                tuned = test_cfg['RISK_PER_TRADE']
                low = mid
            else:
                # dd too deep (more negative) than target, scale down
                high = mid
            # restore dyn flag in base_cfg for next iter (not strictly needed)
            test_cfg['DYNAMIC_DD_RISK'] = dyn_backup
        return max(1e-5, tuned)
    except Exception:
        return float(base_cfg.get('RISK_PER_TRADE', 0.01))

## main() call will appear at real file end (removed here)

# --------------------------------------------------------------------------------------
# Data & Indicators
# --------------------------------------------------------------------------------------
def _normalize_yf_df(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    # Flatten MultiIndex
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join([str(x) for x in tup if x and x!='-']).strip() for tup in df.columns]
    # Normalize names
    df.columns = [str(c).strip().lower().replace('\ufeff','') for c in df.columns]
    # Ersetze Leerzeichen durch Unterstrich für robustere Erkennung
    df.columns = [c.replace(' ', '_') for c in df.columns]
    sym_lower = str(symbol).lower()
    # Dynamisches Mapping: open_xxx, close_xxx etc. auf Basisnamen reduzieren
    mapped_cols = {}
    for c in df.columns:
        base = c
        if '_' in c:
            parts = c.split('_')
            # Fälle wie close_qqq, open_qqq
            if parts[0] in {'open','high','low','close','volume'}:
                base = parts[0]
            # adj_close_qqq oder adjclose_qqq
            elif (parts[0] in {'adj','adjclose','adjclose'} or (parts[0]=='adj' and parts[1]=='close')):
                base = 'adj_close'
            elif parts[0]=='adj' and len(parts)>1 and parts[1] in {'close','cls'}:
                base='adj_close'
        # Falls yfinance Format: qqq_close -> letzte Komponente ist relevant
        if sym_lower in c and c.endswith('_close'):
            base = 'close'
        mapped_cols[c] = base
    # Anwenden des Mappings (nur wenn keine Kollision mit existierendem Namen anderer Spalte ohne Mapping)
    new_cols = []
    counts = {}
    for c in df.columns:
        b = mapped_cols.get(c,c)
        counts[b] = counts.get(b,0)+1
        if counts[b]>1 and b in {'open','high','low','close','volume','adj_close'}:
            # eindeutiger machen
            new_cols.append(f"{b}_{counts[b]}")
        else:
            new_cols.append(b)
    df.columns = new_cols
    # Fallback: erste Spalte mit 'close' im Namen nehmen
    if 'close' not in df.columns:
        cand = [c for c in df.columns if 'close' in c]
        if cand:
            df['close'] = df[cand[0]]
    # Header row misuse heuristic
    if 'close' not in df.columns and len(df)>0:
        first = [str(x).strip().lower() for x in df.iloc[0].values]
        if 'close' in first and len(set(first))==len(first):
            df.columns = first
            df = df.iloc[1:]
    # Adj close fallback
    if 'close' not in df.columns and 'adj close' in df.columns:
        df = df.rename(columns={'adj close':'close'})
    if 'close' not in df.columns and 'adj_close' in df.columns:
        df = df.rename(columns={'adj_close':'close'})
    # Date inference
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
    elif 'timestamp' in df.columns:
        df['date'] = pd.to_datetime(df['timestamp'], unit='ms', errors='coerce')
    else:
        idx = df.index
        df['date'] = pd.to_datetime(idx, errors='coerce')
    keep_cols = ['date'] + [c for c in ['open','high','low','close','volume'] if c in df.columns]
    df = df[keep_cols].copy()
    # Dates tz-naiv erzwingen (verhindert spätere Vergleichsfehler)
    try:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['date'] = df['date'].dt.tz_localize(None)
    except Exception:
        try:
            df['date'] = df['date'].dt.tz_convert(None)
        except Exception:
            pass
    for c in ['open','high','low','close','volume']:
        if c in df.columns:
            try:
                df[c] = pd.to_numeric(df[c], errors='coerce')
            except Exception:
                pass
    if 'close' not in df.columns:
        raise ValueError("CSV/DF enthält keine 'close'-Spalte! Prüfe Datei.")
    df = df.dropna(subset=['date','close']).drop_duplicates(subset=['date']).sort_values('date')
    return df.reset_index(drop=True)
    # Wenn DataFrame leer ist, sofort zurückgeben
    if df is None or df.empty or len(df.columns) == 0:
        return pd.DataFrame()
    # Versuche, alle numerischen Spalten zu konvertieren (nur falls vorhanden)
    for c in ["open", "high", "low", "close"]:
        if c in df.columns:
            col = df[c]
            # Nur konvertieren, wenn es eine Series ist und 1D
            if isinstance(col, pd.Series) and col.ndim == 1 and not col.empty:
                try:
                    df[c] = pd.to_numeric(col, errors='coerce')
                except Exception:
                    pass
    # Spaltennamen normalisieren: trim, lower, unsichtbare Zeichen entfernen
    df.columns = [str(c).strip().lower().replace('\ufeff','').replace('"','').replace("'","") for c in df.columns]
    # Falls alles in einer Spalte: versuche, die erste Zeile als Header zu nehmen
    if len(df.columns) == 1 and ',' in df.columns[0]:
        df = df[df.columns[0]].str.split(',', expand=True)
        df.columns = [str(c).strip().lower() for c in df.iloc[0]]
        df = df.drop(df.index[0]).reset_index(drop=True)
    # Nochmals Spaltennamen normalisieren
    df.columns = [str(c).strip().lower().replace('\ufeff','').replace('"','').replace("'","") for c in df.columns]
    # RADIKALER FALLBACK: Wenn immer noch keine 'close'-Spalte, prüfe, ob die erste Zeile die Header ist
    if 'close' not in df.columns:
        # Prüfe, ob die erste Zeile wie ein Header aussieht (z.B. 'timestamp', 'open', ...)
        first_row = [str(x).strip().lower() for x in df.iloc[0].values]
        if 'close' in first_row:
            df.columns = first_row
            df = df.drop(df.index[0]).reset_index(drop=True)
        # Nochmals normalisieren
        df.columns = [str(c).strip().lower().replace('\ufeff','').replace('"','').replace("'","") for c in df.columns]
    # Wenn immer noch nicht: explizite Fehlermeldung
    if 'close' not in df.columns:
        raise ValueError("CSV/DF enthält keine 'close'-Spalte! Prüfe die Datei und Spaltennamen.")
    # Timestamp/Date konvertieren
    if 'timestamp' in df.columns:
        df['date'] = pd.to_datetime(df['timestamp'], unit='ms', errors='coerce')
    elif 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
    elif df.index.name in ['date','timestamp']:
        df['date'] = pd.to_datetime(df.index, errors='coerce')
    else:
        # Fallback: versuche, die erste Spalte als Datum zu interpretieren
        try:
            df['date'] = pd.to_datetime(df.iloc[:,0], errors='coerce')
        except Exception:
            pass
    return df
def robust_read_csv(path):
    # Versuche, das Trennzeichen automatisch zu erkennen
    with open(path, 'r', encoding='utf-8-sig') as f:
        sample = f.read(2048)
        sniffer = csv.Sniffer()
        try:
            dialect = sniffer.sniff(sample)
            sep = dialect.delimiter
        except Exception:
            sep = ','
    df = pd.read_csv(path, sep=sep, engine='python')
    # Wenn alles in einer Spalte steht, splitte manuell
    if len(df.columns) == 1 and sep in df.columns[0]:
        df = pd.read_csv(path, sep=sep, engine='python')
    if len(df.columns) == 1:
        # Fallback: splitte per Komma
        df = df[df.columns[0]].str.split(',', expand=True)
        df.columns = df.iloc[0]
        df = df.drop(df.index[0]).reset_index(drop=True)
    return df

def _make_tz_naive(series: pd.Series) -> pd.Series:
    try:
        return series.dt.tz_localize(None)
    except Exception:
        try:
            return series.dt.tz_convert(None)
        except Exception:
            return series

def _merge_history(symbol:str, cur:pd.DataFrame, hist_path:str) -> pd.DataFrame:
    if cur is None or cur.empty:
        cur = pd.DataFrame()
    if not os.path.exists(hist_path):
        return cur
    try:
        raw = robust_read_csv(hist_path)
        old = _normalize_yf_df(raw, symbol)
        if not old.empty:
            # Zeitzonen vereinheitlichen
            if 'date' in old.columns:
                old['date'] = _make_tz_naive(pd.to_datetime(old['date'], errors='coerce'))
            if 'date' in cur.columns:
                cur['date'] = _make_tz_naive(pd.to_datetime(cur['date'], errors='coerce'))
            merged = pd.concat([old, cur], ignore_index=True)
            merged = merged.dropna(subset=['date']).drop_duplicates(subset=['date']).sort_values('date').reset_index(drop=True)
            return merged
    except Exception as e:
        print(f"[WARN] Merge {os.path.basename(hist_path)}: {e}")
    return cur

def load_local_data(base_path: str):
    """Load data from local CSV files (resampled from tick/M30 data).
    
    Args:
        base_path: Base path without suffix, e.g. 'daten/xauusd'
                   Will load: {base_path}_daily.csv, {base_path}_h1.csv, {base_path}_m30.csv
    
    Returns:
        Tuple of (daily, h1, m30) DataFrames
    """
    import os
    
    daily_path = f"{base_path}_daily.csv"
    h1_path = f"{base_path}_h1.csv"
    m30_path = f"{base_path}_m30.csv"
    
    print(f"[LOCAL] Loading data from: {base_path}_*.csv")
    
    def _load_csv(path, label):
        if not os.path.exists(path):
            print(f"[LOCAL] {label} not found: {path}")
            return pd.DataFrame()
        
        df = pd.read_csv(path)
        
        # Normalize column names
        df.columns = [c.lower() for c in df.columns]
        
        # Parse date column
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        elif 'timestamp' in df.columns:
            df['date'] = pd.to_datetime(df['timestamp'])
            df = df.drop(columns=['timestamp'])
        
        # Ensure required columns exist
        required = ['open', 'high', 'low', 'close']
        for col in required:
            if col not in df.columns:
                print(f"[LOCAL] Warning: {label} missing column '{col}'")
                return pd.DataFrame()
        
        # Rename to match backtester expectations
        df = df.rename(columns={
            'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close',
            'volume': 'Volume', 'atr': 'ATR'
        })
        
        # Convert back to lowercase (backtester expectation)
        df.columns = [c.lower() for c in df.columns]
        
        print(f"[LOCAL] {label}: {len(df):,} bars | {df['date'].iloc[0].date()} to {df['date'].iloc[-1].date()}")
        return df
    
    daily = _load_csv(daily_path, "Daily")
    h1 = _load_csv(h1_path, "H1")
    m30 = _load_csv(m30_path, "M30")
    
    # Add all required indicators (ATR, ATR_PCT, EMA, RSI, ADX)
    # Use add_all_indicators to ensure consistency with yfinance path
    daily, h1, m30 = add_all_indicators(daily, h1, m30)
    
    return daily, h1, m30


def load_data():
    # Check if local data path is specified
    local_path = CFG.get("LOCAL_DATA_PATH")
    if local_path:
        print(f"[LOCAL] Using local data: {local_path}")
        return load_local_data(local_path)
    
    sym = CFG["SYMBOL"]
    print(f"[{CFG['_PROFILE']}] Lade {sym}: Daily={CFG['DAILY_PERIOD']} | 1H={CFG['H1_PERIOD']} | 30m={CFG['M30_PERIOD']}")
    if CFG.get("USE_CSV", False):
        print("[CSV] Lade Kursdaten aus CSV...")
        base = os.path.dirname(__file__)
        sym_sanit = sym.replace('=','_').replace('^','')
        # Kandidatenlisten (verschiedene Namensschemata und Ordner)
        def _candidates(prefix: str):
            core = [
                f"{prefix}_{sym}.csv",
                f"{prefix}_{sym_sanit}.csv",
                f"{sym}_{prefix}.csv",
                f"{sym_sanit}_{prefix}.csv",
                f"{prefix}.csv",
                f"{sym}.csv",
                f"{sym_sanit}.csv",
            ]
            # Duplikate entfernen, Reihenfolge beibehalten
            seen, uniq = set(), []
            for x in core:
                if x not in seen:
                    seen.add(x); uniq.append(x)
            paths = []
            for folder in [base, os.path.join(base, 'daten')]:
                for fname in uniq:
                    paths.append(os.path.join(folder, fname))
            return paths
        if sym == "EURUSD=X":
            # Spezielle bekannten Historien-Dateien
            daily_candidates = [os.path.join(base, "daten", "eurusd-d1-bid-1988-01-15-2025-09-07.csv")] + _candidates("daily")
            h1_candidates    = [os.path.join(base, "daten", "eurusd-h1-bid-2003-05-04T21-2025-09-07.csv")] + _candidates("h1")
            # M30 Pattern-Suche ergänzend zu Standard-Kandidaten
            m30_candidates = []
            for folder in [base, os.path.join(base, 'daten')]:
                if os.path.isdir(folder):
                    for fn in os.listdir(folder):
                        low = fn.lower()
                        if low.startswith('eurusd-m30-bid-') and low.endswith('.csv'):
                            m30_candidates.append(os.path.join(folder, fn))
            m30_candidates += _candidates("m30")
        else:
            daily_candidates = _candidates("daily")
            h1_candidates    = _candidates("h1")
            m30_candidates   = _candidates("m30")
        def _select(label, candidates):
            existing = [p for p in candidates if os.path.exists(p)]
            print(f"[CSV-TRY] {label} Kandidaten (erste 8 gezeigt):")
            for p in candidates[:8]:
                print(f"   - {p} {'(OK)' if os.path.exists(p) else ''}")
            return existing[0] if existing else candidates[0]
        daily_path = _select('Daily', daily_candidates)
        h1_path    = _select('H1', h1_candidates)
        m30_path   = _select('M30', m30_candidates)
        # Debug Pfade + Existenz
        def _pinfo(label, path):
            if not path:
                print(f"[CSV-PATH] {label}: (None)")
                return
            exists = os.path.exists(path)
            size = os.path.getsize(path) if exists else 0
            print(f"[CSV-PATH] {label}: {path} | exists={exists} | size={size}B")
        _pinfo('Daily', daily_path)
        _pinfo('H1', h1_path)
        _pinfo('M30', m30_path)
        # Einlesen nur wenn vorhanden
        daily = robust_read_csv(daily_path) if os.path.exists(daily_path) else pd.DataFrame()
        h1    = robust_read_csv(h1_path)    if os.path.exists(h1_path)    else pd.DataFrame()
        m30   = robust_read_csv(m30_path)   if os.path.exists(m30_path)   else pd.DataFrame()
        print(f"[CSV-RAW] daily shape={daily.shape} | h1 shape={h1.shape} | m30 shape={m30.shape}")
        daily = _normalize_yf_df(daily, sym) if not daily.empty else pd.DataFrame()
        h1    = _normalize_yf_df(h1, sym)    if not h1.empty    else pd.DataFrame()
        m30   = _normalize_yf_df(m30, sym)   if not m30.empty   else pd.DataFrame()
        print(f"[CSV-NORM] daily shape={daily.shape} | h1 shape={h1.shape} | m30 shape={m30.shape}")
        # Helper für Fallbacks
        def _yf(period, interval, _sym=sym):
            try:
                raw = yf.download(_sym, period=period, interval=interval, auto_adjust=True, group_by="column", progress=False)
                return _normalize_yf_df(raw, _sym)
            except Exception as e:
                print(f"[WARN] yfinance Fallback {interval} fehlgeschlagen: {e}")
                return pd.DataFrame()
        if daily.empty:
            print("[FALLBACK] daily CSV fehlt/leer -> lade von yfinance")
            daily = _yf(CFG['DAILY_PERIOD'], '1d')
        if h1.empty:
            print("[FALLBACK] h1 CSV fehlt/leer -> lade von yfinance")
            h1 = _yf(CFG['H1_PERIOD'], '1h')
        if m30.empty:
            print("[INFO] Keine m30 Daten (CSV leer/fehlend). Verwende nur 1H für Entries.")
        daily, h1, m30 = add_all_indicators(daily, h1, m30)
        for nm, dfx in [("Daily", daily), ("H1", h1), ("M30", m30)]:
            if not dfx.empty and 'date' in dfx.columns:
                try:
                    print(f"[SPAN] {nm}: {dfx['date'].iloc[0].date()} -> {dfx['date'].iloc[-1].date()} ({len(dfx)} Zeilen)")
                except Exception:
                    pass
        # Wenn Daily oder H1 nach Fallback immer noch leer -> Abbruch
        if daily.empty or h1.empty:
            print("[ERROR] Kritische Zeitreihe leer (Daily oder H1). Bitte CSV prüfen oder Internetverbindung für yfinance sicherstellen.")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        return daily, h1, m30
    else:
        # Erst yfinance ziehen (begrenzter Zeitraum)
        d = yf.download(sym, period=CFG["DAILY_PERIOD"], interval="1d",  auto_adjust=True, group_by="column", progress=False)
        h = yf.download(sym, period=CFG["H1_PERIOD"],   interval="1h",  auto_adjust=True, group_by="column", progress=False)
        m = yf.download(sym, period=CFG["M30_PERIOD"],  interval="30m", auto_adjust=True, group_by="column", progress=False)
        d = _normalize_yf_df(d, sym)
        h = _normalize_yf_df(h, sym)
        m = _normalize_yf_df(m, sym)
        # Automatischer Merge langer Historie nur falls explizit CSV-Modus aktiv (USER-WUNSCH)
        if sym == "EURUSD=X" and CFG.get("USE_CSV", False):
            base = os.path.dirname(__file__)
            d = _merge_history(sym, d, os.path.join(base, "daten", "eurusd-d1-bid-1988-01-15-2025-09-07.csv"))
            h = _merge_history(sym, h, os.path.join(base, "daten", "eurusd-h1-bid-2003-05-04T21-2025-09-07.csv"))
            m30_candidates = []
            for folder in [base, os.path.join(base, 'daten')]:
                if os.path.isdir(folder):
                    for fn in os.listdir(folder):
                        if fn.lower().startswith('eurusd-m30-bid-') and fn.lower().endswith('.csv'):
                            m30_candidates.append(os.path.join(folder, fn))
            if m30_candidates:
                m30_candidates.sort()
                hist_m30_path = m30_candidates[0]
                m = _merge_history(sym, m, hist_m30_path)
        # Indikatoren
        d, h, m = add_all_indicators(d, h, m)
        # Einheitlich TZ-naiv (Sicherheitsnetz gegen Mischformen)
        for df_ref in (d,h,m):
            if not df_ref.empty and 'date' in df_ref.columns:
                try:
                    df_ref['date'] = pd.to_datetime(df_ref['date'], errors='coerce').dt.tz_localize(None)
                except Exception:
                    try:
                        df_ref['date'] = pd.to_datetime(df_ref['date'], errors='coerce').dt.tz_convert(None)
                    except Exception:
                        pass
        # Debug-Spanne
        for nm,df in [("Daily",d),("H1",h),("M30",m)]:
            if not df.empty:
                print(f"[SPAN] {nm}: {df['date'].iloc[0].date()} -> {df['date'].iloc[-1].date()} ({len(df)} Zeilen)")
        return d, h, m

def add_indicators(df: pd.DataFrame):
    if df.empty:
        return df
    # Ensure 'close' exists
    if "close" not in df.columns:
        if "adj_close" in df.columns:
            df["close"] = df["adj_close"]
        elif "Close" in df.columns:
            df["close"] = df["Close"]
        else:
            df["close"] = np.nan
    # Ensure 'high' and 'low' exist
    if "high" not in df.columns:
        if "High" in df.columns:
            df["high"] = df["High"]
        else:
            df["high"] = df["close"]
    if "low" not in df.columns:
        if "Low" in df.columns:
            df["low"] = df["Low"]
        else:
            df["low"] = df["close"]
    # Now calculate indicators robustly
    high = pd.to_numeric(df["high"], errors="coerce")
    low = pd.to_numeric(df["low"], errors="coerce")
    close = pd.to_numeric(df["close"], errors="coerce")
    prev = close.shift(1)
    tr = pd.concat([(high - low).abs(), (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(CFG.get("ATR_PERIOD", 14), min_periods=1).mean()
    df["ATR_PCT"] = (df["ATR"] / close) * 100.0
    df["EMA_FAST"] = close.ewm(span=CFG.get("EMA_FAST", 21), adjust=False).mean()
    df["EMA_SLOW"] = close.ewm(span=CFG.get("EMA_SLOW", 55), adjust=False).mean()
    # RSI (Feature)
    delta = close.diff().fillna(0.0)
    up = delta.clip(lower=0).rolling(14).mean()
    down = (-delta.clip(upper=0)).rolling(14).mean()
    rs = up / (down + 1e-12)
    df["RSI"] = 100 - (100 / (1 + rs))
    return df

def add_all_indicators(daily: pd.DataFrame, h1: pd.DataFrame, m30: pd.DataFrame):
    daily = add_indicators(daily)
    h1 = add_indicators(h1)
    m30 = add_indicators(m30)
    # Add ADX to daily for regime analysis/filtering
    compute_adx(daily, n=14, out_col="ADX_14")
    return daily, h1, m30

# Einfacher ADX(14)-Berechner (Fallback, falls pandas_ta nicht verfügbar)
def compute_adx(df: pd.DataFrame, n: int = 14, out_col: str = "ADX_14"):
    if df.empty or not {"high","low","close"}.issubset(df.columns):
        return
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    prev_high = high.shift(1)
    prev_low = low.shift(1)
    prev_close = close.shift(1)
    up = high - prev_high
    down = prev_low - low
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    tr = pd.concat([(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0/n, adjust=False).mean()
    plus_di = 100.0 * (pd.Series(plus_dm, index=df.index).ewm(alpha=1.0/n, adjust=False).mean() / (atr + 1e-12))
    minus_di = 100.0 * (pd.Series(minus_dm, index=df.index).ewm(alpha=1.0/n, adjust=False).mean() / (atr + 1e-12))
    dx = 100.0 * ((plus_di - minus_di).abs() / ((plus_di + minus_di).abs() + 1e-12))
    adx = dx.ewm(alpha=1.0/n, adjust=False).mean()
    df[out_col] = adx

# --------------------------------------------------------------------------------------
# Elliott Engines
# --------------------------------------------------------------------------------------
class Dir(Enum):
    UP=1; DOWN=2

@dataclass
class Pivot:
    idx:int; price:float; kind:str  # 'H'/'L'
    confirm_idx:int = -1  # Index where pivot was confirmed (lookahead-free)

@dataclass
class Impulse:
    direction:Dir; points:List[Pivot]  # [p0..p5]

@dataclass
class ABC:
    direction:Dir; points:List[Pivot]  # [A0,A1,B1,C1]

class ElliottEngine:
    def __init__(self, zz_pct:float, zz_atr_mult:float, min_impulse_atr:float):
        self.zz_pct = zz_pct; self.zz_atr_mult = zz_atr_mult; self.min_imp = min_impulse_atr

    @staticmethod
    def _thr(base:float, atr:float, pct:float, atr_mult:float)->float:
        if pd.isna(atr): return base*pct
        return max(base*pct, atr*atr_mult)

    def zigzag(self, close:np.ndarray, atr:np.ndarray)->List[Pivot]:
        """ZigZag with realistic pivot confirmation.
        
        Each pivot now tracks confirm_idx: the bar index where the pivot was
        confirmed (when price moved away by the threshold). This is the earliest
        bar where a live system could know the pivot exists.
        """
        piv=[]
        if len(close)<3: return piv
        last=close[0]; hi=last; lo=last; hi_i=0; lo_i=0; direction=None
        for i in range(1,len(close)):
            p=close[i]
            thr=self._thr(last, atr[i] if atr is not None and i<len(atr) else np.nan, self.zz_pct, self.zz_atr_mult)
            if direction in (None, Dir.UP):
                if p>hi: hi=p; hi_i=i
                if hi-p>=thr:
                    # Pivot at hi_i is confirmed NOW at bar i (not retroactively at hi_i)
                    piv.append(Pivot(hi_i,float(hi),'H', confirm_idx=i))
                    last=hi; lo=p; lo_i=i; direction=Dir.DOWN
            if direction in (None, Dir.DOWN):
                if p<lo: lo=p; lo_i=i
                if p-lo>=thr:
                    # Pivot at lo_i is confirmed NOW at bar i
                    piv.append(Pivot(lo_i,float(lo),'L', confirm_idx=i))
                    last=lo; hi=p; hi_i=i; direction=Dir.UP
        piv.sort(key=lambda x:x.idx)
        cleaned=[]
        for p in piv:
            if not cleaned or cleaned[-1].kind!=p.kind:
                cleaned.append(p)
            else:
                if (p.kind=='H' and p.price>=cleaned[-1].price) or (p.kind=='L' and p.price<=cleaned[-1].price):
                    cleaned[-1]=p
        return cleaned

    def detect_impulses(self, piv:List[Pivot], close:np.ndarray, atr:np.ndarray)->List[Impulse]:
        res=[]; i=0
        while i<=len(piv)-6:
            s=piv[i:i+6]; kinds=''.join(p.kind for p in s)
            if kinds=='LHLHLH':
                p0,p1,p2,p3,p4,p5=s
                w1=p1.price-p0.price; w3=p3.price-p2.price
                if p2.price<=p0.price or w1<=0 or w3<0.6*w1: i+=1; continue
                if p4.price<=p1.price*0.98: i+=1; continue
                atr_b=atr[min(p3.idx, len(atr)-1)]
                if atr_b>0 and (w3/atr_b)<self.min_imp: i+=1; continue
                res.append(Impulse(Dir.UP,[p0,p1,p2,p3,p4,p5])); i+=3
            elif kinds=='HLHLHL':
                p0,p1,p2,p3,p4,p5=s
                w1=p0.price-p1.price; w3=p2.price-p3.price
                if p2.price>=p0.price or w1<=0 or w3<0.6*w1: i+=1; continue
                if p4.price>=p1.price*1.02: i+=1; continue
                atr_b=atr[min(p3.idx, len(atr)-1)]
                if atr_b>0 and (abs(w3)/atr_b)<self.min_imp: i+=1; continue
                res.append(Impulse(Dir.DOWN,[p0,p1,p2,p3,p4,p5])); i+=3
            else:
                i+=1
        return res

    def detect_abcs(self, piv:List[Pivot])->List[ABC]:
        out=[]; i=0
        while i<=len(piv)-4:
            s=piv[i:i+4]; kinds=''.join(p.kind for p in s)
            if kinds=='HLHL':
                h0,l1,h1,l2=s
                A=h0.price-l1.price; B=h1.price-l1.price
                if A<=0 or not (0.3<=B/A<=0.86) or not (l2.price<l1.price): i+=1; continue
                out.append(ABC(Dir.DOWN,[h0,l1,h1,l2])); i+=2
            elif kinds=='LHLH':
                l0,h1,l1,h2=s
                A=h1.price-l0.price; B=h1.price-l1.price
                if A<=0 or not (0.3<=B/A<=0.86) or not (h2.price>h1.price): i+=1; continue
                out.append(ABC(Dir.UP,[l0,h1,l1,h2])); i+=2
            else:
                i+=1
        return out

    @staticmethod
    def fib_zone(A:float,B:float,d:Dir,zone:Tuple[float,float])->Tuple[float,float]:
        lo,hi=sorted(zone)
        if d==Dir.UP:
            L=B-A; zL=B-L*hi; zH=B-L*lo
        else:
            L=A-B; zL=B+L*lo; zH=B+L*hi
        return (min(zL,zH),max(zL,zH))

    @staticmethod
    def fib_ext(A:float,B:float,d:Dir,ext:float)->float:
        return B+(B-A)*(ext-1.0) if d==Dir.UP else B-(A-B)*(ext-1.0)

# --------------------------------------------------------------------------------------
# Strategy types
# --------------------------------------------------------------------------------------
@dataclass
class Setup:
    setup:str; direction:Dir; start_time:pd.Timestamp; entry_tf:str
    zone:Tuple[float,float]; stop_ref:float; tp1:float; tp2:float; meta:Dict

@dataclass
class SimTrade:
    entry_tf:str; entry_idx:int; exit_idx:int
    entry:float; exit:float; per_share:float; risk_per_share:float
    setup:str; direction:str; time_in:pd.Timestamp; time_out:pd.Timestamp
    stop:float; tp1:float; tp2:float; mae_r:float; mfe_r:float
    features:Dict; label:int; prob:Optional[float]=None  # gespeicherte ML-Wahrscheinlichkeit (OOS gesetzt)

@dataclass
class Trade:
    entry_tf:str; entry_idx:int; exit_idx:int
    entry:float; exit:float; pnl:float; size:int; rr:float
    setup:str; direction:str; time_in:pd.Timestamp; time_out:pd.Timestamp
    stop:float; tp1:float; tp2:float; mae_r:float; mfe_r:float
    prob:Optional[float]=None; risk_per_share:Optional[float]=None

# --------------------------------------------------------------------------------------
# Filters/Sim
# --------------------------------------------------------------------------------------
def ema_trend_ok(r:pd.Series,d:Dir)->bool:
    if not CFG["USE_EMA_TREND"]: return True
    if CFG.get("REQUIRE_PRICE_ABOVE_EMA_FAST", True):
        return (r["EMA_FAST"]>r["EMA_SLOW"] and r["close"]>r["EMA_FAST"]) if d==Dir.UP else (r["EMA_FAST"]<r["EMA_SLOW"] and r["close"]<r["EMA_FAST"])
    else:
        return (r["EMA_FAST"]>r["EMA_SLOW"]) if d==Dir.UP else (r["EMA_FAST"]<r["EMA_SLOW"])

def daily_trend_ok(daily:pd.DataFrame, ts:pd.Timestamp, d:Dir)->bool:
    if not CFG["USE_DAILY_EMA"]: return True
    idx=daily[daily["date"]<=pd.Timestamp(ts)].index
    if len(idx)==0: return True
    r=daily.loc[idx.max()]
    return (r["EMA_FAST"]>r["EMA_SLOW"]) if d==Dir.UP else (r["EMA_FAST"]<r["EMA_SLOW"])

def vol_ok(r:pd.Series)->bool:
    p=float(r["ATR_PCT"]); return CFG["ATR_PCT_MIN"]<=p<=CFG["ATR_PCT_MAX"]

def df_for_tf(h1:pd.DataFrame, m30:pd.DataFrame, tf:str)->pd.DataFrame:
    return m30 if tf=="30m" else h1

def idx_from_time(df:pd.DataFrame, ts:pd.Timestamp)->Optional[int]:
    i=df[df["date"]>=pd.Timestamp(ts)].index
    return i.min() if len(i)>0 else None

def first_touch(df:pd.DataFrame, start_ts:pd.Timestamp, zone:Tuple[float,float], window:int)->Optional[int]:
    start_i=idx_from_time(df,start_ts)
    if start_i is None: return None
    zl,zh=zone; end_i=min(start_i+window, len(df)-1)
    for i in range(start_i,end_i+1):
        lo=float(df.iloc[i]["low"]); hi=float(df.iloc[i]["high"]); cl=float(df.iloc[i]["close"])
        if (lo<=zh and hi>=zl) or (zl<=cl<=zh): return i
    return None

def confirm_idx(df:pd.DataFrame, touch_i:int, d:Dir, bars:int, allow_touch:bool)->Optional[int]:
    if not CFG["REQUIRE_CONFIRM"]: return touch_i
    end=min(touch_i+bars, len(df)-1)
    prev_hi=float(df.iloc[max(0,touch_i-1)]["high"]); prev_lo=float(df.iloc[max(0,touch_i-1)]["low"])
    for i in range(touch_i,end+1):
        r=df.iloc[i]; op=float(r["open"]); cl=float(r["close"]); ef=float(r["EMA_FAST"]); es=float(r["EMA_SLOW"])
        if "break_prev_extreme" in CFG["CONFIRM_RULES"]:
            if d==Dir.UP and cl>prev_hi: return i
            if d==Dir.DOWN and cl<prev_lo: return i
        if "ema_fast_cross" in CFG["CONFIRM_RULES"]:
            if d==Dir.UP and cl>ef and ef>es: return i
            if d==Dir.DOWN and cl<ef and ef<es: return i
    return end if allow_touch else None

def simulate(df:pd.DataFrame, entry_i:int, entry:float, d:Dir, stop:float, tp1:float, tp2:float, max_bars:int,
             higher_tf_df:pd.DataFrame=None, mom_exit_bars:int=0, mom_period:int=14)->Tuple[int,float,float,float,float]:
    """
    Simulate trade with optional momentum-based exit on higher TF.
    mom_exit_bars: number of consecutive declining momentum bars to trigger exit (0=disabled)
    """
    pos=1.0; realized=0.0; end=min(entry_i+max_bars, len(df)-1)
    R=abs(entry-stop); extreme=entry; mae=0.0; mfe=0.0
    
    # Momentum exit tracking
    declining_mom_count = 0
    last_mom = None
    entry_time = df.iloc[entry_i]["date"] if "date" in df.columns else None
    
    for i in range(entry_i+1,end+1):
        r=df.iloc[i]; lo=float(r["low"]); hi=float(r["high"])
        current_time = r["date"] if "date" in df.columns else None
        
        # Check momentum exit on higher TF (if enabled)
        if mom_exit_bars > 0 and higher_tf_df is not None and current_time is not None and pos > 0:
            # Find corresponding bar in higher TF
            htf_mask = higher_tf_df["date"] <= current_time
            if htf_mask.any():
                htf_idx = htf_mask.sum() - 1
                if htf_idx >= mom_period:
                    # Calculate momentum on higher TF
                    price_now = float(higher_tf_df.iloc[htf_idx]["close"])
                    price_past = float(higher_tf_df.iloc[htf_idx - mom_period]["close"])
                    current_mom = (price_now - price_past) / max(price_past, 1e-9)
                    
                    # Track declining momentum
                    if last_mom is not None:
                        # For LONG: momentum should stay positive/increasing
                        # For SHORT: momentum should stay negative/decreasing
                        if d == Dir.UP:
                            if current_mom < last_mom:  # momentum weakening
                                declining_mom_count += 1
                            else:
                                declining_mom_count = 0
                        else:  # SHORT
                            if current_mom > last_mom:  # momentum weakening (becoming less negative)
                                declining_mom_count += 1
                            else:
                                declining_mom_count = 0
                        
                        # Exit if momentum declined for N consecutive bars
                        if declining_mom_count >= mom_exit_bars:
                            exit_price = float(r["close"])
                            if d == Dir.UP:
                                realized += (exit_price - entry) * pos
                                mfe = max(mfe, (hi - entry) / R)
                            else:
                                realized += (entry - exit_price) * pos
                                mfe = max(mfe, (entry - lo) / R)
                            return i, exit_price, realized, mae, mfe
                    
                    last_mom = current_mom
        
        if d==Dir.UP:
            extreme=max(extreme,hi); mae=min(mae,(lo-entry)/R); mfe=max(mfe,(hi-entry)/R)
            if pos==1.0 and (extreme-entry)>=R: stop=max(stop,entry)  # BE nach +1R
            if pos==1.0 and hi>=tp1:
                realized+=(tp1-entry)*0.5; pos=0.5; stop=entry
            if lo<=stop:
                realized+=(stop-entry)*pos; return i, stop, realized, mae, mfe
            if pos==0.5 and hi>=tp2:
                realized+=(tp2-entry)*0.5; return i, tp2, realized, mae, mfe
        else:
            extreme=min(extreme,lo); mae=min(mae,(entry-float(r["high"]))/R); mfe=max(mfe,(entry-lo)/R)
            if pos==1.0 and (entry-extreme)>=R: stop=min(stop,entry)
            if pos==1.0 and lo<=tp1:
                realized+=(entry-tp1)*0.5; pos=0.5; stop=entry
            if float(r["high"])>=stop:
                realized+=(entry-stop)*pos; return i, stop, realized, mae, mfe
            if pos==0.5 and lo<=tp2:
                realized+=(entry-tp2)*0.5; return i, tp2, realized, mae, mfe
    last=float(df.iloc[end]["close"])
    if d==Dir.UP:
        realized+=(last-entry)*pos; mfe=max(mfe,(last-entry)/R)
    else:
        realized+=(entry-last)*pos; mfe=max(mfe,(entry-last)/R)
    return end, last, realized, mae, mfe

# --------------------------------------------------------------------------------------
# Features (ML)
# --------------------------------------------------------------------------------------
FEATURE_COLUMNS = [
    "dir_up","setup_W3","setup_W5","setup_C",
    "atr_pct","atr_pct_rank","atr_slope","ema_fast_slow_pct","ema_fast_slow_slope","price_above_ema_fast",
    "zone_width_pct","dist_to_zone_center_pct","zone_pos","atr_to_zone",
    "rsi","rsi_z","hour_sin","hour_cos","dow_sin","dow_cos","is_us_session",
    "prev_win_rate_20"
]

def build_features(df: pd.DataFrame, entry_idx: int, d: Dir, setup: str, zone: Tuple[float,float]) -> Dict:
    r = df.iloc[entry_idx]
    close = float(r["close"])
    ef = float(r["EMA_FAST"]); es = float(r["EMA_SLOW"])
    atr_pct = float(r.get("ATR_PCT", 0.0))
    # Rolling helper windows
    win_slope = 10
    # ATR slope (pct change over window)
    atr_series = df["ATR_PCT"].astype(float)
    if entry_idx >= win_slope and not np.isnan(atr_series.iloc[entry_idx-win_slope]):
        atr_slope = (atr_series.iloc[entry_idx] - atr_series.iloc[entry_idx-win_slope]) / max(1e-9, atr_series.iloc[entry_idx-win_slope])
    else:
        atr_slope = 0.0
    # EMA slope
    ema_fast_series = df["EMA_FAST"].astype(float)
    ema_slow_series = df["EMA_SLOW"].astype(float)
    if entry_idx >= win_slope and not np.isnan(ema_fast_series.iloc[entry_idx-win_slope]):
        ema_fast_slope = (ema_fast_series.iloc[entry_idx] - ema_fast_series.iloc[entry_idx-win_slope]) / max(1e-9, ema_fast_series.iloc[entry_idx-win_slope])
        ema_fast_slow_slope = ((ema_fast_series.iloc[entry_idx]-ema_slow_series.iloc[entry_idx]) - (ema_fast_series.iloc[entry_idx-win_slope]-ema_slow_series.iloc[entry_idx-win_slope])) / max(1e-9, abs(ema_fast_series.iloc[entry_idx-win_slope]-ema_slow_series.iloc[entry_idx-win_slope]))
    else:
        ema_fast_slope = 0.0
        ema_fast_slow_slope = 0.0
    ema_fast_slow_pct = (ef - es) / close if close else 0.0
    price_above_ema_fast = (close - ef) / close if close else 0.0
    rsi = float(r.get("RSI", 50.0))
    rsi_window = 50
    if "RSI" in df.columns and entry_idx >= rsi_window:
        rsi_hist = df["RSI"].iloc[entry_idx-rsi_window:entry_idx].astype(float)
        mu = float(rsi_hist.mean()); sd = float(rsi_hist.std(ddof=1))
        rsi_z = (rsi - mu)/sd if sd>1e-9 else 0.0
    else:
        rsi_z = 0.0
    zl, zh = zone
    zone_width = abs(zh - zl)
    zone_width_pct = zone_width / close if close else 0.0
    dist_to_zone_center_pct = abs(close - (zl + zh) / 2) / close if close else 0.0
    zone_pos = (close - zl) / max(1e-9, (zh - zl)) if zh>zl else 0.5
    atr_to_zone = (atr_pct/100.0) / max(1e-9, zone_width_pct) if zone_width_pct>0 else 0.0
    ts = pd.Timestamp(r["date"]); hour = ts.hour + ts.minute/60.0; dow = ts.weekday()
    hour_sin = math.sin(2*math.pi*hour/24.0); hour_cos = math.cos(2*math.pi*hour/24.0)
    dow_sin = math.sin(2*math.pi*dow/7.0);     dow_cos = math.cos(2*math.pi*dow/7.0)
    # US Session (13:30 - 20:00 UTC grob für NY Cash + overlap)
    is_us_session = 1.0 if 13.5 <= hour <= 20.0 else 0.0
    # ATR rank (percentile over trailing 250)
    if entry_idx >= 30:
        lookback = df["ATR_PCT"].iloc[max(0, entry_idx-250):entry_idx].astype(float)
        atr_pct_rank = float((lookback < atr_pct).mean()) if len(lookback)>5 else 0.0
    else:
        atr_pct_rank = 0.0
    return {
        "dir_up": 1.0 if d == Dir.UP else 0.0,
        "setup_W3": 1.0 if setup == "W3" else 0.0,
        "setup_W5": 1.0 if setup == "W5" else 0.0,
        "setup_C": 1.0 if setup == "C" else 0.0,
        "atr_pct": atr_pct,
        "atr_pct_rank": atr_pct_rank,
        "atr_slope": atr_slope,
        "ema_fast_slow_pct": ema_fast_slow_pct,
        "ema_fast_slow_slope": ema_fast_slow_slope,
        "price_above_ema_fast": price_above_ema_fast,
        "zone_width_pct": zone_width_pct,
        "dist_to_zone_center_pct": dist_to_zone_center_pct,
        "zone_pos": zone_pos,
        "atr_to_zone": atr_to_zone,
        "rsi": rsi if not np.isnan(rsi) else 50.0,
        "rsi_z": rsi_z,
        "hour_sin": hour_sin, "hour_cos": hour_cos,
        "dow_sin": dow_sin, "dow_cos": dow_cos,
        "is_us_session": is_us_session,
        # Platzhalter für spätere Injektion prev_win_rate_20
        # NOTE: prev_win_rate is potentially leaky (uses future labels); set to 0 in neutral mode
        "prev_win_rate_20": 0.0
    }

# --------------------------------------------------------------------------------------
# Backtester (inkl. ML mit Mindest-Pass-Rate)
# --------------------------------------------------------------------------------------
class Backtester:
    def __init__(self, daily:pd.DataFrame, h1:pd.DataFrame, m30:pd.DataFrame):
        self.daily = daily
        self.h1 = h1
        self.m30 = m30
        self.primary_engine = ElliottEngine(CFG["PRIMARY_ZZ_PCT"], CFG["PRIMARY_ZZ_ATR_MULT"], CFG["PRIMARY_MIN_IMP_ATR"])
        self.h1_engine = ElliottEngine(CFG["H1_ZZ_PCT"], CFG["H1_ZZ_ATR_MULT"], CFG["H1_MIN_IMP_ATR"])
        self.prim_imp: List[Impulse] = []
        self.prim_abc: List[ABC] = []
        self.impulses: List[Impulse] = []
        self.abcs: List[ABC] = []
        self.setups: List[Setup] = []
        self.sim_trades: List[SimTrade] = []
        self.trades: List[Trade] = []
        self.equity: List[Dict] = []
        self.model = None
        self.threshold = 0.5
        self.telemetry = dict(setups=0, filtered_daily=0, filtered_ema=0, filtered_vol=0, filtered_volatility=0, filtered_regime=0, filtered_momentum=0, filtered_pf=0, no_touch=0, no_confirm=0, accepted=0, vola_sized=0)
        
        # Volatility Forecast sizing (walk-forward)
        self.vola_forecaster = None
        self.vola_forecast_ready = False
        if CFG.get('USE_VOLA_FORECAST', False) and VOLA_FORECAST_AVAILABLE:
            try:
                window = CFG.get('VOLA_FORECAST_WINDOW', 252)
                self.vola_forecaster = create_vola_forecaster(train_window=window)
                # Prepare daily data for forecasting
                if len(self.daily) > window:
                    self.daily = self.vola_forecaster.prepare_data(self.daily.copy())
                    self.vola_forecast_ready = True
                    print(f"[VOLA] Forecaster initialized with {window}-day window")
                else:
                    print(f"[VOLA] Not enough daily data ({len(self.daily)} < {window})")
            except Exception as e:
                print(f"[VOLA] Forecaster init failed: {e}")

    # ---------- Struktur ----------
    def analyze_structure(self):
        piv_d = self.primary_engine.zigzag(self.daily["close"].values, self.daily["ATR"].values)
        self.prim_imp = self.primary_engine.detect_impulses(piv_d, self.daily["close"].values, self.daily["ATR"].values)
        self.prim_abc = self.primary_engine.detect_abcs(piv_d)
        piv_h = self.h1_engine.zigzag(self.h1["close"].values, self.h1["ATR"].values)
        self.impulses = self.h1_engine.detect_impulses(piv_h, self.h1["close"].values, self.h1["ATR"].values)
        self.abcs     = self.h1_engine.detect_abcs(piv_h)

    def _preferred_tf(self, start_time:pd.Timestamp)->str:
        if not self.m30.empty and self.m30["date"].iloc[0] <= start_time <= self.m30["date"].iloc[-1]:
            return "30m"
        return "1h"

    def build_setups(self):
        """Build setups using lookahead-free pivot confirmation.
        
        Setup start time is now based on when the LAST pivot of the pattern
        was confirmed (confirm_idx), not when it occurred (idx). This prevents
        lookahead bias where the backtest 'knows' a pivot before it's confirmed.
        """
        self.setups.clear()
        for imp in self.impulses:
            p0,p1,p2,p3,p4,p5 = imp.points
            # W3 setup: needs p2 confirmed (end of wave 2)
            # Use confirmation index of last required pivot for this setup
            # For W3, we need at least p2 confirmed to know wave 1-2 structure
            setup_confirm_idx = max(p2.confirm_idx if p2.confirm_idx >= 0 else p2.idx + 1, 0)
            if setup_confirm_idx >= len(self.h1):
                continue
            z3 = self.h1_engine.fib_zone(p0.price,p1.price,imp.direction,CFG["ENTRY_ZONE_W3"])
            t3 = self.h1.iloc[setup_confirm_idx]["date"]; tf3=self._preferred_tf(t3)
            tp1_3=self.h1_engine.fib_ext(p0.price,p1.price,imp.direction,CFG["TP1"])
            tp2_3=self.h1_engine.fib_ext(p0.price,p1.price,imp.direction,CFG["TP2"])
            self.setups.append(Setup("W3", imp.direction, t3, tf3, z3, p0.price, tp1_3, tp2_3, dict(src="impulse", confirm_delay=setup_confirm_idx - p2.idx)))
            if CFG["USE_W5"]:
                # W5 setup: needs p4 confirmed
                setup_confirm_idx_w5 = max(p4.confirm_idx if p4.confirm_idx >= 0 else p4.idx + 1, 0)
                if setup_confirm_idx_w5 >= len(self.h1):
                    continue
                z5=self.h1_engine.fib_zone(p2.price,p3.price,imp.direction,CFG["ENTRY_ZONE_W5"])
                t5=self.h1.iloc[setup_confirm_idx_w5]["date"]; tf5=self._preferred_tf(t5)
                tp1_5=self.h1_engine.fib_ext(p2.price,p3.price,imp.direction,CFG["TP1"])
                tp2_5=self.h1_engine.fib_ext(p2.price,p3.price,imp.direction,CFG["TP2"])
                self.setups.append(Setup("W5", imp.direction, t5, tf5, z5, p2.price, tp1_5, tp2_5, dict(src="impulse", confirm_delay=setup_confirm_idx_w5 - p4.idx)))
        for abc in self.abcs:
            a0,a1,b1,c1 = abc.points
            # ABC setup: needs b1 (the correction low/high) confirmed
            setup_confirm_idx = max(b1.confirm_idx if b1.confirm_idx >= 0 else b1.idx + 1, 0)
            if setup_confirm_idx >= len(self.h1):
                continue
            zc=self.h1_engine.fib_zone(a0.price,a1.price,abc.direction,CFG["ENTRY_ZONE_C"])
            tc=self.h1.iloc[setup_confirm_idx]["date"]; tfc=self._preferred_tf(tc)
            tp1_c=self.h1_engine.fib_ext(a0.price,a1.price,abc.direction,CFG["TP1"])
            tp2_c=self.h1_engine.fib_ext(a0.price,a1.price,abc.direction,CFG["TP2"])
            self.setups.append(Setup("C", abc.direction, tc, tfc, zc, b1.price, tp1_c, tp2_c, dict(src="abc", confirm_delay=setup_confirm_idx - b1.idx)))
        self.setups.sort(key=lambda s:s.start_time)
        self.telemetry["setups"]=len(self.setups)

    # ---------- Simulation ----------
    def simulate_all(self):
        self.sim_trades.clear()
        for sp in self.setups:
            # Regime-Filter (ADX) als erstes Gate
            if CFG.get("USE_ADX", False):
                try:
                    didx=self.daily[self.daily["date"]<=pd.Timestamp(sp.start_time)].index
                    if len(didx)>0 and "ADX_14" in self.daily.columns:
                        cur_adx=float(self.daily.loc[didx.max(), "ADX_14"])
                        if not np.isnan(cur_adx) and cur_adx < CFG.get("ADX_TREND_THRESHOLD",25):
                            self.telemetry["filtered_regime"] = self.telemetry.get("filtered_regime",0)+1
                            continue
                except Exception:
                    pass
            df = df_for_tf(self.h1, self.m30, sp.entry_tf)
            if df.empty: continue
            start_i = idx_from_time(df, sp.start_time)
            if start_i is None: self.telemetry["no_touch"]+=1; continue
            if not ema_trend_ok(df.loc[start_i], sp.direction):
                self.telemetry["filtered_ema"]+=1; continue
            if not vol_ok(df.loc[start_i]):
                self.telemetry["filtered_vol"]+=1; continue
            
            # Volatility Regime Filter (filter high/low vola phases)
            vola_min_pct = CFG.get("VOLA_FILTER_MIN", 0)  # min ATR percentile (0-100)
            vola_max_pct = CFG.get("VOLA_FILTER_MAX", 100)  # max ATR percentile (0-100)
            if vola_min_pct > 0 or vola_max_pct < 100:
                try:
                    entry_date = pd.Timestamp(sp.start_time).normalize()
                    didx = self.daily[self.daily["date"] <= entry_date].index
                    if len(didx) > 60:
                        current_atr = float(self.daily.loc[didx.max(), "ATR_PCT"])
                        lookback = 60
                        start_idx = max(0, didx.max() - lookback)
                        historical_atr = self.daily["ATR_PCT"].iloc[start_idx:didx.max()+1].dropna()
                        if len(historical_atr) >= 10:
                            percentile = (historical_atr < current_atr).sum() / len(historical_atr) * 100
                            if percentile < vola_min_pct or percentile > vola_max_pct:
                                self.telemetry["filtered_vola_regime"] = self.telemetry.get("filtered_vola_regime", 0) + 1
                                continue
                except Exception:
                    pass
            
            # Momentum Filter (like live system)
            if CFG.get("USE_MOMENTUM_FILTER", False):
                mom_period = CFG.get("MOMENTUM_PERIOD", 14)
                mom_threshold = CFG.get("MOMENTUM_THRESHOLD", 0.0)
                if start_i >= mom_period:
                    price_now = float(df.iloc[start_i]["close"])
                    price_past = float(df.iloc[start_i - mom_period]["close"])
                    momentum = (price_now - price_past) / max(price_past, 1e-9)
                    # LONG: momentum sollte positiv sein, SHORT: negativ
                    if sp.direction == Dir.UP and momentum < mom_threshold:
                        self.telemetry["filtered_momentum"] = self.telemetry.get("filtered_momentum", 0) + 1
                        continue
                    elif sp.direction == Dir.DOWN and momentum > -mom_threshold:
                        self.telemetry["filtered_momentum"] = self.telemetry.get("filtered_momentum", 0) + 1
                        continue

            win = CFG["ENTRY_WINDOW_M30"] if sp.entry_tf=="30m" else CFG["ENTRY_WINDOW_H1"]
            t_idx = first_touch(df, sp.start_time, sp.zone, win)
            if t_idx is None: self.telemetry["no_touch"]+=1; continue
            bars = CFG["CONFIRM_BARS_M30"] if sp.entry_tf=="30m" else CFG["CONFIRM_BARS_H1"]
            e_idx = confirm_idx(df, t_idx, sp.direction, bars, CFG["ALLOW_TOUCH_IF_NO_CONFIRM"])
            if e_idx is None: self.telemetry["no_confirm"]+=1; continue

            atr=float(df.iloc[e_idx]["ATR"])
            atr_mult = CFG["ATR_MULT_BUFFER"]
            buffer=atr_mult*atr
            stop = sp.stop_ref - buffer if sp.direction==Dir.UP else sp.stop_ref + buffer
            entry=float(df.iloc[e_idx]["close"])
            rps=abs(entry-stop)
            
            # Minimum stop distance: at least 0.3% of entry price (prevents absurd position sizes)
            min_stop_pct = CFG.get("MIN_STOP_PCT", 0.003)  # 0.3% default
            min_rps = entry * min_stop_pct
            if rps < min_rps:
                # Adjust stop to minimum distance
                if sp.direction == Dir.UP:
                    stop = entry - min_rps
                else:
                    stop = entry + min_rps
                rps = min_rps
            
            if rps<=1e-9: continue
            
            # Minimum Profit Factor Filter (wie Live System)
            min_pf = CFG.get("MIN_PF", 0.0)
            if min_pf > 0:
                tp_distance = abs(sp.tp1 - entry)
                stop_distance = rps
                expected_pf = tp_distance / stop_distance if stop_distance > 0 else 0.0
                if expected_pf < min_pf:
                    self.telemetry["filtered_pf"] = self.telemetry.get("filtered_pf", 0) + 1
                    continue

            max_hold=CFG["MAX_HOLD_M30"] if sp.entry_tf=="30m" else CFG["MAX_HOLD_H1"]
            
            # Momentum Exit: use H1 as higher TF for M30 entries, Daily for H1 entries
            mom_exit_bars = CFG.get("MOMENTUM_EXIT_BARS", 0)
            mom_period = CFG.get("MOMENTUM_PERIOD", 14)
            if mom_exit_bars > 0:
                # Higher TF: H1 for M30 entries, Daily for H1 entries
                higher_tf_df = self.h1 if sp.entry_tf == "30m" else self.daily
                x_idx,x_price,ps,mae,mfe = simulate(df, e_idx, entry, sp.direction, stop, sp.tp1, sp.tp2, max_hold,
                                                     higher_tf_df=higher_tf_df, mom_exit_bars=mom_exit_bars, mom_period=mom_period)
            else:
                x_idx,x_price,ps,mae,mfe = simulate(df, e_idx, entry, sp.direction, stop, sp.tp1, sp.tp2, max_hold)
            
            feats = build_features(df, e_idx, sp.direction, sp.setup, sp.zone)
            label = 1 if ps>0 else 0

            self.sim_trades.append(SimTrade(sp.entry_tf, e_idx, x_idx, entry, x_price, ps, rps,
                                            sp.setup, "LONG" if sp.direction==Dir.UP else "SHORT",
                                            df.iloc[e_idx]["date"], df.iloc[x_idx]["date"],
                                            stop, sp.tp1, sp.tp2, mae, mfe, feats, label))
            self.telemetry["accepted"]+=1
        # Rolling prev win rate (based on per_share >0 among prior simulated trades)
        wins_so_far=0; total_so_far=0
        for t in sorted(self.sim_trades, key=lambda x:x.time_in):
            if total_so_far>=1:
                prev_win_rate = wins_so_far / max(1,total_so_far)
            else:
                prev_win_rate = 0.0
            t.features["prev_win_rate_20"] = prev_win_rate if total_so_far<20 else (
                sum(1 for s in self.sim_trades if s.time_in < t.time_in and s.per_share>0 and (t.time_in - s.time_in).days <= 3650 and (t.time_in - s.time_in).total_seconds()>=0 and s.time_in >= (t.time_in - pd.Timedelta(days=3650))) / max(1, min(20, total_so_far))
            )
            # update counters
            wins_so_far += 1 if t.per_share>0 else 0
            total_so_far += 1

    # ---------- ML ----------
    def _XY(self, trades:List[SimTrade])->Tuple[pd.DataFrame,np.ndarray]:
        X=pd.DataFrame([t.features for t in trades])
        for c in FEATURE_COLUMNS:
            if c not in X.columns: X[c]=0.0
        return X[FEATURE_COLUMNS], np.array([t.label for t in trades], dtype=int)

    def _make_calibrator(self, estimator, method, cv):
        # Nicht genutzt in der Original-Baseline
        return estimator

    def train_model(self, train_trades:List[SimTrade]):
        X,y=self._XY(train_trades)
        # Stärkeres Modell (mehr Trees, flacher Lernschritt)
        clf=GradientBoostingClassifier(random_state=42, n_estimators=400, learning_rate=0.03, max_depth=3).fit(X,y)
        self.model=clf
        # Erst Standard Schwelle
        self.threshold=0.5
        probs=clf.predict_proba(X)[:,1]
        self.ml_train_pass_rate = float((probs>=self.threshold).mean())
        # Erwartungswert-optimierte Schwelle (auf Trainingsdaten – heuristisch, vermeidet Overfit durch Mindestanzahl)
        wins=[t.per_share for t in train_trades if t.per_share>0]; losses=[t.per_share for t in train_trades if t.per_share<=0]
        avg_win=np.mean(wins) if wins else 1.0; avg_loss=abs(np.mean(losses)) if losses else 1.0
        cand=np.quantile(probs, np.linspace(0.2,0.95,30))
        best_thr=self.threshold; best_ev=-1e18
        for thr in cand:
            mask=probs>=thr
            if mask.sum()<10: continue
            p_sel=y[mask]
            p_rate=p_sel.mean()
            ev = p_rate*avg_win - (1-p_rate)*avg_loss
            if ev>best_ev:
                best_ev=ev; best_thr=float(thr)
        self.threshold=best_thr

    # ------------------------------------------------------------
    # Constrained Threshold + Risk Optimizer
    # Ziele über CFG optional definierbar (Fallback Defaults):
    #  TARGET_WINRATE_MIN (0.70), TARGET_WINRATE_MAX (0.80), TARGET_CAGR_MIN (90), TARGET_MAX_DD (-5.0), AUTO_RISK_SCALE (True)
    # ------------------------------------------------------------
    def _simulate_equity_for_threshold(self, thr: float, train_until: pd.Timestamp):
        """Repliziere Kern von build_equity für OOS Trades bei gegebenem Threshold.
        Rückgabe: dict(winrate, trades, cagr, max_dd, total_return)
        Vereinfachung: nutzt aktuelles CFG Risiko & Prob Sizing.
        """
        if self.model is None:
            return None
        # Precompute probabilities
        probs_cache = {}
        for sim in self.sim_trades:
            if sim.time_in <= train_until:
                continue
            X=pd.DataFrame([sim.features])[FEATURE_COLUMNS]
            probs_cache[id(sim)] = float(self.model.predict_proba(X)[0,1])
        cap = CFG["START_CAPITAL"]
        start_cap = cap
        highest = cap
        wins = 0; losses = 0; pnl_list=[]
        # Simple equity timeline for DD
        eq_points=[]
        base_risk=CFG["RISK_PER_TRADE"]
        for sim in sorted(self.sim_trades, key=lambda x:x.time_in):
            if sim.time_in <= train_until:
                # Training Trades: wir ignorieren für OOS Kennzahlen
                continue
            p = probs_cache.get(id(sim), 0.0)
            # Dynamischer Setup-spezifischer Offset
            setup = getattr(sim, 'setup', '').upper()
            if 'W3' in setup:
                thr_eff = thr + CFG.get('THR_OFFSET_W3',0.0)
            elif setup.startswith('C') or setup=='C':
                thr_eff = thr + CFG.get('THR_OFFSET_C',0.0)
            elif 'W5' in setup:
                thr_eff = thr + CFG.get('THR_OFFSET_W5',0.0)
            else:
                thr_eff = thr + CFG.get('THR_OFFSET_OTHER',0.0)
            if p < thr_eff:
                continue
            # Größe
            rps = max(sim.risk_per_share, 1e-9)
            size=(base_risk*cap)/rps
            if sim.direction=="SHORT":
                size*=CFG.get("SIZE_SHORT_FACTOR",1.0)
            if CFG.get("SIZE_BY_PROB", False):
                frac=max(0.0,(p-thr)/max(1e-6,1-thr))
                scale=CFG.get("PROB_SIZE_MIN",1.0)+(CFG.get("PROB_SIZE_MAX",1.0)-CFG.get("PROB_SIZE_MIN",1.0))*frac
                size*=scale
            size=int(max(1,size))
            trade_pnl = sim.per_share * size
            cap += trade_pnl
            pnl_list.append(trade_pnl)
            if trade_pnl>0: wins+=1
            else: losses+=1
            highest = max(highest, cap)
            dd = (cap / highest - 1.0) * 100.0
            eq_points.append(dd)
        trades = wins + losses
        if trades==0:
            return dict(trades=0, winrate=0.0, cagr=0.0, max_dd=0.0, total_return=0.0)
        winrate = wins / trades
        total_return = (cap/start_cap - 1.0) * 100.0
        # Approx Jahranzahl über OOS Zeitspanne
        oos_trades=[t for t in self.sim_trades if t.time_in>train_until]
        if oos_trades:
            years=max((oos_trades[-1].time_out - oos_trades[0].time_in).days/365.0, 1e-6)
        else:
            years=1.0
        cagr = ((cap/start_cap)**(1/years) - 1.0)*100.0 if years>0 else total_return
        max_dd = min(eq_points) if eq_points else 0.0
        return dict(trades=trades, winrate=winrate, cagr=cagr, max_dd=max_dd, total_return=total_return)

    def optimize_threshold_and_risk(self, train_until: pd.Timestamp):
        if self.model is None:
            return
        target_wr_min = CFG.get("TARGET_WINRATE_MIN", 0.70)
        target_wr_max = CFG.get("TARGET_WINRATE_MAX", 0.80)
        target_cagr_min = CFG.get("TARGET_CAGR_MIN", 90.0)
        target_max_dd = CFG.get("TARGET_MAX_DD", -5.0)  # drawdown ist negativ
        # Sammle alle OOS Wahrscheinlichkeiten
        probs=[]
        for sim in self.sim_trades:
            if sim.time_in <= train_until: continue
            X=pd.DataFrame([sim.features])[FEATURE_COLUMNS]
            probs.append(float(self.model.predict_proba(X)[0,1]))
        if len(probs)<30:
            return  # zu wenig für Optimierung
        qs = np.linspace(0.3,0.95,40)
        candidate_thresholds = [float(np.quantile(probs, q)) for q in qs]
        evals=[]
        for thr in candidate_thresholds:
            m = self._simulate_equity_for_threshold(thr, train_until)
            if not m: continue
            m['thr']=thr
            evals.append(m)
        if not evals:
            return
        # Filter nach harten Constraints (DD <=5% => max_dd >= -5) & Winrate Range
        feasible=[e for e in evals if (target_wr_min <= e['winrate'] <= target_wr_max) and (e['max_dd'] >= target_max_dd)]
        if feasible:
            # wähle max CAGR
            best = max(feasible, key=lambda x:(x['cagr'], x['trades']))
        else:
            # Penalize Abweichungen
            def penalty(e):
                # Winrate deviation
                if e['winrate'] < target_wr_min:
                    pw = (target_wr_min - e['winrate'])*5
                elif e['winrate'] > target_wr_max:
                    pw = (e['winrate'] - target_wr_max)*5
                else:
                    pw = 0
                # DD penalty (if beyond)
                pdd = 0 if e['max_dd'] >= target_max_dd else (target_max_dd - e['max_dd'])*2
                # CAGR penalty if below min
                pc = 0 if e['cagr'] >= target_cagr_min*0.4 else (target_cagr_min*0.4 - e['cagr'])/50.0
                return pw + pdd + pc
            best = min(evals, key=penalty)
        self.threshold = best['thr']
        # Risiko dynamisch skalieren falls Ziel-CAGR nicht erreicht aber DD Luft
        if CFG.get("AUTO_RISK_SCALE", True) and best['trades']>20:
            cur_cagr = best['cagr']
            cur_dd = best['max_dd']  # negativ
            if cur_cagr < target_cagr_min and cur_dd > -1e-9:  # kein Drawdown -> limit scale
                pass
            else:
                if cur_cagr < target_cagr_min and cur_dd > target_max_dd:  # dd milder als Grenze
                    # Skaliere basierend auf DD Verhältnis & CAGR Gap (lineare Approx)
                    dd_scale = abs(target_max_dd)/max(1e-6, abs(cur_dd)) if cur_dd!=0 else 1.0
                    cagr_scale = target_cagr_min/max(1e-6, cur_cagr)
                    scale = min(dd_scale, cagr_scale, 3.0)
                    if scale>1.05:  # nur wenn sinnvoll
                        CFG['RISK_PER_TRADE'] = min(CFG['RISK_PER_TRADE']*scale, 0.05)  # harte Obergrenze 5%
                        print(f"[OPT] Risk scale applied: x{scale:.2f} -> RISK_PER_TRADE={CFG['RISK_PER_TRADE']:.4f}")
        print(f"[OPT] Threshold gewählt: {self.threshold:.3f} (Winrate~{best['winrate']*100:.2f}%, CAGR~{best['cagr']:.1f}%, MaxDD {best['max_dd']:.2f}%)")

    def build_equity(self, train_until:pd.Timestamp):
        self.trades.clear(); self.equity.clear()
        cap=CFG["START_CAPITAL"]; eq_map:Dict[pd.Timestamp,float]={}

        # Test-Passrate prüfen/relaxen
        oos=[t for t in self.sim_trades if t.time_in>train_until]
        if self.model is not None and oos:
            Xo,_=self._XY(oos); probs=self.model.predict_proba(Xo)[:,1]
            raw_rate=float((probs>=self.threshold).mean())
            self.ml_test_pass_rate_raw=raw_rate
            if raw_rate < CFG["ML_MIN_PASS_RATE_TEST"]:
                thr_relaxed=float(np.quantile(probs, 1-CFG["ML_MIN_PASS_RATE_TEST"]))
                self.threshold=min(self.threshold, thr_relaxed)

        # --- Dynamische Risiko Hilfsfunktionen ---
        def _dd_percent(current_cap:float, highest_cap:float)->float:
            return (current_cap/highest_cap - 1.0)*100.0 if highest_cap>0 else 0.0
        def _risk_multiplier_for_dd(cur_dd:float)->float:
            if not CFG.get("DYNAMIC_DD_RISK", False):
                return 1.0
            steps=CFG.get("DD_RISK_STEPS", [])
            mult=1.0
            for thr,m in sorted(steps, key=lambda x:x[0]):  # thr ist negativ (z.B. -10)
                if cur_dd <= thr:
                    mult=m
            return mult
        trade_returns=[]  # für Vol-Zielsteuerung
        def _vol_adjustment()->float:
            if not CFG.get("USE_VOL_TARGET", False):
                return 1.0
            target=CFG.get("TARGET_ANNUAL_VOL", 0.25)
            window=CFG.get("VOL_WINDOW_TRADES", 40)
            if len(trade_returns)<5:
                return 1.0
            recent=trade_returns[-window:]
            # Approx annual vol: std(per-trade R) * sqrt(trades_per_year). Für Proxy nehmen wir std und normalisieren Richtung Ziel.
            s=np.std(recent, ddof=1)
            if s<=1e-9:
                return 1.0
            # Ziel: grob s -> target/ sqrt(annual_trades_factor). Vereinfachung: scale = target_ref / s
            # Da wir R-Multiples haben: angenommene annual trades ~ len(self.trades)/years später -> hier simpler scaler clampen.
            scale=target/ (s*4)  # heuristisch (4 ~ sqrt(approx trades/year Anteil))
            return max(0.4, min(1.6, scale))

        max_stop_dd=CFG.get("MAX_DRAWDOWN_STOP", -1e9)
        highest_global=cap

        def add(sim:SimTrade, prob:Optional[float]):
            nonlocal cap
            nonlocal highest_global
            if CFG.get("MAX_DRAWDOWN_STOP", -1e9) > -1e8:  # wurde gesetzt
                # Prüfe aktuellen Drawdown (gegen highest_global)
                cur_dd=_dd_percent(cap, highest_global)
                if cur_dd <= max_stop_dd:
                    return  # Trade verweigern – Hard Stop
            # Basis-Risiko
            base_risk=CFG["RISK_PER_TRADE"]
            # Drawdown Multiplikator
            cur_dd=_dd_percent(cap, highest_global)
            dd_mult=_risk_multiplier_for_dd(cur_dd)
            # Vol-Ziel Multiplikator
            vol_mult=_vol_adjustment()
            eff_risk=base_risk*dd_mult*vol_mult
            eff_risk=max(CFG.get("RISK_PER_TRADE_MIN", eff_risk), min(eff_risk, CFG.get("RISK_PER_TRADE_MAX", eff_risk)))
            size=(eff_risk*cap)/max(sim.risk_per_share,1e-9)
            if sim.direction=="SHORT": size *= CFG["SIZE_SHORT_FACTOR"]
            if prob is not None and CFG["SIZE_BY_PROB"]:
                base_thr = getattr(self,'threshold',0.5)
                frac=max(0.0,(prob-base_thr)/max(1e-6,1-base_thr))
                scale=CFG.get("PROB_SIZE_MIN",1.0)+(CFG.get("PROB_SIZE_MAX",1.0)-CFG.get("PROB_SIZE_MIN",1.0))*frac
                size*=scale
            
            # Volatility forecast-based position sizing (walk-forward, no lookahead)
            vola_mult = 1.0
            if self.vola_forecast_ready and self.vola_forecaster is not None:
                try:
                    # Find daily bar index for this trade's entry time
                    daily_mask = self.daily['date'] <= sim.time_in
                    if daily_mask.any():
                        daily_idx = daily_mask.sum() - 1  # last bar before/at entry
                        window = CFG.get('VOLA_FORECAST_WINDOW', 252)
                        if daily_idx >= window:
                            # Get volatility forecast and size multiplier
                            vola_result = self.vola_forecaster.forecast_volatility(
                                self.daily, daily_idx
                            )
                            if vola_result and 'size_multiplier' in vola_result:
                                vola_mult = vola_result['size_multiplier']
                                self.telemetry['vola_sized'] += 1
                except Exception as e:
                    if self.telemetry['vola_sized'] == 0:  # Only log first error
                        print(f"[VOLA] Forecast error: {e}")
            size *= vola_mult
            
            # Cap position size: max notional = 100% of current capital (no leverage beyond 1x)
            max_exposure_pct = CFG.get("MAX_POSITION_EXPOSURE", 1.0)  # 1.0 = 100% of account
            max_size = (cap * max_exposure_pct) / max(sim.entry, 1e-9)
            size = min(size, max_size)
            
            size=int(max(1,size))
            pnl = float(sim.per_share * size)
            
            # Apply trading costs (fee + slippage) - realistic execution costs
            # Fee: percentage of trade notional (entry_price × shares)
            # Slippage: additional execution cost as percentage of entry price
            # Typical values: fee=0.0001 (1 bps), slippage=0.0005 (5 bps)
            fee_rate = float(CFG.get('FEE', 0.0) or 0.0)
            slip_rate = float(CFG.get('SLIPPAGE', 0.0) or 0.0)
            trade_notional = float(sim.entry) * float(size)  # position value
            if fee_rate > 0:
                # Fee as % of notional, round-trip (entry + exit)
                fee_cost = trade_notional * fee_rate * 2
                pnl -= fee_cost
            if slip_rate > 0:
                # Slippage as % of entry price, applied to position
                slip_cost = trade_notional * slip_rate * 2  # round-trip
                pnl -= slip_cost
            
            # FTMO payout split: only a fraction of profits are retained, losses full
            split = float(CFG.get('FTMO_SPLIT', 1.0))
            if split < 0.9999 and pnl > 0:
                pnl = pnl * split
            cap += pnl
            # Speichere R-Multiple für Vol-Steuerung
            trade_returns.append(sim.per_share/max(sim.risk_per_share,1e-9))
            # Exit auf nächste 1H-Zeit mappen
            h1_after=self.h1[self.h1["date"]>=sim.time_out]
            map_time=h1_after["date"].iloc[0] if len(h1_after) else self.h1["date"].iloc[-1]
            eq_map[map_time]=cap
            rr=sim.per_share/max(sim.risk_per_share,1e-9)
            self.trades.append(Trade(sim.entry_tf,sim.entry_idx,sim.exit_idx,sim.entry,sim.exit,pnl,size,rr,
                                     sim.setup,sim.direction,sim.time_in,sim.time_out,sim.stop,sim.tp1,sim.tp2,sim.mae_r,sim.mfe_r))
            highest_global=max(highest_global, cap)

        pre_ml=len([t for t in self.sim_trades if t.time_in>train_until]); post_ml=0
        for sim in self.sim_trades:
            if sim.time_in<=train_until:
                add(sim, None)
            else:
                if self.model is None:
                    add(sim, None)
                else:
                    X=pd.DataFrame([sim.features])[FEATURE_COLUMNS]
                    p=float(self.model.predict_proba(X)[0,1])
                    if p>=self.threshold:
                        add(sim, p); post_ml+=1
        self.ml_test_pass_rate = (post_ml/max(1,pre_ml)) if pre_ml>0 else None

        highest=CFG["START_CAPITAL"]; cur=CFG["START_CAPITAL"]
        for ts in self.h1["date"]:
            if ts in eq_map: cur=eq_map[ts]
            highest=max(highest, cur)
            dd=(cur - highest)/max(highest,1e-9)*100
            self.equity.append(dict(date=ts, capital=cur, dd=dd))

    def run(self)->Dict:
        self.analyze_structure()
        self.build_setups()
        self.simulate_all()
        if not self.sim_trades: return {}
        times=sorted([t.time_in for t in self.sim_trades])
        split_idx=max(1, int(len(times)*CFG["TRAIN_FRAC"]))
        # Purging gap: skip trades between train and test to avoid leakage
        purge_bars = CFG.get('TRAIN_TEST_PURGE_BARS', 0)
        if purge_bars > 0 and split_idx + purge_bars < len(times):
            train_until = times[split_idx - 1]
            # Adjust OOS start to skip purge window
            oos_start_idx = split_idx + purge_bars
            self._purged_oos_start = times[oos_start_idx] if oos_start_idx < len(times) else train_until
        else:
            train_until = times[split_idx - 1]
            self._purged_oos_start = train_until
        # Speichern für spätere Re-Simulationen (Deep Rerun)
        self.train_until = train_until

        if CFG["USE_ML"]:
            # Apply purging: training trades before purge window only
            purge_start = getattr(self, '_purged_oos_start', train_until)
            train=[t for t in self.sim_trades if t.time_in<=train_until]
            if len(train)>=20:
                self.train_model(train)
                # In neutral mode, use fixed threshold from training (no OOS optimization)
                if CFG.get('NEUTRAL_MODE', False):
                    print(f"[NEUTRAL] Fester ML-Threshold: {self.threshold:.3f} (aus Training)")
                elif CFG.get("OPTIMIZE_ML_THRESHOLD", False) and self.model is not None:
                    try:
                        val=[t for t in self.sim_trades if t.time_in>train_until]
                        if len(val)>=25:
                            Xv,_=self._XY(val); probs=self.model.predict_proba(Xv)[:,1]
                            labels=np.array([t.label for t in val])
                            qs=np.linspace(0.2,0.9,15)
                            best_thr=self.threshold; best_score=-1e9
                            for q in qs:
                                thr=np.quantile(probs,q)
                                mask=probs>=thr
                                if mask.sum()<5: continue
                                sel=labels[mask]
                                win_rate=sel.mean() if len(sel)>0 else 0
                                score=win_rate - 0.5*(1-win_rate)
                                if score>best_score:
                                    best_score=score; best_thr=thr
                            self.threshold=float(best_thr)
                    except Exception:
                        pass
                    try:
                        self.optimize_threshold_and_risk(train_until)
                    except Exception as e:
                        print(f"[WARN] Ziel-Optimierung fehlgeschlagen: {e}")
            else:
                self.model=None; self.threshold=0.5; self.ml_train_pass_rate=None
        else:
            self.model=None; self.threshold=0.5; self.ml_train_pass_rate=None

        self.build_equity(train_until)
        # Nach Equity-Aufbau: Wahrscheinlichkeiten für alle OOS-Trades berechnen & speichern (falls Modell vorhanden)
        if self.model is not None:
            for sim in self.sim_trades:
                if sim.time_in <= self.train_until:
                    # Trainingsanteil: optional nicht speichern, lassen None
                    continue
                if sim.prob is None:
                    try:
                        X=pd.DataFrame([sim.features])[FEATURE_COLUMNS]
                        sim.prob = float(self.model.predict_proba(X)[0,1])
                    except Exception:
                        sim.prob = None
        metrics=self.metrics()
        print("\n--- Telemetrie ---")
        print(f"Setups gesamt: {self.telemetry['setups']} | akzeptiert bis Entry: {self.telemetry['accepted']}")
        print(f"Filter: daily={self.telemetry.get('filtered_daily',0)}, regime(ADX)={self.telemetry.get('filtered_regime',0)}, vola_regime={self.telemetry.get('filtered_vola_regime',0)}, ema={self.telemetry['filtered_ema']}, vol={self.telemetry['filtered_vol']}, momentum={self.telemetry.get('filtered_momentum',0)}, pf={self.telemetry.get('filtered_pf',0)}, no_touch={self.telemetry['no_touch']}, no_confirm={self.telemetry['no_confirm']}")
        if self.telemetry.get('vola_sized', 0) > 0:
            print(f"[VOLA-SIZING] {self.telemetry['vola_sized']} Trades mit Vola-Forecast angepasst")
        if CFG["USE_ML"]:
            print(f"ML threshold: {self.threshold:.3f} | Train pass-rate: {self.ml_train_pass_rate} | Test pass-rate used: {self.ml_test_pass_rate}")
        
        # Regime Analysis (by ADX levels)
        if CFG.get("ANALYZE_REGIMES", False):
            self._analyze_regimes()
        
        return metrics

    # ---------- Regime Analysis ----------
    def _analyze_regimes(self):
        """Analyze performance by VOLATILITY regime (ATR-based)"""
        if not self.trades:
            print("\n[REGIME] Keine Trades verfügbar")
            return
        
        # Calculate ATR percentile at entry for each trade
        # Use daily ATR_PCT for regime classification
        if "ATR_PCT" not in self.daily.columns:
            print("\n[REGIME] Keine ATR-Daten verfügbar")
            return
        
        # Calculate rolling ATR percentile (where does current ATR stand vs last 60 days)
        atr_series = self.daily["ATR_PCT"].dropna()
        
        # Bucket trades by volatility regime
        regimes = {
            "low_vola (<25%)": [],      # Bottom quartile - stagnation
            "mid_vola (25-75%)": [],    # Middle 50%
            "high_vola (>75%)": [],     # Top quartile - high volatility
        }
        
        for t in self.trades:
            # Find ATR at trade entry
            entry_date = pd.Timestamp(t.time_in).normalize()
            didx = self.daily[self.daily["date"] <= entry_date].index
            if len(didx) == 0:
                continue
            
            current_atr = float(self.daily.loc[didx.max(), "ATR_PCT"])
            if np.isnan(current_atr):
                continue
            
            # Calculate percentile vs last 60 days
            lookback = 60
            start_idx = max(0, didx.max() - lookback)
            historical_atr = atr_series.iloc[start_idx:didx.max()+1]
            if len(historical_atr) < 10:
                continue
            
            percentile = (historical_atr < current_atr).sum() / len(historical_atr) * 100
            
            if percentile < 25:
                regimes["low_vola (<25%)"].append(t)
            elif percentile < 75:
                regimes["mid_vola (25-75%)"].append(t)
            else:
                regimes["high_vola (>75%)"].append(t)
        
        print("\n--- Regime-Analyse (nach Volatilität) ---")
        print(f"{'Regime':<20} {'Trades':>8} {'Winrate':>10} {'Avg PnL':>12} {'Total PnL':>14} {'PF':>8}")
        print("-" * 75)
        
        for regime, trades in regimes.items():
            if not trades:
                print(f"{regime:<20} {'---':>8}")
                continue
            
            pnls = [t.pnl for t in trades]
            wins = [p for p in pnls if p > 0]
            losses = [p for p in pnls if p <= 0]
            
            n = len(trades)
            winrate = len(wins) / n * 100 if n else 0
            avg_pnl = np.mean(pnls) if pnls else 0
            total_pnl = sum(pnls)
            pf = sum(wins) / abs(sum(losses)) if losses and sum(losses) != 0 else float('inf')
            
            print(f"{regime:<20} {n:>8} {winrate:>9.1f}% ${avg_pnl:>10.2f} ${total_pnl:>12.2f} {pf:>8.2f}")
        
        print("-" * 75)
        
        # Recommendation
        low_trades = regimes["low_vola (<25%)"]
        high_trades = regimes["high_vola (>75%)"]
        if low_trades and high_trades:
            low_pnls = [t.pnl for t in low_trades]
            low_avg = np.mean(low_pnls)
            high_pnls = [t.pnl for t in high_trades]
            high_avg = np.mean(high_pnls)
            
            if low_avg < 0 and high_avg > 0:
                print(f"[EMPFEHLUNG] System performt besser bei hoher Volatilität. Filtere Low-Vola Phasen!")
            elif high_avg < low_avg:
                print(f"[INFO] System performt überraschenderweise besser bei niedriger Vola.")
            else:
                print(f"[INFO] Beide Regimes profitabel - Moderate Unterschiede.")

    # ---------- Metrics ----------
    def metrics(self)->Dict:
        if not self.equity: return {}
        pnl=[t.pnl for t in self.trades]
        wins=[x for x in pnl if x>0]; losses=[x for x in pnl if x<=0]
        start=CFG["START_CAPITAL"]; end=self.equity[-1]["capital"]
        total_return=(end-start)/start*100 if start>0 else 0.0

        eq=pd.DataFrame(self.equity)
        eq["date"]=pd.to_datetime(eq["date"])
        
        # ========== KORRIGIERTE SHARPE/SORTINO BERECHNUNG ==========
        # Problem: Equity wird nur bei Trades aktualisiert, nicht täglich
        # Lösung: Berechne Volatilität nur aus Tagen MIT Trades, aber annualisiere korrekt
        
        # Methode 1: Tägliche Returns (mit Forward-Fill für korrekte Zeitbasis)
        eq_daily = eq.set_index("date").resample("D").last().ffill()
        eq_daily = eq_daily.dropna(subset=["capital"])
        
        # Log-Renditen berechnen
        eq_daily["log_ret"] = np.log(eq_daily["capital"] / eq_daily["capital"].shift(1))
        eq_daily["log_ret"] = eq_daily["log_ret"].fillna(0.0)
        log_rets_all = eq_daily["log_ret"].values
        
        # Nur Tage mit tatsächlichen Änderungen (Trades) für Volatilitätsberechnung
        # Dies ist realistischer, da die Strategie nicht täglich exponiert ist
        log_rets_nonzero = log_rets_all[np.abs(log_rets_all) > 1e-10]
        
        # Annualisierungsfaktor
        trading_days_per_year = 252
        actual_days = (eq_daily.index[-1] - eq_daily.index[0]).days
        years_actual = actual_days / 365.0 if actual_days > 0 else 1.0
        
        # Exposure: Anteil der Zeit im Markt
        trades_per_year = len(self.trades) / years_actual if years_actual > 0 else 0
        exposure_days = len(log_rets_nonzero)
        exposure_ratio = exposure_days / len(log_rets_all) if len(log_rets_all) > 0 else 0
        
        # Tägliche Statistiken (aus Tagen MIT Trades)
        mu_trade_day = float(np.mean(log_rets_nonzero)) if len(log_rets_nonzero) > 1 else 0.0
        sigma_trade_day = float(np.std(log_rets_nonzero, ddof=1)) if len(log_rets_nonzero) > 1 else 0.0
        
        # Annualisierte Volatilität
        # Skaliere mit sqrt(trades_per_year) für realistischere Vol
        # Alternativ: sqrt(exposure_days / years_actual) für Trade-Frequenz-adjustiert
        annualization_factor = math.sqrt(len(log_rets_nonzero) / years_actual) if years_actual > 0 else 1.0
        vol_annual = sigma_trade_day * annualization_factor
        vol_pct = vol_annual * 100.0
        
        # Risk-free rate
        rf_ann = RISK_FREE_RATE / 100.0
        rf_daily = rf_ann / trading_days_per_year
        mu_annual = mu_trade_day * (len(log_rets_nonzero) / years_actual) if years_actual > 0 else 0.0
        
        # CAGR für Sharpe (aus tatsächlicher Equity-Entwicklung)
        if end > 0 and start > 0 and years_actual > 0:
            cagr_for_sharpe = (end / start) ** (1 / years_actual) - 1
        else:
            cagr_for_sharpe = 0.0
        
        # Sharpe Ratio
        sharpe = (cagr_for_sharpe - rf_ann) / vol_annual if vol_annual > 1e-12 else 0.0
        
        # Sortino Ratio (nur Downside-Volatilität)
        downside_rets = np.minimum(log_rets_nonzero, 0.0)
        down_sigma_trade = float(np.std(downside_rets, ddof=1)) if len(downside_rets) > 1 and np.any(downside_rets < 0) else 0.0
        down_sigma_annual = down_sigma_trade * annualization_factor
        sortino = (cagr_for_sharpe - rf_ann) / down_sigma_annual if down_sigma_annual > 1e-12 else 0.0
        
        # Debug: Zeige Sharpe-Komponenten bei erster Berechnung
        if not hasattr(self, '_sharpe_debug_shown'):
            self._sharpe_debug_shown = True
            print(f"[SHARPE] Years: {years_actual:.2f} | Trade-days: {len(log_rets_nonzero)} | Exposure: {exposure_ratio*100:.1f}%")
            print(f"[SHARPE] sigma_per_trade: {sigma_trade_day*100:.3f}% | vol_annual: {vol_pct:.2f}%")
            print(f"[SHARPE] CAGR: {cagr_for_sharpe*100:.2f}% | Sharpe: {sharpe:.2f} | Sortino: {sortino:.2f}")
        
        # Legacy: einfache Renditen für Kompatibilität
        eq["ret"] = eq["capital"].pct_change().fillna(0.0)
        rets = eq["ret"].values
        span_years = max((eq["date"].iloc[-1] - eq["date"].iloc[0]).days / 365.0, 1e-6)
        periods_per_year = len(rets) / span_years if span_years > 0 else 0.0
        rf_per_period = rf_daily  # für Kompatibilität mit altem Code

        years = max((pd.Timestamp(self.equity[-1]["date"]) - pd.Timestamp(self.equity[0]["date"])).days / 365.0, 1e-6)
        try:
            ratio = end/start if start > 0 else 0.0
            # Avoid complex numbers from negative base exponentiation
            if ratio <= 0:
                cagr = -100.0  # Total loss or worse
            else:
                cagr = (ratio**(1/years)-1)*100
            # Sanity check
            if not isinstance(cagr, (int, float)) or not np.isfinite(cagr) or cagr > 1e6 or cagr < -1e6:
                cagr = 0.0
        except (OverflowError, ZeroDivisionError, ValueError, TypeError):
            cagr = 0.0

        max_dd=min(e["dd"] for e in self.equity) if self.equity else 0.0  # in %
        dd_dec=np.array([-min(0.0, e["dd"]/100.0) for e in self.equity], dtype=float)  # positive Tiefen
        ulcer_index=math.sqrt(float(np.mean(dd_dec**2))) if len(dd_dec)>0 else 0.0
        upi=((cagr/100.0)-rf_ann)/ulcer_index if ulcer_index>1e-12 else 0.0

        pos_sum=float(np.sum(rets[rets>0])) if np.any(rets>0) else 0.0
        neg_sum=float(np.sum(rets[rets<0])) if np.any(rets<0) else 0.0
        gain_to_pain = (pos_sum/abs(neg_sum)) if neg_sum!=0 else np.inf

        total_hours = (eq["date"].iloc[-1]-eq["date"].iloc[0]).total_seconds()/3600.0
        held_hours = float(np.sum([(t.time_out - t.time_in).total_seconds()/3600.0 for t in self.trades]))
        exposure = min(1.0, held_hours/max(total_hours,1e-6)) if total_hours>0 else 0.0

        hit=len(wins)/len(pnl)*100 if pnl else 0.0
        avg_win=float(np.mean(wins)) if wins else 0.0
        avg_loss=float(np.mean(losses)) if losses else 0.0
        payoff=abs(avg_win)/abs(avg_loss) if avg_loss!=0 else np.nan
        profit_factor=(sum(wins)/abs(sum(losses))) if losses else np.inf
        expectancy=(hit/100.0)*avg_win + (1-hit/100.0)*avg_loss

        by_type:Dict[str,List[float]]={}; by_dir={"LONG":[], "SHORT":[]}
        for t in self.trades:
            by_type.setdefault(t.setup,[]).append(t.pnl)
            by_dir[t.direction].append(t.pnl)
        type_stats={k:dict(count=len(v),sum=float(np.sum(v)),avg=float(np.mean(v)) if v else 0.0) for k,v in by_type.items()}
        dir_stats={k:dict(count=len(v),sum=float(np.sum(v)),avg=float(np.mean(v)) if v else 0.0) for k,v in by_dir.items()}
        durations=[(t.time_out - t.time_in).total_seconds()/3600.0 for t in self.trades]
        mae=[t.mae_r for t in self.trades]; mfe=[t.mfe_r for t in self.trades]

        long_pnls=by_dir.get("LONG",[]); short_pnls=by_dir.get("SHORT",[])
        long_wins=[x for x in long_pnls if x>0]; long_losses=[x for x in long_pnls if x<=0]
        short_wins=[x for x in short_pnls if x>0]; short_losses=[x for x in short_pnls if x<=0]
        winrate_long=len(long_wins)/len(long_pnls)*100 if long_pnls else 0.0
        winrate_short=len(short_wins)/len(short_pnls)*100 if short_pnls else 0.0
        pf_long=(sum(long_wins)/abs(sum(long_losses))) if long_losses else (np.inf if long_wins else 0.0)
        pf_short=(sum(short_wins)/abs(sum(short_losses))) if short_losses else (np.inf if short_wins else 0.0)

        R_list=[t.rr for t in self.trades]
        expectancy_R=float(np.mean(R_list)) if R_list else 0.0
        avg_R_win=float(np.mean([r for r,p in zip(R_list,pnl) if p>0])) if any(p>0 for p in pnl) else 0.0
        avg_R_loss=abs(float(np.mean([r for r,p in zip(R_list,pnl) if p<=0]))) if any(p<=0 for p in pnl) else 0.0
        p_win=len(wins)/len(pnl) if pnl else 0.0
        kelly = (p_win - (1-p_win)/max(avg_R_win/avg_R_loss,1e-9)) if avg_R_loss>0 and avg_R_win>0 else 0.0
        if not np.isfinite(kelly): kelly=0.0
        kelly_pct=kelly*100.0

        avg_mae_r=float(np.mean(mae)) if mae else 0.0
        avg_mfe_r=float(np.mean(mfe)) if mfe else 0.0
        median_mae_r=float(np.median(mae)) if mae else 0.0
        median_mfe_r=float(np.median(mfe)) if mfe else 0.0
        best_trade=float(max(pnl)) if pnl else 0.0
        worst_trade=float(min(pnl)) if pnl else 0.0
        avg_size=float(np.mean([t.size for t in self.trades])) if self.trades else 0.0
        pnl_std=float(np.std(pnl, ddof=1)) if len(pnl)>1 else 0.0
        downside_deviation=float(np.std([x for x in pnl if x<0], ddof=1)) if len([x for x in pnl if x<0])>1 else 0.0
        equity_curve_len=len(self.equity)

        signs=[1 if x>0 else -1 for x in pnl]
        streaks=[]; cur=0; prev=None
        for s in signs:
            if prev is None or s==prev: cur+=1
            else: streaks.append(prev*cur); cur=1
            prev=s
        if prev is not None: streaks.append(prev*cur)
        max_win_streak=max([x for x in streaks if x>0], default=0)
        max_loss_streak=max([abs(x) for x in streaks if x<0], default=0)

        eq["month"]=eq["date"].dt.to_period("M")
        monthly=(eq.groupby("month")["ret"].apply(lambda s:(1+s).prod()-1)).reset_index()
        monthly["ret_pct"]=monthly["ret"]*100.0
        monthly["year"]=monthly["month"].dt.year
        monthly["m"]=monthly["month"].dt.month
        heatmap = monthly.pivot_table(index="year", columns="m", values="ret_pct", aggfunc="first").sort_index()

        trades_per_year = len(self.trades)/years if years>0 else 0.0

        out=dict(
            total_return=total_return,cagr=cagr,trades=len(self.trades),hit=hit,avg_win=avg_win,avg_loss=avg_loss,
            payoff=payoff,profit_factor=profit_factor,expectancy=expectancy,
            vol=vol_pct,sharpe=sharpe,sortino=sortino,max_dd=max_dd,calmar=(cagr/abs(max_dd) if max_dd<0 else 0.0),
            pnl=pnl,returns_r=[t.rr for t in self.trades],equity=self.equity,trades_list=self.trades,
            type_stats=type_stats,dir_stats=dir_stats,durations=durations,mae=mae,mfe=mfe,monthly=monthly,
            periods_per_year=periods_per_year, rf_per_period=rf_per_period, ulcer_index=ulcer_index, upi=upi,
            gain_to_pain=gain_to_pain, trades_per_year=trades_per_year, exposure=exposure,
            max_win_streak=max_win_streak, max_loss_streak=max_loss_streak, avg_R=float(np.mean([t.rr for t in self.trades])) if self.trades else 0.0,
            median_hold_hours=float(np.median(durations)) if durations else 0.0, avg_hold_hours=float(np.mean(durations)) if durations else 0.0,
            heatmap=heatmap
        )

        out.update(dict(
            winrate_long=winrate_long, winrate_short=winrate_short,
            pf_long=pf_long, pf_short=pf_short,
            expectancy_R=expectancy_R,
            avg_R_win=avg_R_win, avg_R_loss=avg_R_loss,
            kelly_pct=kelly_pct,
            avg_mae_r=avg_mae_r, avg_mfe_r=avg_mfe_r,
            median_mae_r=median_mae_r, median_mfe_r=median_mfe_r,
            best_trade=best_trade, worst_trade=worst_trade,
            avg_size=avg_size, pnl_std=pnl_std, downside_deviation=downside_deviation,
            equity_curve_len=equity_curve_len
        ))

        if self.model is not None and self.sim_trades:
            times=sorted([t.time_in for t in self.sim_trades]); split_idx=max(1,int(len(times)*CFG["TRAIN_FRAC"])); split_time=times[split_idx-1]
            te=[t for t in self.sim_trades if t.time_in>split_time]
            if len(te)>=5:
                Xte,_=self._XY(te); yte=np.array([t.label for t in te],dtype=int)
                prob_te=self.model.predict_proba(Xte)[:,1]
                fpr,tpr,_=roc_curve(yte,prob_te); rc,pr,_=precision_recall_curve(yte,prob_te)
                out["ml_auc"]=auc(fpr,tpr); out["ml_ap"]=average_precision_score(yte,prob_te)
                pt,pp=calibration_curve(yte,prob_te, n_bins=min(10,len(yte)))
                out["ml_calib"]=(pt,pp); out["ml_roc"]=(fpr,tpr); out["ml_pr"]=(rc,pr)
                try:
                    pi=permutation_importance(self.model,Xte,yte,n_repeats=10,random_state=42)
                    out["ml_perm_importance"]=(list(Xte.columns), list(pi.importances_mean))
                except Exception:
                    pass
                out["ml_threshold"]=self.threshold
        return out

    # -----------------------------------------------------
    # Rebuild Trades + Equity mit angewandtem Base-Shift & Offsets
    # Verwendet gespeicherte sim_trades (OOS) und deren Wahrscheinlichkeiten
    # -----------------------------------------------------
    def rebuild_with_shift_offsets(self, base_shift:float, off_w3:float, off_c:float, off_w5:float, off_o:float, commit:bool=False)->Optional[Dict]:
        # erweitert: commit=True ersetzt aktuelle Trades/Equity durch gefilterte OOS Trades
        if not getattr(self,'sim_trades', None) or self.model is None:
            return None
        # OOS Wahrscheinlichkeiten sicherstellen
        for sim in self.sim_trades:
            if sim.time_in <= getattr(self,'train_until', pd.Timestamp.min):
                continue
            if sim.prob is None:
                try:
                    X=pd.DataFrame([sim.features])[FEATURE_COLUMNS]
                    sim.prob=float(self.model.predict_proba(X)[0,1])
                except Exception:
                    sim.prob=None
        base_thr=getattr(self,'threshold',0.5)
        thr_shifted=max(0.0, min(0.99, base_thr + base_shift))
        kept=[]; cap=CFG['START_CAPITAL']; highest=cap; wins=losses=0; dd_list=[]; pnl_list=[]; rebuilt_trades=[]
        # Trainings Equity rekonstruieren (alle train trades unverändert mit ursprünglicher Sizing Logik)
        # Wir benutzen vorhandene self.trades für train nicht, bauen sie neu minimal nach (ohne vollständige Rekonstruktion -> nur Startkapital lineare Aggregation)
        # Vereinfachung: cap nach Train = Kapital zum Split aus existierender Equity falls vorhanden
        try:
            if self.equity:
                split_cap = next((e['capital'] for e in self.equity if pd.Timestamp(e['date'])>= self.train_until), self.equity[0]['capital'])
                cap=float(split_cap)
                highest=cap
        except Exception:
            pass
        for sim in sorted(self.sim_trades, key=lambda x:x.time_in):
            if sim.time_in <= getattr(self,'train_until', pd.Timestamp.min):
                continue
            p=sim.prob
            if p is None: continue
            setup=getattr(sim,'setup','').upper()
            if 'W3' in setup:
                thr_eff=thr_shifted + off_w3
            elif setup.startswith('C') or setup=='C':
                thr_eff=thr_shifted + off_c
            elif 'W5' in setup:
                thr_eff=thr_shifted + off_w5
            else:
                thr_eff=thr_shifted + off_o
            thr_eff=max(0.0, min(0.995, thr_eff))
            if p < thr_eff: continue
            kept.append(sim)
            rps=max(sim.risk_per_share,1e-9)
            size=(CFG['RISK_PER_TRADE']*cap)/rps
            if sim.direction=='SHORT':
                size*=CFG.get('SIZE_SHORT_FACTOR',1.0)
            if CFG.get('SIZE_BY_PROB', False):
                frac=max(0.0,(p - thr_eff)/max(1e-6,1-thr_eff))
                scale=CFG.get('PROB_SIZE_MIN',1.0)+(CFG.get('PROB_SIZE_MAX',1.0)-CFG.get('PROB_SIZE_MIN',1.0))*frac
                size*=scale
            size=int(max(1,size))
            trade_pnl=sim.per_share*size
            split=float(CFG.get('FTMO_SPLIT',1.0))
            if split < 0.9999 and trade_pnl>0:
                trade_pnl*=split
            cap+=trade_pnl
            pnl_list.append(trade_pnl)
            if trade_pnl>0: wins+=1
            else: losses+=1
            highest=max(highest,cap)
            dd_list.append((cap/highest -1.0)*100.0)
            if commit:
                rebuilt_trades.append(Trade(sim.entry_tf,sim.entry_idx,sim.exit_idx,sim.entry,sim.exit,trade_pnl,size,sim.per_share/max(sim.risk_per_share,1e-9),
                                            sim.setup,sim.direction,sim.time_in,sim.time_out,sim.stop,sim.tp1,sim.tp2,sim.mae_r,sim.mfe_r, prob=p, risk_per_share=sim.risk_per_share))
        trades=len(kept)
        if trades==0:
            return dict(trades=0, winrate=0.0, total_ret=0.0, cagr=0.0, max_dd=0.0, pf=0.0)
        start_cap=CFG['START_CAPITAL']
        total_ret=(cap/start_cap -1.0)*100.0
        yrs=max((kept[-1].time_out - kept[0].time_in).days/365.0,1e-6)
        cagr=((cap/start_cap)**(1/yrs)-1.0)*100.0 if yrs>0 else total_ret
        max_dd=min(dd_list) if dd_list else 0.0
        gross=sum(p for p in pnl_list if p>0); loss=sum(p for p in pnl_list if p<=0)
        pf=(gross/abs(loss)) if loss<0 else float('inf')
        metrics=dict(trades=trades, winrate=wins/max(1,trades), total_ret=total_ret, cagr=cagr, max_dd=max_dd, pf=pf)
        if commit:
            # Ersetze nur OOS Trades + Equity ab train_until, behalte Train-Equity unverändert
            self.trades=[t for t in self.trades if t.time_in <= self.train_until] + rebuilt_trades
            # Equity Neuaufbau ab train_until (vereinfacht stützpunkte anhand OOS trade exits)
            # Entferne Equity Punkte nach train_until
            self.equity=[e for e in self.equity if pd.Timestamp(e['date'])<= self.train_until]
            current_cap=self.equity[-1]['capital'] if self.equity else start_cap
            highest_cap=current_cap
            for rt in rebuilt_trades:
                current_cap+=rt.pnl
                highest_cap=max(highest_cap,current_cap)
                self.equity.append(dict(date=rt.time_out, capital=current_cap, dd=(current_cap/highest_cap-1)*100.0))
        return metrics

# --------------------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------------------
def plot_report(daily:pd.DataFrame, h1:pd.DataFrame, bt:Backtester, metrics:Dict, pdf_path:str):
    theme = CFG.get("REPORT_THEME","light")
    bg = "#111418" if theme=="dark" else "#ffffff"
    fg = "#f2f2f2" if theme=="dark" else "#222222"
    grid_c = "#333" if theme=="dark" else "#cccccc"
    accent = "#ff5252"
    plt.rcParams.update({
        'axes.facecolor': bg,
        'figure.facecolor': bg,
        'text.color': fg,
        'axes.labelcolor': fg,
        'axes.edgecolor': fg,
        'xtick.color': fg,
        'ytick.color': fg,
        'grid.color': grid_c,
        'font.size': 10
    })
    eq = metrics['equity']
    dates=[pd.Timestamp(e['date']) for e in eq]
    caps=[e['capital'] for e in eq]
    dds=[e['dd'] for e in eq]
    with PdfPages(pdf_path) as pdf:
        # Seite 1: Equity & Drawdown getrennt + kumulierte Returns
        fig = plt.figure(figsize=(15,9), constrained_layout=True)
        gs = fig.add_gridspec(3,1, height_ratios=[3,1,1])
        ax_eq = fig.add_subplot(gs[0,0])
        ax_dd = fig.add_subplot(gs[1,0])
        ax_ret = fig.add_subplot(gs[2,0])
        ax_eq.plot(dates, caps, color=accent, lw=1.5, label=f"Final: ${int(caps[-1]):,}")
        ax_eq.axhline(CFG['START_CAPITAL'], ls='--', color='#888', lw=0.9, label='Initial')
        if CFG.get('REPORT_EQUITY_LOG') or ( (max(caps)-min(caps))/max(min(caps),1)>CFG.get('EQUITY_LOG_THRESHOLD',5.0)):
            ax_eq.set_yscale('log')
        ax_eq.set_title('Equity Curve', color=fg)
        ax_eq.legend(facecolor=bg, edgecolor=fg)
        ax_eq.grid(alpha=0.25)
        # Drawdown
        ax_dd.fill_between(dates, dds, 0, color=accent, alpha=0.35)
        ax_dd.plot(dates, dds, color=accent, lw=0.8)
        ax_dd.set_ylim(min(dds)*1.1, 2)
        ax_dd.set_title('Drawdown (%)', color=fg)
        ax_dd.grid(alpha=0.25)
        # Cumulative returns (%)
        eq_df=pd.DataFrame(eq)
        eq_df['ret'] = eq_df['capital'].pct_change().fillna(0)
        eq_df['cum_ret_pct'] = (1+eq_df['ret']).cumprod()-1
        ax_ret.plot(dates, eq_df['cum_ret_pct']*100, color='#29b6f6', lw=1.2)
        ax_ret.set_title('Kumulative Rendite (%)', color=fg)
        ax_ret.grid(alpha=0.25)
        for ax in (ax_eq, ax_dd, ax_ret):
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=35, ha='right')
        pdf.savefig(fig); plt.close(fig)

        # Seite 2: KPI Tabelle mit mehr Spalten
        from math import ceil
        all_rows=[]
        def add_row(k,v):
            all_rows.append((k,str(v)))
        # Fallback falls Keys fehlen (z.B. bei älteren Metric-Versionen)
        if 'total_return' not in metrics or 'cagr' not in metrics:
            try:
                eq_cap=[e['capital'] for e in metrics.get('equity',[])]
                if eq_cap:
                    start_cap=CFG.get('START_CAPITAL', eq_cap[0])
                    tr=(eq_cap[-1]/start_cap-1)*100 if start_cap>0 else 0.0
                    add_row('Total Return', f"{tr:.2f}%")
                else:
                    add_row('Total Return', 'n/a')
                # CAGR Fallback
                eq_dates=[pd.Timestamp(e['date']) for e in metrics.get('equity',[])]
                if eq_cap and eq_dates:
                    years=max((eq_dates[-1]-eq_dates[0]).days/365.0,1e-6)
                    cagr_fb=((eq_cap[-1]/eq_cap[0])**(1/years)-1)*100 if eq_cap[0]>0 else 0.0
                    add_row('CAGR', f"{cagr_fb:.2f}%")
                else:
                    add_row('CAGR','n/a')
            except Exception:
                add_row('Total Return','n/a')
                add_row('CAGR','n/a')
        else:
            add_row('Total Return', f"{metrics['total_return']:.2f}%")
            add_row('CAGR', f"{metrics['cagr']:.2f}%")
        add_row('Winrate', f"{metrics['hit']:.2f}%")
        add_row('Profit Factor', f"{metrics['profit_factor']:.2f}")
        add_row('Expectancy $', f"{metrics['expectancy']:.2f}")
        add_row('Sharpe', f"{metrics['sharpe']:.2f}")
        add_row('Sortino', f"{metrics['sortino']:.2f}")
        add_row('UPI', f"{metrics['upi']:.2f}")
        add_row('UlcerIdx', f"{metrics['ulcer_index']:.2f}")
        add_row('Gain/Pain', f"{metrics['gain_to_pain']:.2f}")
        add_row('Max DD', f"{metrics['max_dd']:.2f}%")
        add_row('Calmar', f"{metrics['calmar']:.2f}")
        add_row('Ann.Vol', f"{metrics['vol']:.2f}%")
        add_row('Trades', metrics['trades'])
        add_row('Avg Win', f"{metrics['avg_win']:.0f}")
        add_row('Avg Loss', f"{metrics['avg_loss']:.0f}")
        add_row('Payoff', f"{metrics['payoff']:.2f}")
        add_row('Expect R', f"{metrics.get('expectancy_R',0):.2f}")
        add_row('Kelly %', f"{metrics.get('kelly_pct',0):.2f}%")
        add_row('Win Long', f"{metrics.get('winrate_long',0):.2f}%")
        add_row('Win Short', f"{metrics.get('winrate_short',0):.2f}%")
        add_row('PF Long', f"{metrics.get('pf_long',0):.2f}")
        add_row('PF Short', f"{metrics.get('pf_short',0):.2f}")
        add_row('Avg Hold h', f"{metrics.get('avg_hold_hours',0):.1f}")
        add_row('Median Hold h', f"{metrics.get('median_hold_hours',0):.1f}")
        add_row('MAE R', f"{metrics.get('avg_mae_r',0):.2f}")
        add_row('MFE R', f"{metrics.get('avg_mfe_r',0):.2f}")
        add_row('Avg Size', f"{metrics.get('avg_size',0):.1f}")
        add_row('Best $', f"{metrics.get('best_trade',0):.0f}")
        add_row('Worst $', f"{metrics.get('worst_trade',0):.0f}")
        add_row('Max Win Streak', metrics.get('max_win_streak',0))
        add_row('Max Loss Streak', metrics.get('max_loss_streak',0))
        cols=CFG.get('REPORT_MAX_KPI_COLUMNS',2)
        rows=len(all_rows)
        per_col=ceil(rows/cols)
        fig,axes=plt.subplots(1,cols, figsize=(16, 0.45*per_col+1), constrained_layout=True)
        if cols==1: axes=[axes]
        for ci in range(cols):
            subset=all_rows[ci*per_col:(ci+1)*per_col]
            ax=axes[ci]
            ax.axis('off')
            y=1.0
            fs=CFG.get('REPORT_TABLE_FONT_SIZE',11)
            for k,v in subset:
                ax.text(0.02,y,f"{k}", color=fg, fontsize=fs, ha='left', va='top', fontweight='semibold')
                ax.text(0.55,y,f"{v}", color=accent, fontsize=fs, ha='left', va='top')
                y-=1/per_col
            ax.set_title('Kennzahlen', color=fg)
        pdf.savefig(fig); plt.close(fig)

        # Seite 3: Monats-Heatmap + Verteilung der Trade-PnLs
        if metrics.get('heatmap') is not None:
            hm=metrics['heatmap']
            fig=plt.figure(figsize=(15,8), constrained_layout=True)
            gs=fig.add_gridspec(2,2)
            ax_hm=fig.add_subplot(gs[:,0])
            months=[1,2,3,4,5,6,7,8,9,10,11,12]
            hm_plot=hm.reindex(columns=months)
            norm=TwoSlopeNorm(vmin=min(-20, np.nanmin(hm_plot.values) if np.isfinite(np.nanmin(hm_plot.values)) else -10), vcenter=0, vmax=max(20, np.nanmax(hm_plot.values) if np.isfinite(np.nanmax(hm_plot.values)) else 10))
            im=ax_hm.imshow(hm_plot.values, aspect='auto', cmap='RdYlGn', norm=norm)
            ax_hm.set_yticks(range(len(hm_plot.index))); ax_hm.set_yticklabels(hm_plot.index, color=fg)
            ax_hm.set_xticks(range(12)); ax_hm.set_xticklabels(['Jan','Feb','Mär','Apr','Mai','Jun','Jul','Aug','Sep','Okt','Nov','Dez'], rotation=0, color=fg)
            ax_hm.set_title('Monatsrenditen (%)', color=fg)
            for i in range(hm_plot.shape[0]):
                for j in range(hm_plot.shape[1]):
                    val=hm_plot.values[i,j]
                    if np.isfinite(val):
                        ax_hm.text(j,i,f"{val:+.1f}", ha='center', va='center', color='black', fontsize=8)
            cbar=fig.colorbar(im, ax=ax_hm, fraction=0.046, pad=0.04)
            cbar.ax.yaxis.set_tick_params(color=fg)
            for spine in cbar.ax.spines.values(): spine.set_edgecolor(fg)
            # Distribution
            ax_dist=fig.add_subplot(gs[0,1])
            pnl=np.array(metrics.get('pnl',[]))
            if pnl.size>0:
                ax_dist.hist(pnl, bins=40, color=accent, alpha=0.7)
                ax_dist.set_title('Trade PnL Distribution', color=fg)
                ax_dist.grid(alpha=0.2)
            # R-Multiple distribution
            ax_r=fig.add_subplot(gs[1,1])
            rvals=np.array(metrics.get('returns_r',[]))
            if rvals.size>0:
                ax_r.hist(rvals, bins=40, color='#29b6f6', alpha=0.7)
                ax_r.set_title('R-Multiple Distribution', color=fg)
                ax_r.grid(alpha=0.2)
            pdf.savefig(fig); plt.close(fig)

        # Seite 4: Setup & Direction Stats
        if CFG.get('REPORT_INCLUDE_SETUP_STATS', True):
            fig,axes=plt.subplots(1,2, figsize=(16,6), constrained_layout=True)
            # Setup Performance (avg pnl)
            type_stats=metrics.get('type_stats',{})
            if type_stats:
                names=list(type_stats.keys())
                avgs=[type_stats[k]['avg'] for k in names]
                axes[0].bar(names, avgs, color=accent, alpha=0.7)
                axes[0].set_title('Avg PnL by Setup', color=fg)
                axes[0].tick_params(axis='x', rotation=45)
                axes[0].grid(alpha=0.25)
            dir_stats=metrics.get('dir_stats',{})
            if dir_stats:
                names=list(dir_stats.keys())
                avgs=[dir_stats[k]['avg'] for k in names]
                axes[1].bar(names, avgs, color='#29b6f6', alpha=0.7)
                axes[1].set_title('Avg PnL by Direction', color=fg)
                axes[1].grid(alpha=0.25)
            pdf.savefig(fig); plt.close(fig)

        # Seite 5: ML Diagnostics (optional)
        if CFG.get('REPORT_SHOW_ML', True) and 'ml_auc' in metrics:
            fig=plt.figure(figsize=(15,9), constrained_layout=True)
            gs=fig.add_gridspec(2,2)
            ax1=fig.add_subplot(gs[0,0]); ax2=fig.add_subplot(gs[0,1]); ax3=fig.add_subplot(gs[1,0]); ax4=fig.add_subplot(gs[1,1])
            fpr,tpr=metrics['ml_roc']; ax1.plot(fpr,tpr,color='#29b6f6'); ax1.plot([0,1],[0,1],ls='--',color='#777'); ax1.set_title(f"ROC (AUC={metrics['ml_auc']:.3f})", color=fg)
            rc,pr=metrics['ml_pr']; ax2.plot(rc,pr,color=accent); ax2.set_title(f"PR (AP={metrics['ml_ap']:.3f})", color=fg)
            if 'ml_calib' in metrics:
                pt,pp=metrics['ml_calib']; ax3.plot(pp,pt,marker='o',color='#ffb300'); ax3.plot([0,1],[0,1],ls='--',color='#777'); ax3.set_title('Calibration', color=fg)
            if 'ml_perm_importance' in metrics:
                names,imps=metrics['ml_perm_importance']; order=np.argsort(imps)[::-1][:12]
                ax4.bar([names[i] for i in order],[imps[i] for i in order],color='#8E24AA'); ax4.set_title('Permutation Importance', color=fg); ax4.tick_params(axis='x', rotation=45)
            for ax in (ax1,ax2,ax3,ax4): ax.grid(alpha=0.25)
            pdf.savefig(fig); plt.close(fig)

        # Seite 6: Elliott Struktur (optional)
        if CFG.get('REPORT_INCLUDE_STRUCTURE', True):
            fig,ax=plt.subplots(figsize=(18,9), constrained_layout=True)
            ax.plot(daily['date'], daily['close'], color='#90caf9', lw=1.0, alpha=0.75, label='Price (Daily)')
            min_pct = CFG.get('WAVE_MIN_PCT',0.08); min_dur=CFG.get('WAVE_MIN_DURATION_DAYS',40)
            def _wave_ok(points):
                if len(points)<2: return False
                i0,i1=points[0].idx, points[-1].idx
                if i0>=len(daily) or i1>=len(daily): return False
                p0,p1=daily.iloc[i0]['close'], daily.iloc[i1]['close']
                if p0==0 or pd.isna(p0) or pd.isna(p1): return False
                move=abs(p1-p0)/p0; dur=(daily.iloc[i1]['date']-daily.iloc[i0]['date']).days
                return move>=min_pct and dur>=min_dur
            prim_imp_filtered=[w for w in bt.prim_imp if _wave_ok(w.points)] if getattr(bt,'prim_imp',None) else []
            prim_abc_filtered=[w for w in bt.prim_abc if _wave_ok(w.points)] if getattr(bt,'prim_abc',None) else []
            def _plot_degree(df, imps, abcs, color_imp, color_abc):
                label_gap=pd.Timedelta(days=CFG.get('WAVE_LABEL_GAP_DAYS',60))
                last=None
                for imp in imps:
                    xs=[df.iloc[p.idx]['date'] for p in imp.points]; ys=[p.price for p in imp.points]
                    ax.plot(xs,ys,color=color_imp,lw=1.6,alpha=0.95)
                    for ki,name in zip([0,2,len(imp.points)-1],["1","3","(end)"]):
                        t=df.iloc[imp.points[ki].idx]['date']; y=imp.points[ki].price
                        if (last is None) or (t-last)>=label_gap:
                            ax.text(t,y,name,fontsize=9,bbox=dict(boxstyle='round,pad=0.25',fc=bg,ec=color_imp,alpha=0.85),color=color_imp)
                            last=t
                for pat in abcs:
                    xs=[df.iloc[p.idx]['date'] for p in pat.points]; ys=[p.price for p in pat.points]
                    ax.plot(xs,ys,color=color_abc,lw=1.4,alpha=0.9,ls='--')
                    for idx,name in [(0,'A'),(1,'B'),(-1,'C')]:
                        real=pat.points[idx].idx
                        t=df.iloc[real]['date']; y=pat.points[idx].price
                        if (last is None) or (t-last)>=label_gap:
                            ax.text(t,y,name,fontsize=9,bbox=dict(boxstyle='round,pad=0.25',fc=bg,ec=color_abc,alpha=0.85),color=color_abc)
                            last=t
            _plot_degree(daily, prim_imp_filtered, prim_abc_filtered, '#8E24AA', '#FB8C00')
            ax.set_title('Elliott Structure – Primary (gefiltert)', color=fg)
            ax.grid(alpha=0.25)
            ax.legend(loc='upper left', facecolor=bg, edgecolor=fg)
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
            pdf.savefig(fig); plt.close(fig)

        # Seite 7: Glossar (Kennzahlen Erklärungen)
        fig,ax=plt.subplots(figsize=(18,10), constrained_layout=True)
        ax.axis('off')
        explanations=[
            ("Total Return","Gesamtrendite in % vom Startkapital."),
            ("CAGR","Jährlich geometrisch gemittelte Wachstumsrate."),
            ("Winrate","Prozent der Gewinn-Trades."),
            ("Profit Factor","Summe Gewinne / Summe Verluste (abs)."),
            ("Expectancy","Erwarteter Gewinn pro Trade in $."),
            ("Expect R","Durchschnittliches R-Multiple pro Trade."),
            ("Sharpe","(µ-Rf)/σ annualisiert auf Periodenbasis."),
            ("Sortino","(µ-Rf)/Downside-Volatilität."),
            ("Ulcer Index","Wurzel des mittleren quadratischen Drawdowns."),
            ("UPI","(CAGR-Rf)/UlcerIndex (Ulcer Performance Index)."),
            ("Gain/Pain","Summe positiver Periodenrenditen / Summe negativer."),
            ("Max DD","Tiefster kumulativer Drawdown in %."),
            ("Calmar","CAGR / |Max Drawdown|."),
            ("Ann.Vol","Annualisierte Volatilität (Std * sqrt(periods/year))."),
            ("Kelly %","Kelly-Fraktion basierend auf Winrate & Avg R."),
            ("MAE/MFE (R)","Durchschnittlicher maximaler Verlust/Gewinn in R während Trade."),
            ("Exposure","Zeitlicher Anteil mit offenen Positionen."),
            ("Gain-to-Pain","Alternative zu PF mit Zeitreihenrenditen."),
            ("UPI","Robuster Drawdown-bereinigter Performance-Index."),
            ("PF Long/Short","Profit Factor getrennt nach Richtung."),
            ("Payoff","Avg Win / |Avg Loss|."),
            ("Kelly %","Theoretische optimale Risikofraktion (vereinfachte Formel)."),
            ("R-Multiple","P&L / initiales Risiko (Entry-Stop)."),
            ("Streaks","Längste Folge von Gewinn- bzw. Verlusttrades."),
        ]
        fs=11; y=0.95
        ax.text(0.02,0.98,'Glossar / Kennzahlen – Erläuterungen', fontsize=14, fontweight='bold', color=fg, ha='left', va='top')
        col_break=math.ceil(len(explanations)/2)
        left=explanations[:col_break]; right=explanations[col_break:]
        def _block(items, x):
            y=0.93
            for k,v in items:
                ax.text(x,y,f"{k}", fontsize=fs, fontweight='bold', color=fg, ha='left', va='top')
                ax.text(x+0.18,y,f"{v}", fontsize=fs, color=fg, ha='left', va='top', wrap=True)
                y-=0.035
        _block(left, 0.02); _block(right, 0.52)
        pdf.savefig(fig); plt.close(fig)
    print(f"Report gespeichert: {pdf_path}")


# --------------------------------------------------------------------------------------
# Final main() (ensures all definitions above are loaded before execution)
# --------------------------------------------------------------------------------------
def main():
    """Main mit Portfolio-/Allocation-Unterstützung und erweitertem Portfolio-Report."""
    global CFG
    args = parse_args()
    # Normalize FTMO challenge aliases
    if getattr(args, 'ftmo_challenge100k', False) and not args.ftmo_challenge:
        args.ftmo_challenge = 100000.0
    if getattr(args, 'ftmo_challenge1mio', False) and not args.ftmo_challenge:
        args.ftmo_challenge = 1000000.0
    # Symbole aus CLI oder Portfolio-Datei
    symbols = [s.strip() for s in args.symbol.split(',') if s.strip()]
    allocations_df = None
    if getattr(args,'portfolio_file',None):
        pf=args.portfolio_file
        tried=[pf]
        if pf and not os.path.isfile(pf):
            base=os.path.basename(pf)
            alt=os.path.join('Portfolios', base)
            if os.path.isfile(alt):
                pf=alt
                tried.append(alt)
        try:
            allocations_df = load_portfolio_file(pf)
            symbols = allocations_df['symbol'].tolist()
            print(f"[Portfolio-Datei] {len(symbols)} Symbole geladen: {symbols} (Quelle: {pf})")
        except Exception as e:
            print(f"[WARN] Portfolio Datei Problem: {e} – verwende CLI Symbole (versucht: {', '.join(tried)})")
            allocations_df=None
    portfolio_trades=[]
    symbol_equities={}
    base_start_cap = PROFILES[args.profile]['START_CAPITAL']

    # Resolve firm preset and overrides into rule set
    def _resolve_rules(a):
        # defaults resemble FTMO standard
        rules = {
            'daily_loss_pct': 5.0,
            'max_loss_pct': 10.0,
            'profit_target_pct': 10.0,
            'min_days': 4,
            'time_limit_days': 30,
            'trailing': False,
        }
        firm = getattr(a,'firm', None)
        if firm == 'ftmo-aggressive':
            rules.update({'daily_loss_pct': 10.0, 'max_loss_pct': 20.0})
        elif firm == 'ftmo':
            rules.update({'daily_loss_pct': 5.0, 'max_loss_pct': 10.0})
        elif firm == 'apex':
            # Generic trailing model; adjust via --rules-* if needed
            rules.update({'daily_loss_pct': None, 'max_loss_pct': 5.0, 'trailing': True, 'min_days': 0, 'time_limit_days': 30})
        elif firm == 'fundednext':
            # No time limit; allow higher static max loss by default
            rules.update({'daily_loss_pct': None, 'max_loss_pct': 15.0, 'time_limit_days': 0, 'min_days': 0})
        # Apply explicit overrides if provided
        if getattr(a,'rules_daily_loss', None) is not None:
            rules['daily_loss_pct'] = float(a.rules_daily_loss)
        if getattr(a,'rules_max_loss', None) is not None:
            rules['max_loss_pct'] = float(a.rules_max_loss)
        if getattr(a,'rules_profit_target', None) is not None:
            rules['profit_target_pct'] = float(a.rules_profit_target)
        if getattr(a,'rules_min_days', None) is not None:
            rules['min_days'] = int(a.rules_min_days)
        if getattr(a,'rules_time_limit', None) is not None:
            rules['time_limit_days'] = int(a.rules_time_limit)
        if getattr(a,'rules_trailing', False):
            rules['trailing'] = True
        return rules

    firm_rules = _resolve_rules(args)

    for sym in symbols:
        base = PROFILES[args.profile].copy()
        base['_PROFILE']=args.profile
        base['SYMBOL']=sym
        if allocations_df is not None:
            w=float(allocations_df.loc[allocations_df.symbol==sym,'weight'].iloc[0])
            base['START_CAPITAL']=base_start_cap * w
        # Toggles / Flags
        if args.no_ml: base['USE_ML']=False
        if args.show_intermediate: base['SHOW_INTERMEDIATE']=True
        base['USE_CSV']=getattr(args,'csv',False)
        base['REALISIEREN']=getattr(args,'realisieren',False)
        base['FEE']=getattr(args,'fee',0.0)
        base['SLIPPAGE']=getattr(args,'slippage',0.0)
        base['SLOW']=min(max(getattr(args,'slow',0.0),0.0),30.0)
        base['FTMO_SPLIT']=1.0 - (0.2 if getattr(args,'ftmo', False) else 0.0)
        if getattr(args,'monthly_payout',0.0)>0: base['MONTHLY_PAYOUT']=args.monthly_payout
        if getattr(args,'no_ema_trend',False): base['USE_EMA_TREND']=False
        if getattr(args,'no_daily_ema',False): base['USE_DAILY_EMA']=False
        if getattr(args,'no_adx',False): base['USE_ADX']=False
        if getattr(args,'use_adx',False): base['USE_ADX']=True
        if getattr(args,'adx_threshold',None): base['ADX_TREND_THRESHOLD']=args.adx_threshold
        base['ANALYZE_REGIMES'] = getattr(args,'analyze_regimes',False)
        # Volatility Regime Filter
        base['VOLA_FILTER_MIN'] = getattr(args,'vola_min',0)
        base['VOLA_FILTER_MAX'] = getattr(args,'vola_max',100)
        # Volatility Forecast Sizing
        base['USE_VOLA_FORECAST'] = getattr(args,'vola_forecast',False) and VOLA_FORECAST_AVAILABLE
        base['VOLA_FORECAST_WINDOW'] = getattr(args,'vola_forecast_window',252)
        if getattr(args,'vola_forecast',False) and not VOLA_FORECAST_AVAILABLE:
            print("[WARN] --vola-forecast requested but volatility_backtest.py not available")
        # Local data path
        base['LOCAL_DATA_PATH'] = getattr(args,'local_data',None)
        if getattr(args,'no_confirm',False): base['REQUIRE_CONFIRM']=False
        base['DEEP_CF']=getattr(args,'deep_counterfactuals',False)
        base['FULL_GRID_CF']=getattr(args,'full_grid_cf',False)
        for k,v in [('APPLY_THRESHOLD_SHIFT',0.0),('THR_OFFSET_W3',0.0),('THR_OFFSET_C',0.0),('THR_OFFSET_W5',0.0),('THR_OFFSET_OTHER',0.0)]:
            base.setdefault(k,v)
        # --- NEUTRAL MODE: disable overfitting-prone features ---
        if getattr(args, 'neutral', False):
            base['OPTIMIZE_ML_THRESHOLD'] = False      # no OOS threshold search
            base['AUTO_RISK_SCALE'] = False            # no dynamic risk scaling
            base['DYNAMIC_DD_RISK'] = False            # no DD-based risk reduction
            base['USE_VOL_TARGET'] = False             # no vol-target sizing
            base['SIZE_BY_PROB'] = False               # no probability-based sizing
            base['APPLY_THRESHOLD_SHIFT'] = 0.0        # no manual threshold shift
            base['THR_OFFSET_W3'] = 0.0                # no setup-specific offsets
            base['THR_OFFSET_C'] = 0.0
            base['THR_OFFSET_W5'] = 0.0
            base['THR_OFFSET_OTHER'] = 0.0
            base['TRAIN_TEST_PURGE_BARS'] = 20         # gap between train/test
            base['NEUTRAL_MODE'] = True
            print("[NEUTRAL] Backtest läuft ohne Optimierungen (fester Threshold, fixes Risiko, keine Offsets)")
        else:
            base['NEUTRAL_MODE'] = False
            base['TRAIN_TEST_PURGE_BARS'] = 0
        
        # --- MOMENTUM FILTER ---
        if getattr(args, 'momentum', False):
            base['USE_MOMENTUM_FILTER'] = True
            base['MOMENTUM_PERIOD'] = getattr(args, 'momentum_period', 14)
            base['MOMENTUM_THRESHOLD'] = getattr(args, 'momentum_threshold', 0.0)
            print(f"[MOMENTUM] Filter aktiv: Periode={base['MOMENTUM_PERIOD']}, Threshold={base['MOMENTUM_THRESHOLD']:.4f}")
        else:
            base['USE_MOMENTUM_FILTER'] = False
        
        # --- MOMENTUM EXIT (early exit when momentum weakens) ---
        mom_exit_bars = getattr(args, 'momentum_exit', 0)
        base['MOMENTUM_EXIT_BARS'] = mom_exit_bars
        if mom_exit_bars > 0:
            base['MOMENTUM_PERIOD'] = getattr(args, 'momentum_period', 14)
            print(f"[MOMENTUM-EXIT] Aktiv: Exit nach {mom_exit_bars} abnehmenden Bars (Periode={base['MOMENTUM_PERIOD']})")
        
        # --- MINIMUM PROFIT FACTOR (wie Live System) ---
        min_pf = getattr(args, 'min_pf', 0.0)
        base['MIN_PF'] = min_pf
        if min_pf > 0:
            print(f"[MIN-PF] Filter aktiv: Nur Trades mit TP/SL >= {min_pf:.2f} (wie Live System)")
        
        CFG=base
        if not getattr(args, 'neutral', False):
            _apply_manual_offsets(CFG,args)
        print(f"[START] Profil={CFG['_PROFILE']} Symbol={sym} StartCap={CFG['START_CAPITAL']:.2f} ML={CFG.get('USE_ML')} Shift={CFG.get('APPLY_THRESHOLD_SHIFT',0.0):+0.2f}")
        try:
            daily,h1,m30 = load_data()
            # If FTMO challenge sizing requested: auto-tune risk per trade toward allowed MaxDD
            if getattr(args,'ftmo_challenge', None):
                # Auto-tune risk to utilize allowed max loss for the selected rules
                tuned_risk = ftmo_tune_risk(daily, h1, m30, CFG,
                                            allowed_max_loss_pct=(float(firm_rules['max_loss_pct'])/100.0 if firm_rules.get('max_loss_pct') else 0.10),
                                            target_utilization=0.95, iters=4)
                if tuned_risk and isinstance(tuned_risk, float):
                    CFG['RISK_PER_TRADE'] = min(max(tuned_risk, CFG.get('RISK_PER_TRADE_MIN', tuned_risk)), CFG.get('RISK_PER_TRADE_MAX', tuned_risk))
                    print(f"[FTMO] Auto-Tuned RISK_PER_TRADE -> {CFG['RISK_PER_TRADE']:.4f}")
            bt=Backtester(daily,h1,m30)
            metrics=bt.run()
            if not metrics:
                print(f"[{sym}] Keine Ergebnisse."); continue
            print("\n--- Ergebnisse ---")
            print(f"Total Return: {metrics['total_return']:.2f}% | CAGR: {metrics['cagr']:.2f}% | Trades: {metrics['trades']}")
            print(f"Winrate: {metrics['hit']:.2f}% | PF: {metrics['profit_factor']:.2f} | Expectancy: ${metrics['expectancy']:.2f}")
            print(f"Vol (ann.): {metrics['vol']:.2f}% | Sharpe: {metrics['sharpe']:.2f} | Sortino: {metrics['sortino']:.2f} | UPI: {metrics['upi']:.2f} | GPR: {metrics['gain_to_pain']:.2f}")
            # FTMO Challenge (symbol-level): pass-rate across attempts under FTMO rules
            if getattr(args,'ftmo_challenge', None):
                try:
                    import pandas as _pd
                    eq_vals=[e['capital'] for e in metrics.get('equity',[])]
                    if eq_vals:
                        start_cap = float(eq_vals[0])
                        # Build daily equity series for attempts
                        eq_series = _pd.Series(eq_vals, index=[_pd.to_datetime(e['date']) if 'date' in e else i for i,e in enumerate(metrics.get('equity',[]))])
                        # If index isn't datetime, resample via integer days
                        if not isinstance(eq_series.index, _pd.DatetimeIndex):
                            base_date = _pd.Timestamp.today().normalize() - _pd.Timedelta(days=len(eq_series))
                            eq_series.index = [base_date + _pd.Timedelta(days=i) for i in range(len(eq_series))]
                        daily_eq = eq_series.resample('1D').last().ffill()
                        # collect trade dates if available for min_trading_days rule
                        tdates = None
                        try:
                            tdf = _pd.DataFrame([t.__dict__ for t in bt.trades])
                            tdf['time_out'] = _pd.to_datetime(tdf['time_out'])
                            tdates = set(tdf['time_out'].dt.normalize().unique())
                        except Exception:
                            pass
                        attempts, passed = ftmo_evaluate_attempts(
                            daily_eq,
                            float(args.ftmo_challenge),
                            daily_loss_pct=(float(firm_rules['daily_loss_pct'])/100.0 if firm_rules.get('daily_loss_pct') else 0.0),
                            max_loss_pct=(float(firm_rules['max_loss_pct'])/100.0 if firm_rules.get('max_loss_pct') else 0.10),
                            profit_target_pct=(float(firm_rules['profit_target_pct'])/100.0 if firm_rules.get('profit_target_pct') else 0.10),
                            min_trading_days=int(firm_rules.get('min_days',4)),
                            time_limit_days=int(firm_rules.get('time_limit_days',30)),
                            trade_dates=tdates,
                            trailing_max_loss=bool(firm_rules.get('trailing', False))
                        )
                        rate = (passed/attempts*100.0) if attempts>0 else 0.0
                        print(f"[FTMO] Symbol {sym}: Passrate {rate:.1f}% (Passes {passed}/{attempts})")
                except Exception:
                    pass
            if CFG.get('_DEEP_RERUN_METRICS'):
                rr=CFG['_DEEP_RERUN_METRICS']
                print(f"[RERUN] Trades={rr['trades']} Winrate={rr['winrate']*100:.2f}% CAGR={rr['cagr']:.2f}% MaxDD={rr['max_dd']:.2f}% PF={rr['pf']:.2f}")
            os.makedirs(RESULTS_PDF_DIR, exist_ok=True); os.makedirs(RESULTS_CSV_DIR, exist_ok=True)
            ts=datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            safe_sym=sanitize_symbol(sym)
            out_csv=os.path.join(RESULTS_CSV_DIR,f"{safe_sym}_{ts}.csv")
            out_pdf=os.path.join(RESULTS_PDF_DIR,f"{safe_sym}_{ts}.pdf")
            pd.DataFrame([t.__dict__ for t in bt.trades]).to_csv(out_csv, index=False)
            for t in bt.trades:
                d=t.__dict__.copy(); d['symbol']=sym; portfolio_trades.append(d)
            symbol_equities[sym]=build_equity_series(bt.trades, base['START_CAPITAL'])
            try:
                plot_report(daily,h1,bt,metrics,pdf_path=out_pdf)
                print(f"Report gespeichert: {out_pdf}")
            except Exception as e:
                print(f"Report Fehler: {e}")
            print(f"CSV: {out_csv} | PDF: {out_pdf}")
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[ERROR] Symbol {sym}: {e}")

    # Portfolio Report
    if len(symbols)>1 and portfolio_trades:
        try:
            import pandas as _pd
            dfp=_pd.DataFrame(portfolio_trades)
            dfp['time_out']=_pd.to_datetime(dfp['time_out'])
            dfp=dfp.sort_values('time_out')
            # Optional Winsorization / Trimming für Visuals
            trim_q = getattr(args,'portfolio_trim_pnl', None)
            if trim_q and 0.5 < trim_q < 1.0:
                upper = dfp['pnl'].quantile(trim_q)
                lower = dfp['pnl'].quantile(1-trim_q)
                dfp['pnl_trimmed'] = dfp['pnl'].clip(lower, upper)
            else:
                dfp['pnl_trimmed'] = dfp['pnl']
            # Build aggregated equity (sum of per-symbol equity series)
            # unify index
            all_index=None
            for ser in symbol_equities.values():
                all_index = ser.index if all_index is None else all_index.union(ser.index)
            aligned={s: ser.reindex(all_index).ffill() for s,ser in symbol_equities.items()}
            total_equity = sum(aligned.values())
            # Reconstruct PnL mapping for dfp (map to day equity then diff)
            dfp['equity_day']=dfp['time_out'].dt.floor('D').map(total_equity)
            # KPIs
            def _drawdown(e):
                roll=e.cummax(); dd=(e/roll-1.0)
                return dd, dd.min()*100.0
            dd_series, max_dd = _drawdown(total_equity)
            start=total_equity.iloc[0]; end=total_equity.iloc[-1]
            total_return=(end/start-1)*100.0
            days=(total_equity.index[-1]-total_equity.index[0]).days or 1
            cagr=((end/start)**(365.25/days)-1)*100.0 if end>0 and start>0 else 0.0
            daily_ret = total_equity.pct_change().dropna()
            vol = daily_ret.std()* (252**0.5) *100.0 if not daily_ret.empty else 0.0
            rf = RISK_FREE_RATE/100.0
            sharpe = ((daily_ret.mean()*252 - rf)/ (daily_ret.std()* (252**0.5))) if daily_ret.std()>0 else 0.0
            # FTMO pass-rate on portfolio equity across attempts
            ftmo_rate = None
            ftmo_passes = 0
            ftmo_attempts = 0
            if getattr(args,'ftmo_challenge', None):
                try:
                    # attempt evaluation on aggregated equity
                    tdates = None
                    try:
                        # union of trade dates across symbols
                        tdates = set(pd.to_datetime(dfp['time_out']).dt.normalize().unique())
                    except Exception:
                        pass
                    attempts, passed = ftmo_evaluate_attempts(
                        total_equity,
                        float(args.ftmo_challenge),
                        daily_loss_pct=(float(firm_rules['daily_loss_pct'])/100.0 if firm_rules.get('daily_loss_pct') else 0.0),
                        max_loss_pct=(float(firm_rules['max_loss_pct'])/100.0 if firm_rules.get('max_loss_pct') else 0.10),
                        profit_target_pct=(float(firm_rules['profit_target_pct'])/100.0 if firm_rules.get('profit_target_pct') else 0.10),
                        min_trading_days=int(firm_rules.get('min_days',4)),
                        time_limit_days=int(firm_rules.get('time_limit_days',30)),
                        trade_dates=tdates,
                        trailing_max_loss=bool(firm_rules.get('trailing', False))
                    )
                    ftmo_attempts, ftmo_passes = attempts, passed
                    ftmo_rate = (passed/attempts*100.0) if attempts>0 else 0.0
                except Exception:
                    ftmo_rate = None
            # Contribution
            contrib_raw = dfp.groupby('symbol')['pnl'].sum().sort_values(ascending=False)
            contrib_trim = dfp.groupby('symbol')['pnl_trimmed'].sum().reindex(contrib_raw.index)
            max_contrib = getattr(args,'portfolio_max_contrib',25) or 25
            if len(contrib_raw) > max_contrib:
                top_raw = contrib_raw.iloc[:max_contrib]
                rest_raw = contrib_raw.iloc[max_contrib:].sum()
                top_trim = contrib_trim.iloc[:max_contrib]
                rest_trim = contrib_trim.iloc[max_contrib:].sum()
                contrib = top_trim.copy()
                contrib.loc['(REST)'] = rest_trim
                contrib_raw_for_table = top_raw.copy()
                contrib_raw_for_table.loc['(REST)'] = rest_raw
            else:
                contrib = contrib_trim
                contrib_raw_for_table = contrib_raw
            # Correlation
            returns_df = _pd.DataFrame({s: aligned[s].pct_change() for s in aligned}).dropna(how='all')
            corr = returns_df.corr().fillna(0.0) if returns_df.shape[1]>1 else None
            # Erweiterte Metriken auf Trades (aggregiert)
            trades_df = dfp.copy()
            wins = trades_df[trades_df['pnl']>0]
            losses = trades_df[trades_df['pnl']<0]
            winrate = len(wins)/len(trades_df) if len(trades_df)>0 else 0.0
            pf = (wins['pnl'].sum()/abs(losses['pnl'].sum())) if losses['pnl'].sum()!=0 else float('inf') if len(wins)>0 else 0.0
            avg_trade = trades_df['pnl'].mean() if len(trades_df)>0 else 0.0
            expectancy = avg_trade
            ret_maxdd = (total_return/abs(max_dd)) if max_dd<0 else float('inf')
            # Distribution stats on daily returns of portfolio
            skew = daily_ret.skew() if not daily_ret.empty else 0.0
            kurt = daily_ret.kurtosis() if not daily_ret.empty else 0.0
            # Output files
            ts_port=datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            port_csv=os.path.join(RESULTS_CSV_DIR,f"PORTFOLIO_{ts_port}.csv")
            dfp.to_csv(port_csv,index=False)
            port_pdf=os.path.join(RESULTS_PDF_DIR,f"PORTFOLIO_{ts_port}.pdf")
            # Build PDF
            import matplotlib.pyplot as _plt
            from matplotlib.backends.backend_pdf import PdfPages as _PdfPages
            with _PdfPages(port_pdf) as pdf:
                # Page 1: KPIs + Allocation
                fig,ax=_plt.subplots(figsize=(16,9))
                ax.axis('off')
                ax.text(0.02,0.96,'Portfolio Report', fontsize=20, fontweight='bold')
                # Symbol Liste umbrechen
                sym_line = ', '.join(symbols)
                wrap_width = 110
                wrapped=[]
                while len(sym_line)>wrap_width:
                    cut = sym_line.rfind(',',0,wrap_width)
                    if cut==-1: break
                    wrapped.append(sym_line[:cut])
                    sym_line = sym_line[cut+1:].lstrip()
                wrapped.append(sym_line)
                ax.text(0.02,0.905,'Symbole:', fontsize=12, fontweight='bold')
                ax.text(0.02,0.885,'\n'.join(wrapped[:4]) + ('\n...' if len(wrapped)>4 else ''), fontsize=10, va='top')
                # KPI Block (linke Spalte)
                kpi_y=0.80
                kpis=[
                    ("Total Return", f"{total_return:.2f}%"),
                    ("CAGR", f"{cagr:.2f}%"),
                    ("MaxDD", f"{max_dd:.2f}%"),
                    ("Sharpe", f"{sharpe:.2f}"),
                    ("Ann.Vol", f"{vol:.2f}%"),
                    ("Winrate", f"{winrate*100:.2f}%"),
                    ("PF", f"{pf:.2f}" if pf!=float('inf') else 'inf'),
                    ("AvgTrade", f"{avg_trade:,.2f}"),
                    ("Expectancy", f"{expectancy:,.2f}"),
                    ("Ret/MaxDD", f"{ret_maxdd:.2f}" if ret_maxdd!=float('inf') else 'inf'),
                    ("Skew", f"{skew:.2f}"),
                    ("Kurtosis", f"{kurt:.2f}"),
                ]
                if ftmo_rate is not None:
                    kpis.insert(0,("FTMO Passrate", f"{ftmo_rate:.1f}% ({ftmo_passes}/{ftmo_attempts})"))
                for name,val in kpis:
                    ax.text(0.02,kpi_y,f"{name}:", fontsize=12, fontweight='bold');
                    ax.text(0.18,kpi_y,f"{val}", fontsize=12)
                    kpi_y-=0.035
                ax.text(0.02,kpi_y-0.01,f"Start: {start:,.0f}  Ende: {end:,.0f}", fontsize=11)
                # Allocations (rechte Spalte)
                if allocations_df is not None:
                    alloc_sorted = allocations_df.sort_values('weight', ascending=False).copy()
                    alloc_lines = [f"{r.symbol}: {r.weight:.2%}" for r in alloc_sorted.itertuples()]
                    alloc_text='\n'.join(alloc_lines)
                    ax.text(0.48,0.90,'Allocations:', fontsize=13, fontweight='bold')
                    ax.text(0.48,0.88,alloc_text, fontsize=10, va='top')
                    ax.text(0.48,0.88 - 0.035*len(alloc_lines) - 0.015, f"Summe: {alloc_sorted['weight'].sum():.2%}", fontsize=11, fontweight='bold')
                # Contribution Tabelle (mittig)
                contrib_y=0.90
                ax.text(0.30,0.90,'Beiträge (PnL, getrimmt):', fontsize=13, fontweight='bold')
                total_pnl = contrib.sum() if len(contrib)>0 else 1.0
                for s,v in contrib.items():
                    if contrib_y < 0.10: break
                    ax.text(0.30,contrib_y-0.05,f"{s}", fontsize=10)
                    ax.text(0.43,contrib_y-0.05,f"{v:,.0f}", fontsize=10, ha='right')
                    ax.text(0.45,contrib_y-0.05,f"{(v/total_pnl):.2%}", fontsize=10, ha='left')
                    contrib_y-=0.028
                pdf.savefig(fig); _plt.close(fig)

                # Page 2: Equity + Drawdown
                fig,axes=_plt.subplots(2,1, figsize=(15,8), sharex=True, gridspec_kw={'height_ratios':[3,1]})
                axes[0].plot(total_equity.index,total_equity.values,color='navy',label='Equity')
                axes[0].set_title('Portfolio Equity')
                axes[0].grid(alpha=0.3); axes[0].legend()
                axes[1].fill_between(dd_series.index, dd_series.values*100.0, 0, color='firebrick', alpha=0.5)
                axes[1].set_ylabel('Drawdown %')
                axes[1].grid(alpha=0.3)
                pdf.savefig(fig); _plt.close(fig)

                # Page 3: Contribution Bars (horizontal) + Correlation Heatmap
                fig,_axes=_plt.subplots(1,2, figsize=(17,7))
                if len(contrib)>0:
                    sorted_contrib = contrib.sort_values()
                    _axes[0].barh(sorted_contrib.index, sorted_contrib.values, color='teal')
                    _axes[0].set_title('PnL Beiträge (getrimmt)')
                    for i,(s,v) in enumerate(sorted_contrib.items()):
                        _axes[0].text(v, i, f" {v:,.0f}", va='center', fontsize=8)
                    _axes[0].grid(alpha=0.3, axis='x')
                if corr is not None:
                    im=_axes[1].imshow(corr.values, cmap='coolwarm', vmin=-1, vmax=1)
                    _axes[1].set_title('Korrelation (Daily Returns)')
                    _axes[1].set_xticks(range(len(corr.columns)))
                    _axes[1].set_xticklabels(corr.columns, rotation=90, ha='center', fontsize=7)
                    _axes[1].set_yticks(range(len(corr.index)))
                    _axes[1].set_yticklabels(corr.index, fontsize=7)
                    if getattr(args,'portfolio_corr_values', False):
                        for i in range(corr.shape[0]):
                            for j in range(corr.shape[1]):
                                _axes[1].text(j,i,f"{corr.values[i,j]:+.2f}", ha='center', va='center', fontsize=5)
                    fig.colorbar(im, ax=_axes[1], fraction=0.046, pad=0.04)
                pdf.savefig(fig); _plt.close(fig)
            print(f"Portfolio CSV: {port_csv} | Portfolio PDF: {port_pdf}")
        except Exception as e:
            print(f"[Portfolio] Fehler: {e}")

if __name__ == "__main__":
    main()