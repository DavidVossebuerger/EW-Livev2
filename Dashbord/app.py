from __future__ import annotations

import hashlib
import io
import re
import secrets
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, List, Optional, Tuple

import altair as alt
import numpy as np
import pandas as pd
import pyotp
import qrcode
import streamlit as st

# Configure Streamlit page early so the layout matches the dashboard use case.
st.set_page_config(page_title="EW Live Dashboard", layout="wide")

DEFAULT_LOG = Path(__file__).resolve().parents[1] / "logs" / "live_execution.txt"
REMOTE_SEGMENT_DIR = Path(r"C:\Users\Administrator\Documents\EW-Livev2.1\logs\segments")
LOCAL_SEGMENT_DIR = Path(__file__).resolve().parents[1] / "logs" / "segments"
RESULTS_SEGMENT_DIR = Path(__file__).resolve().parents[1] / "Ergebnisse" / "segments"
AUTH_DB_PATH = Path(__file__).resolve().parents[1] / "auth.db"
DEFAULT_ADMIN_EMAIL = "vossebuerger@fmmuc.com"
DEFAULT_ADMIN_PASSWORD = "mimiKatze1!"
DEFAULT_ADMIN_TOTP_SECRET = "H3OTZX3E66ETCN4XXMHHFLV4BLDET3D4"
EMERGENCY_EMAIL = "guest1@fmmuc.com"
EMERGENCY_PASSWORD = "guest1"
EMERGENCY_TOTP_CODE = "000000"
TIMESTAMP_OFFSET_PATTERN = re.compile(r"([+-]\d{2})(\d{2})$")
CR_PATTERN = re.compile(r"Chance/Risiko\s+([0-9.,]+)")
MIN_FACTOR_PATTERN = re.compile(r"Mindestfaktor\s+([0-9.,]+)")
PRICE_GUARD_PATTERN = re.compile(
    r"aktueller Preis\s+([0-9.,]+)\s+(<=|>=)\s+(Stop|TP)\s+([0-9.,]+)",
    re.IGNORECASE,
)
STOP_DIST_PATTERN = re.compile(r"Stop-Distanz\s+([0-9.,]+)\s+<\s+Mindestabstand\s+([0-9.,]+)")
CONFIDENCE_PATTERN = re.compile(r"Confidence\s+([0-9.,]+)\s+<\s+Threshold\s+([0-9.,]+)")
EXPOSURE_PATTERN = re.compile(
    r"Exponierung\s+[0-9.,]+\s+>\s+Limit\s+[0-9.,]+\s+\(max\s+([0-9.,]+)%\s+vom\s+Konto,\s+aktuell\s+([0-9.,]+)%\s+vom\s+Konto(?:,\s+Basis\s+([^\)]+))?\)"
)
COOLDOWN_PATTERN = re.compile(r"Cooldown aktiv\s+\(([0-9.,]+)m verbleibend\)")
CYCLE_DURATION_PATTERN = re.compile(r"Dauer=([0-9.,]+)s")
ENTRY_SIGNAL_PATTERN = re.compile(r"LastEntry=EntrySignal\((?P<payload>[^)]*)\)")
ENTRY_TIME_PATTERN = re.compile(r"entry_time=Timestamp\('(?P<entry_time>[^']+)'\)")
ENTRY_DIRECTION_PATTERN = re.compile(r"direction=<Dir\.\w+:\s*'(?P<direction>[A-Z]+)'\>")
ENTRY_ZONE_PATTERN = re.compile(r"entry_zone=\((?P<low>[0-9.,+-]+),\s*(?P<high>[0-9.,+-]+)\)")
CYCLE_RATE_PATTERN = re.compile(
    r"ValidationRate=(?P<validation_rate>[0-9.]+).*?ExecutionRate=(?P<execution_rate>[0-9.]+).*?DuplicateRate=(?P<duplicate_rate>[0-9.]+)",
    re.IGNORECASE,
)
EXPOSURE_SUMMARY_PATTERN = re.compile(
    r"Balance=(?P<balance>[0-9.]+) Exposure=(?P<exposure>[0-9.]+) ExposurePct=(?P<exposure_pct>[0-9.]+)% Drawdown=(?P<drawdown>-?[0-9.]+)%"
)
CYCLE_STATS_PATTERN = re.compile(
    r"Signals=(?P<signals>\d+).*?Validated=(?P<validated>\d+).*?Executed=(?P<executed>\d+).*?Duplicates=(?P<duplicates>\d+)",
    re.IGNORECASE,
)

ALLOWED_EMAIL_DOMAIN = "@fmmuc.com"
MAX_LOGIN_ATTEMPTS = 5
LOCKOUT_SECONDS = 30

DEPARTMENTS = [
    "Research",
    "Quantitative Investments",
    "Asset Management",
    "M&A",
    "Macroeconomics",
]

def _default_segment_dir() -> Path:
    for candidate in (REMOTE_SEGMENT_DIR, LOCAL_SEGMENT_DIR, RESULTS_SEGMENT_DIR):
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            return candidate
        except (OSError, PermissionError):
            continue
    raise FileNotFoundError("Keine Logsegment-Quelle verfügbar")


def _get_auth_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(AUTH_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _ensure_auth_table() -> None:
    with _get_auth_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                email TEXT PRIMARY KEY,
                password_hash TEXT NOT NULL,
                password_salt TEXT NOT NULL,
                totp_secret TEXT NOT NULL,
                is_admin INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL
            )
            """
        )


def _hash_password(password: str, salt: bytes) -> str:
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 200_000)
    return digest.hex()


def _create_user(email: str, password: str, is_admin: bool = False, totp_secret: Optional[str] = None) -> str:
    salt = secrets.token_bytes(16)
    password_hash = _hash_password(password, salt)
    if not totp_secret:
        totp_secret = pyotp.random_base32()
    with _get_auth_connection() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO users (email, password_hash, password_salt, totp_secret, is_admin, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (
                email,
                password_hash,
                salt.hex(),
                totp_secret,
                int(is_admin),
                datetime.utcnow().isoformat(),
            ),
        )
    return totp_secret


def _get_user(email: str) -> Optional[sqlite3.Row]:
    with _get_auth_connection() as conn:
        row = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
    return row


def _list_users() -> List[sqlite3.Row]:
    with _get_auth_connection() as conn:
        rows = conn.execute("SELECT * FROM users ORDER BY created_at DESC").fetchall()
    return rows


def _reset_totp_secret(email: str) -> str:
    secret = pyotp.random_base32()
    with _get_auth_connection() as conn:
        conn.execute("UPDATE users SET totp_secret = ? WHERE email = ?", (secret, email))
    return secret


def _ensure_default_admin() -> Optional[str]:
    _ensure_auth_table()
    existing = _get_user(DEFAULT_ADMIN_EMAIL)
    if existing:
        return None
    return _create_user(
        DEFAULT_ADMIN_EMAIL,
        DEFAULT_ADMIN_PASSWORD,
        is_admin=True,
        totp_secret=DEFAULT_ADMIN_TOTP_SECRET,
    )


def _ensure_emergency_account() -> None:
    _ensure_auth_table()
    if _get_user(EMERGENCY_EMAIL):
        return
    _create_user(EMERGENCY_EMAIL, EMERGENCY_PASSWORD, is_admin=False)


def _verify_credentials(email: str, password: str, totp_code: str) -> tuple[bool, str, Optional[sqlite3.Row]]:
    normalized_email = email.strip().lower()
    if not normalized_email.endswith(ALLOWED_EMAIL_DOMAIN):
        return False, f"Nur {ALLOWED_EMAIL_DOMAIN}-Accounts sind zugelassen.", None
    row = _get_user(email)
    if not row:
        return False, "Benutzer nicht gefunden.", None

    salt = bytes.fromhex(row["password_salt"])
    candidate_hash = _hash_password(password, salt)
    if not secrets.compare_digest(candidate_hash, row["password_hash"]):
        return False, "Ungültiges Passwort.", None

    if email.lower() == EMERGENCY_EMAIL.lower() and totp_code == EMERGENCY_TOTP_CODE:
        return True, "", row

    totp = pyotp.TOTP(row["totp_secret"])
    if not totp.verify(totp_code, valid_window=1):
        return False, "Ungültiger 2FA-Code.", None

    return True, "", row


def _format_totp_info(email: str, secret: str) -> str:
    totp = pyotp.TOTP(secret)
    uri = totp.provisioning_uri(email, issuer_name="EW Live Dashboard")
    return f"TOTP-Secret: {secret}\nProvisioning URI: {uri}"


def _render_totp_qr(secret: str, email: str) -> None:
    totp = pyotp.TOTP(secret)
    uri = totp.provisioning_uri(email, issuer_name="EW Live Dashboard")
    qr = qrcode.make(uri)
    buffer = io.BytesIO()
    qr.save(buffer, format="PNG")
    buffer.seek(0)
    st.image(buffer, caption="Diet einen QR-Code in deinen Authenticator", use_column_width=True)


def _inject_dashboard_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --card-bg: linear-gradient(135deg, #050913, #101a32);
            --card-border: rgba(255, 255, 255, 0.12);
            --text-muted: rgba(255, 255, 255, 0.6);
            --accent: #7ee1ff;
            --accent-strong: #4c7dff;
        }
        body {
            background-color: #01040b;
        }
        .user-chip {
            display: inline-flex;
            align-items: center;
            gap: 0.75rem;
            padding: 0.35rem 1rem;
            border-radius: 999px;
            border: 1px solid rgba(255, 255, 255, 0.18);
            background: rgba(255, 255, 255, 0.04);
            font-size: 0.95rem;
            margin-bottom: 0.75rem;
        }
        .user-chip strong {
            color: var(--accent);
        }
        .fmmuc-hero {
            background: linear-gradient(145deg, rgba(5, 9, 20, 0.95), rgba(15, 27, 54, 0.95));
            border-radius: 32px;
            padding: clamp(1.75rem, 2vw, 2.5rem);
            border: 1px solid var(--card-border);
            box-shadow: 0 30px 70px rgba(0, 4, 15, 0.85);
            overflow: hidden;
            position: relative;
        }
        .fmmuc-hero::after {
            content: "";
            position: absolute;
            inset: 0;
            background: radial-gradient(circle, rgba(126, 225, 255, 0.15), transparent 55%);
            mix-blend-mode: screen;
            opacity: 0.8;
            z-index: 0;
            pointer-events: none;
        }
        .fmmuc-hero > * {
            position: relative;
            z-index: 1;
        }
        .fmmuc-hero .hero-kicker {
            text-transform: uppercase;
            letter-spacing: 0.35em;
            font-size: 0.8rem;
            color: var(--text-muted);
            margin: 0;
        }
        .fmmuc-hero .hero-title {
            margin: 0.6rem 0 0.4rem;
            font-size: clamp(2.2rem, 3vw, 2.8rem);
            line-height: 1.2;
        }
        .fmmuc-hero .hero-description {
            margin: 0;
            color: var(--text-muted);
            line-height: 1.6;
            max-width: 640px;
        }
        .hero-metrics {
            margin-top: 1.5rem;
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
            gap: 0.8rem;
        }
        .hero-metric-card {
            padding: 1rem 1.2rem;
            border-radius: 16px;
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid rgba(255, 255, 255, 0.08);
            font-size: 0.95rem;
            color: #fff;
        }
        .hero-metric-card .subtle {
            color: var(--text-muted);
            font-size: 0.7rem;
            text-transform: uppercase;
            letter-spacing: 0.3em;
        }
        .hero-metric-card .context {
            color: var(--text-muted);
            font-size: 0.75rem;
        }
        .hero-metric-card strong {
            display: block;
            font-size: 1.6rem;
            color: var(--accent);
        }
        .hero-metric-card .subtle {
            color: var(--text-muted);
            font-size: 0.8rem;
            letter-spacing: 0.15em;
            text-transform: uppercase;
        }
        .insight-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 0.75rem;
            margin-top: 1.2rem;
        }
        .insight-card {
            padding: 0.85rem 1rem;
            border-radius: 12px;
            background: rgba(17, 22, 40, 0.8);
            border: 1px solid rgba(255, 255, 255, 0.05);
            color: #fff;
            font-size: 0.85rem;
        }
        .insight-card .context {
            color: var(--text-muted);
            font-size: 0.7rem;
            letter-spacing: 0.2em;
            text-transform: uppercase;
        }
        .insight-card strong {
            display: block;
            font-size: 1.25rem;
            color: var(--accent-strong);
        }
        .insight-summary {
            margin-top: 1rem;
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 0.9rem;
        }
        .insight-summary-card {
            padding: 1rem;
            border-radius: 14px;
            background: rgba(15, 24, 44, 0.9);
            border: 1px solid rgba(126, 225, 255, 0.2);
            box-shadow: 0 12px 40px rgba(2, 2, 17, 0.8);
        }
        .insight-summary-card strong {
            display: block;
            font-size: 1.5rem;
            color: var(--accent);
        }
        .insight-summary-card span {
            color: var(--text-muted);
            font-size: 0.8rem;
            letter-spacing: 0.15em;
            text-transform: uppercase;
        }
        .status-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 16px;
            margin-bottom: 16px;
        }
        .status-card {
            padding: 18px 22px;
            border-radius: 18px;
            background: var(--card-bg);
            border: 1px solid rgba(255, 255, 255, 0.06);
            box-shadow: 0 25px 45px rgba(0, 0, 0, 0.5);
            animation: pulse 8s ease-in-out infinite;
            min-height: 130px;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
        }
        .status-card strong {
            display: block;
            font-size: 1.8rem;
        }
        .status-card .context {
            color: var(--text-muted);
            font-size: 0.9rem;
        }
        .insight-panel,
        .insight-panel.chart-panel {
            border-radius: 18px;
            background: linear-gradient(180deg, rgba(5, 12, 26, 0.95), rgba(12, 24, 44, 0.95));
            border: 1px solid rgba(255, 255, 255, 0.12);
            padding: 16px;
            box-shadow: 0 20px 45px rgba(2, 8, 24, 0.7);
            animation: float 12s ease-in-out infinite;
        }
        @keyframes pulse {
            0% { transform: translateY(0); }
            50% { transform: translateY(-4px); box-shadow: 0 15px 25px rgba(12, 123, 255, 0.45); }
            100% { transform: translateY(0); }
        }
        @keyframes float {
            0% { transform: translateY(0); }
            50% { transform: translateY(-6px); }
            100% { transform: translateY(0); }
        }
        .chart-panel {
            padding: 0 !important;
        }
        .chart-panel .stMarkdown {
            margin-bottom: 0;
        }
        @media (max-width: 768px) {
            .fmmuc-hero {
                padding: 24px;
            }
            .status-card strong {
                font-size: 1.4rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _rerun_app() -> None:
    rerun_func = getattr(st, "experimental_rerun", None) or getattr(st, "rerun", None)
    if rerun_func:
        rerun_func()


@st.cache_data(show_spinner=False)
def load_log_entries(log_path: str, max_files: Optional[int]) -> pd.DataFrame:
    """Parse the execution log into a normalized DataFrame."""
    path = Path(log_path).expanduser()
    files: List[Path]
    if path.is_dir():
        files = sorted(p for p in path.iterdir() if p.is_file())
        if not files:
            raise FileNotFoundError(f"Keine Logdateien im Ordner: {path}")
        if max_files:
            files = files[-max_files:]
    else:
        if not path.exists():
            raise FileNotFoundError(f"Logdatei nicht gefunden: {path}")
        files = [path]

    entries: List[dict] = []
    for file_path in files:
        with file_path.open("r", encoding="utf-8", errors="ignore") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line or line.startswith("PS "):
                    continue

                parsed = _parse_log_line(line)
                if parsed:
                    parsed["source"] = file_path.name
                    entries.append(parsed)

    return pd.DataFrame(entries)


def _parse_log_line(line: str) -> Optional[dict]:
    parts = line.split(" ", 2)
    if len(parts) != 3 or not _looks_like_timestamp(parts[0]):
        metrics = _extract_metrics(line)
        return {
            "timestamp": pd.NaT,
            "level": "INFO",
            "symbol": "sys",
            "message": line,
            "category": _classify_message(line, None),
            **metrics,
        }

    ts_raw, level, remainder = parts
    try:
        timestamp = _parse_timestamp(ts_raw)
    except ValueError:
        metrics = _extract_metrics(remainder)
        return {
            "timestamp": pd.NaT,
            "level": level,
            "symbol": "sys",
            "message": remainder,
            "category": _classify_message(remainder, None),
            **metrics,
        }

    symbol: Optional[str] = None
    message = remainder
    if remainder.startswith("[") and "]" in remainder:
        bracket_end = remainder.find("]")
        symbol = remainder[1:bracket_end]
        message = remainder[bracket_end + 2 :]

    metrics = _extract_metrics(message)

    return {
        "timestamp": timestamp,
        "level": level,
        "symbol": symbol or "sys",
        "message": message,
        "category": _classify_message(message, symbol),
        **metrics,
    }


def _parse_timestamp(value: str) -> datetime:
    # The MT5 log format ends with "+0000" (without a colon). Insert the colon to keep
    # datetime.fromisoformat happy.
    match = TIMESTAMP_OFFSET_PATTERN.search(value)
    if match:
        value = f"{value[:-5]}{match.group(1)}:{match.group(2)}"
    return datetime.fromisoformat(value)


def _looks_like_timestamp(value: str) -> bool:
    return bool(re.match(r"^\d{4}-\d{2}-\d{2}T", value))


def _classify_message(message: str, symbol: Optional[str]) -> str:
    lowered = message.lower()
    if "broker untersagt" in lowered:
        return "Broker untersagt"
    if "chance/risiko" in lowered or "mindestfaktor" in lowered:
        return "Profit-Faktor"
    if "aktueller preis" in lowered and ("<= stop" in lowered or ">= tp" in lowered):
        return "Preisschutz"
    if "mock-raten" in lowered:
        return "Mock-Daten"
    if "signal" in lowered and "signals=" in lowered:
        return "Signalübersicht"
    if "cooldown aktiv" in lowered:
        return "Cooldown"
    if "confidence" in lowered and "threshold" in lowered:
        return "Confidence-Filter"
    if "stop-distanz" in lowered:
        return "Stop-Distanz"
    if "trades/std" in lowered:
        return "Trade-Limit"
    if "keine symbolinformationen" in lowered:
        return "Symbolinfo"
    if "kein aktueller preis" in lowered:
        return "Preisdaten"
    if "logsegment gespeichert" in lowered:
        return "Segmentierung"
    if symbol == "cycle":
        return "Cycle"
    if "verbunden" in lowered:
        return "Connection"
    return "Sonstiges"


def _extract_metrics(message: str) -> dict:
    data = {
        "cr_ratio": None,
        "min_factor": None,
        "price_gap": None,
        "price_relation": None,
        "stop_shortfall": None,
        "confidence_gap": None,
        "cooldown_minutes": None,
        "cycle_duration": None,
        "exposure_pct": None,
        "exposure_limit_pct": None,
        "exposure_basis": None,
        "cycle_signals": None,
        "cycle_validated": None,
        "cycle_executed": None,
        "cycle_duplicates": None,
        "validation_rate": None,
        "execution_rate": None,
        "duplicate_rate": None,
        "balance": None,
        "exposure": None,
        "drawdown": None,
    }

    cr_match = CR_PATTERN.search(message)
    if cr_match:
        data["cr_ratio"] = _safe_float(cr_match.group(1))

    mf_match = MIN_FACTOR_PATTERN.search(message)
    if mf_match:
        data["min_factor"] = _safe_float(mf_match.group(1))

    price_match = PRICE_GUARD_PATTERN.search(message)
    if price_match:
        current = _safe_float(price_match.group(1))
        operator = price_match.group(2)
        barrier_label = price_match.group(3)
        barrier_value = _safe_float(price_match.group(4))
        if current is not None and barrier_value is not None:
            gap = barrier_value - current if operator == "<=" else current - barrier_value
            data["price_gap"] = gap
            data["price_relation"] = barrier_label.capitalize()

    stop_match = STOP_DIST_PATTERN.search(message)
    if stop_match:
        actual = _safe_float(stop_match.group(1))
        required = _safe_float(stop_match.group(2))
        if actual is not None and required is not None:
            data["stop_shortfall"] = required - actual

    conf_match = CONFIDENCE_PATTERN.search(message)
    if conf_match:
        value = _safe_float(conf_match.group(1))
        threshold = _safe_float(conf_match.group(2))
        if value is not None and threshold is not None:
            data["confidence_gap"] = threshold - value

    cooldown_match = COOLDOWN_PATTERN.search(message)
    if cooldown_match:
        minutes = _safe_float(cooldown_match.group(1))
        if minutes is not None:
            data["cooldown_minutes"] = minutes

    cycle_match = CYCLE_DURATION_PATTERN.search(message)
    if cycle_match:
        data["cycle_duration"] = _safe_float(cycle_match.group(1))

    stats_match = CYCLE_STATS_PATTERN.search(message)
    if stats_match:
        data["cycle_signals"] = _safe_int(stats_match.group("signals"))
        data["cycle_validated"] = _safe_int(stats_match.group("validated"))
        data["cycle_executed"] = _safe_int(stats_match.group("executed"))
        data["cycle_duplicates"] = _safe_int(stats_match.group("duplicates"))

    rate_match = CYCLE_RATE_PATTERN.search(message)
    if rate_match:
        data["validation_rate"] = _safe_float(rate_match.group("validation_rate"))
        data["execution_rate"] = _safe_float(rate_match.group("execution_rate"))
        data["duplicate_rate"] = _safe_float(rate_match.group("duplicate_rate"))

    summary_match = EXPOSURE_SUMMARY_PATTERN.search(message)
    if summary_match:
        data["balance"] = _safe_float(summary_match.group("balance"))
        data["exposure"] = _safe_float(summary_match.group("exposure"))
        data["exposure_pct"] = _safe_float(summary_match.group("exposure_pct"))
        data["drawdown"] = _safe_float(summary_match.group("drawdown"))

    exposure_match = EXPOSURE_PATTERN.search(message)
    if exposure_match:
        limit_pct = _safe_float(exposure_match.group(1))
        actual_pct = _safe_float(exposure_match.group(2))
        if limit_pct is not None:
            data["exposure_limit_pct"] = limit_pct
        if actual_pct is not None:
            data["exposure_pct"] = actual_pct
        basis_value = exposure_match.group(3)
        if basis_value:
            data["exposure_basis"] = basis_value.strip()

    return data


def _safe_float(value: str) -> Optional[float]:
    try:
        return float(value.replace(",", "."))
    except (AttributeError, ValueError):
        return None


def _extract_numeric(payload: str, field: str) -> Optional[float]:
    match = re.search(fr"{field}=([0-9.,]+)", payload)
    return _safe_float(match.group(1)) if match else None


def _parse_entry_signal(payload: str) -> Optional[dict]:
    time_match = ENTRY_TIME_PATTERN.search(payload)
    direction_match = ENTRY_DIRECTION_PATTERN.search(payload)
    if not time_match or not direction_match:
        return None

    entry_time = pd.Timestamp(time_match.group("entry_time"))
    entry_time = entry_time.tz_localize("UTC") if entry_time.tzinfo is None else entry_time
    zone = _extract_entry_zone(payload)
    return {
        "entry_time": entry_time,
        "direction": direction_match.group("direction"),
        "entry_price": _extract_numeric(payload, "entry_price"),
        "stop_loss": _extract_numeric(payload, "stop_loss"),
        "take_profit": _extract_numeric(payload, "take_profit"),
        "confidence": _extract_numeric(payload, "confidence"),
        "setup": _extract_string_value(payload, "setup"),
        **({"entry_zone": zone, "entry_zone_low": zone[0], "entry_zone_high": zone[1]} if zone else {}),
        "entry_tf": _extract_string_value(payload, "entry_tf"),
    }


def _extract_string_value(payload: str, field: str) -> Optional[str]:
    match = re.search(fr"{field}='([^']+)'", payload)
    return match.group(1) if match else None


def _extract_entry_zone(payload: str) -> Optional[Tuple[float, float]]:
    match = ENTRY_ZONE_PATTERN.search(payload)
    if not match:
        return None
    low = _safe_float(match.group("low"))
    high = _safe_float(match.group("high"))
    if low is None or high is None:
        return None
    return (low, high)


def _prepare_entry_signals(df: pd.DataFrame) -> pd.DataFrame:
    entries: List[dict] = []
    mask = df["message"].str.contains("LastEntry=EntrySignal", na=False)
    for _, row in df[mask].iterrows():
        payload_match = ENTRY_SIGNAL_PATTERN.search(row["message"])
        if not payload_match:
            continue
        info = _parse_entry_signal(payload_match.group("payload"))
        if not info:
            continue
        info["symbol"] = row["symbol"] or "unknown"
        info["log_timestamp"] = row["timestamp"]
        info["source_file"] = row.get("source", "")
        entries.append(info)
    if not entries:
        return pd.DataFrame(
            columns=[
                "symbol",
                "entry_time",
                "direction",
                "entry_price",
                "stop_loss",
                "take_profit",
                "confidence",
                "setup",
                "log_timestamp",
                "source_file",
                "entry_zone",
                "entry_zone_low",
                "entry_zone_high",
                "entry_tf",
            ]
        )
    return pd.DataFrame(entries)
def _segment_file_stats(segment_dir: Path) -> tuple[int, Optional[pd.Timestamp]]:
    if not segment_dir.exists() or not segment_dir.is_dir():
        return 0, None
    files = [p for p in segment_dir.iterdir() if p.is_file()]
    if not files:
        return 0, None
    latest = max(pd.to_datetime(p.stat().st_mtime, unit="s", utc=True) for p in files)
    return len(files), latest


def _format_duration(seconds: Optional[float]) -> str:
    if seconds is None or (isinstance(seconds, float) and np.isnan(seconds)):
        return "-"
    return f"{seconds:.1f}s"


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_numeric_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _format_insight_value(value: Optional[float], precision: int = 2, suffix: str = "") -> str:
    if value is None:
        return "-"
    return f"{value:.{precision}f}{suffix}"


def _format_timestamp(ts: Optional[pd.Timestamp]) -> str:
    if ts is None or pd.isna(ts):
        return "-"
    ts = ts.tz_convert("UTC") if ts.tzinfo is not None else ts.tz_localize("UTC")
    return ts.strftime("%Y-%m-%d %H:%M:%S UTC")


def _render_status_cards(cards: List[Tuple[str, str, str]]) -> None:
    card_html = "".join(
        f"""
        <div class=\"status-card\">
            <span class=\"context\">{title}</span>
            <strong>{value}</strong>
            <span class=\"context\">{subtitle}</span>
        </div>
        """
        for title, value, subtitle in cards
    )
    st.markdown(f"<div class=\"status-grid\">{card_html}</div>", unsafe_allow_html=True)


def _render_user_context() -> None:
    user_email = st.session_state.get("user_email") or "Unbekannt"
    department = st.session_state.get("department") or DEPARTMENTS[0]
    st.markdown(
        f"""
        <div class=\"user-chip\">
            <span>Angemeldet als <strong>{user_email}</strong></span>
            <span>Abteilung <strong>{department}</strong></span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _cycle_snapshot(cycles: pd.DataFrame) -> dict[str, Any]:
    defaults = {
        "cycle_duration": None,
        "cycle_signals": 0,
        "cycle_validated": 0,
        "cycle_executed": 0,
        "cycle_duplicates": 0,
    }
    if cycles.empty:
        return defaults
    latest = cycles.sort_values("timestamp").iloc[-1]
    return {
        "cycle_duration": _safe_numeric_float(latest.get("duration_seconds")),
        "cycle_signals": _safe_int(latest.get("cycle_signals")),
        "cycle_validated": _safe_int(latest.get("cycle_validated")),
        "cycle_executed": _safe_int(latest.get("cycle_executed")),
        "cycle_duplicates": _safe_int(latest.get("cycle_duplicates")),
    }


def _insight_metrics(filtered: pd.DataFrame) -> dict[str, Any]:
    insights: dict[str, Any] = {}
    confidence_gap = filtered.get("confidence_gap")
    price_gap = filtered.get("price_gap")
    exposure = filtered.get("exposure_pct")
    insights["confidence_gap"] = (
        float(confidence_gap.dropna().mean()) if isinstance(confidence_gap, pd.Series) and not confidence_gap.dropna().empty else None
    )
    insights["price_gap"] = (
        float(price_gap.dropna().mean()) if isinstance(price_gap, pd.Series) and not price_gap.dropna().empty else None
    )
    insights["exposure_pct"] = (
        float(exposure.dropna().max()) if isinstance(exposure, pd.Series) and not exposure.dropna().empty else None
    )
    insights["active_symbols"] = int(filtered["symbol"].nunique()) if not filtered.empty else 0
    return insights


def _format_rate(value: Optional[float]) -> str:
    if value is None:
        return "-"
    return f"{value * 100:.1f}%"


def _execution_efficiency_snapshot(cycles: pd.DataFrame) -> dict[str, Optional[float]]:
    snapshot = {
        "signals": None,
        "validated": None,
        "executed": None,
        "duplicates": None,
        "validation_rate": None,
        "execution_rate": None,
        "duplicate_rate": None,
    }
    if cycles.empty:
        return snapshot
    latest = cycles.sort_values("timestamp").iloc[-1]
    signals = _safe_int(latest.get("cycle_signals"))
    validated = _safe_int(latest.get("cycle_validated"))
    executed = _safe_int(latest.get("cycle_executed"))
    duplicates = _safe_int(latest.get("cycle_duplicates"))
    snapshot.update(
        signals=signals,
        validated=validated,
        executed=executed,
        duplicates=duplicates,
    )
    if signals > 0:
        snapshot["validation_rate"] = validated / signals
        snapshot["duplicate_rate"] = duplicates / signals
    if validated > 0:
        snapshot["execution_rate"] = executed / validated
    return snapshot


def _execution_efficiency_timeseries(cycles: pd.DataFrame) -> pd.DataFrame:
    if cycles.empty:
        return pd.DataFrame(columns=["timestamp", "cycle_signals", "cycle_validated", "cycle_executed", "cycle_duplicates"])
    ts = cycles.copy()
    ts = ts[ts["timestamp"].notna()]
    ts = ts.sort_values("timestamp")
    ts = ts.assign(
        cycle_signals=ts["cycle_signals"].fillna(0),
        cycle_validated=ts["cycle_validated"].fillna(0),
        cycle_executed=ts["cycle_executed"].fillna(0),
        cycle_duplicates=ts["cycle_duplicates"].fillna(0),
    )
    return ts[["timestamp", "cycle_signals", "cycle_validated", "cycle_executed", "cycle_duplicates"]]


def _render_pipeline_hero(
    segment_count: int,
    last_segment_ts: Optional[pd.Timestamp],
    freshness_label: str,
    last_cycle_time: Optional[pd.Timestamp],
    stats_snapshot: dict[str, Any],
    insights: dict[str, Any],
) -> None:
    last_update_label = _format_timestamp(last_segment_ts)
    cycle_label = _format_timestamp(last_cycle_time)
    duration_label = _format_duration(stats_snapshot.get("cycle_duration"))
    signals = stats_snapshot.get("cycle_signals", 0)
    validated = stats_snapshot.get("cycle_validated", 0)
    executed = stats_snapshot.get("cycle_executed", 0)
    duplicates = stats_snapshot.get("cycle_duplicates", 0)
    insight_cards = f"""
        <div class=\"insight-card\">
            <span class=\"context\">Avg. Confidence Gap</span>
            <strong>{_format_insight_value(insights.get('confidence_gap'))}</strong>
            <span class=\"context\">Signalqualität</span>
        </div>
        <div class=\"insight-card\">
            <span class=\"context\">Avg. Price Gap</span>
            <strong>{_format_insight_value(insights.get('price_gap'))}</strong>
            <span class=\"context\">Marktversatz</span>
        </div>
        <div class=\"insight-card\">
            <span class=\"context\">Max. Exponierung</span>
            <strong>{_format_insight_value(insights.get('exposure_pct'), precision=1, suffix='%')}</strong>
            <span class=\"context\">Kontolimit</span>
        </div>
        <div class=\"insight-card\">
            <span class=\"context\">Aktive Symbole</span>
            <strong>{int(insights.get('active_symbols') or 0)}</strong>
            <span class=\"context\">Abgedeckte Märkte</span>
        </div>
        """
    st.markdown(
        f"""
        <div class=\"fmmuc-hero\">
            <p class=\"hero-kicker\">Live-Execution Monitoring</p>
            <h2 class=\"hero-title\">Transparente Ausführung & Signalqualität</h2>
            <p class=\"hero-description\">Logsegmente, Cycle-Zyklen und aktuelle Metriken zeigen, ob der Live-Betrieb sauber am Backtest dranbleibt.</p>
            <div class=\"hero-metrics\">
                <div class=\"hero-metric-card\">
                    <span class=\"subtle\">Logsegmente</span>
                    <strong>{segment_count}</strong>
                    <span class=\"context\">Aktualisiert vor {freshness_label}</span>
                </div>
                <div class=\"hero-metric-card\">
                    <span class=\"subtle\">Letzter Cycle</span>
                    <strong>{duration_label}</strong>
                    <span class=\"context\">Endet um {cycle_label}</span>
                </div>
                <div class=\"hero-metric-card\">
                    <span class=\"subtle\">Cycle-Signale</span>
                    <strong>{signals}</strong>
                    <span class=\"context\">V:{validated} · E:{executed} · D:{duplicates}</span>
                </div>
            </div>
            <div class=\"insight-grid\">{insight_cards}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _apply_chart_theme(chart: alt.Chart) -> alt.Chart:
    return (
        chart.configure_view(strokeWidth=0, fill="transparent")
        .configure_axis(
            labelColor="rgba(255, 255, 255, 0.8)",
            titleColor="rgba(255, 255, 255, 0.95)",
            gridColor="rgba(255, 255, 255, 0.15)",
        )
        .configure_legend(
            labelColor="rgba(255, 255, 255, 0.85)",
            titleColor="rgba(255, 255, 255, 0.95)",
        )
        .configure_title(color="#ffffff")
    )


def _timeline_frequency(span: pd.Timedelta) -> str:
    if span <= pd.Timedelta(hours=12):
        return "30min"
    if span <= pd.Timedelta(days=2):
        return "1H"
    if span <= pd.Timedelta(days=10):
        return "4H"
    return "1D"


def _render_login_screen(initial_secret: Optional[str]) -> None:
    st.header("Gesicherter Zugriff")
    st.caption("Bitte melde dich mit deinem @fmmuc.com-Account, Passwort und TOTP-Code an.")
    if initial_secret and not st.session_state.get("initial_totp_secret_shown"):
        st.warning(
            "Initialer TOTP-Secret (nur einmal anzeigen):\n" + _format_totp_info(DEFAULT_ADMIN_EMAIL, initial_secret)
        )
        _render_totp_qr(initial_secret, DEFAULT_ADMIN_EMAIL)
        st.session_state.initial_totp_secret_shown = True

    now = datetime.utcnow()
    lockout_until_ts = st.session_state.get("login_lockout_until")
    lockout_active = bool(lockout_until_ts and now.timestamp() < lockout_until_ts)
    if lockout_active:
        wait_seconds = max(1, int(lockout_until_ts - now.timestamp()))
        st.warning(f"Zu viele Fehlversuche. Bitte warte {wait_seconds}s, bevor du es erneut versuchst.")

    with st.form("login_form"):
        email = st.text_input("E-Mail-Adresse")
        password = st.text_input("Passwort", type="password")
        totp_code = st.text_input("2FA-Code", max_chars=6)
        st.caption("Wähle die Abteilung, in der du arbeitest. Dies dient nur der Einordnung.")
        department = st.selectbox("Abteilung", DEPARTMENTS, key="login_department")
        submitted = st.form_submit_button("Anmelden", disabled=lockout_active)

    if submitted:
        now = datetime.utcnow()
        lockout_until_ts = st.session_state.get("login_lockout_until")
        if lockout_until_ts and now.timestamp() < lockout_until_ts:
            wait_seconds = max(1, int(lockout_until_ts - now.timestamp()))
            st.error(f"Zu viele Fehlversuche. Bitte warte noch {wait_seconds}s.")
            return

        success, message, row = _verify_credentials(email, password, totp_code)
        if success and row:
            st.session_state.authenticated = True
            st.session_state.user_email = email
            st.session_state.is_admin = bool(row["is_admin"])
            st.session_state.login_error = ""
            st.session_state.department = department or DEPARTMENTS[0]
            st.session_state.login_failed_attempts = 0
            st.session_state.login_lockout_until = None
            _rerun_app()
        else:
            attempts = st.session_state.get("login_failed_attempts", 0) + 1
            st.session_state.login_failed_attempts = attempts
            remaining = max(0, MAX_LOGIN_ATTEMPTS - attempts)
            if remaining == 0:
                st.session_state.login_lockout_until = (
                    now + timedelta(seconds=LOCKOUT_SECONDS)
                ).timestamp()
                st.session_state.login_error = f"Zu viele Fehlversuche. Bitte warte {LOCKOUT_SECONDS}s."
            else:
                st.session_state.login_error = f"{message} Noch {remaining} Versuch(e)."

    if st.session_state.login_error:
        st.error(st.session_state.login_error)


def _render_admin_panel() -> None:
    users = _list_users()
    with st.sidebar.expander("Admin-Panel", expanded=True):
        st.markdown("**Nutzerverwaltung**")
        if users:
            summary = pd.DataFrame(
                [
                    {
                        "E-Mail": user["email"],
                        "Admin": "Ja" if user["is_admin"] else "Nein",
                        "Erstellt": user["created_at"],
                    }
                    for user in users
                ]
            )
            st.dataframe(summary, height=180, hide_index=True)
        else:
            st.info("Noch keine Nutzer vorhanden.")

        with st.form("new_user_form"):
            st.subheader("Neuen Nutzer anlegen")
            new_email = st.text_input("E-Mail", key="new_user_email")
            new_password = st.text_input("Passwort", type="password", key="new_user_password")
            is_admin = st.checkbox("Admin-Rechte", key="new_user_admin")
            create_submitted = st.form_submit_button("Nutzer anlegen")
        if create_submitted:
            if not new_email or not new_password:
                st.error("E-Mail und Passwort werden benötigt.")
            else:
                secret = _create_user(new_email, new_password, is_admin=is_admin)
                st.success("Nutzer angelegt. Konfiguriere den TOTP-Code im Authenticator.")
                st.code(_format_totp_info(new_email, secret), language="text")
                _render_totp_qr(secret, new_email)

        if users:
            st.markdown("---")
            selected_user = st.selectbox("2FA zurücksetzen", options=[user["email"] for user in users], key="reset_user")
            if st.button("TOTP regenerieren", key="reset_totp"):
                new_secret = _reset_totp_secret(selected_user)
                st.success("Neues TOTP-Secret erstellt.")
                st.code(_format_totp_info(selected_user, new_secret), language="text")
                _render_totp_qr(new_secret, selected_user)


def main() -> None:
    initial_totp_secret = _ensure_default_admin()
    _ensure_emergency_account()
    if initial_totp_secret and "initial_totp_secret" not in st.session_state:
        st.session_state.initial_totp_secret = initial_totp_secret
        st.session_state.initial_totp_secret_shown = False

    st.session_state.setdefault("authenticated", False)
    st.session_state.setdefault("user_email", "")
    st.session_state.setdefault("is_admin", False)
    st.session_state.setdefault("login_error", "")
    st.session_state.setdefault("department", DEPARTMENTS[0])
    st.session_state.setdefault("login_department", DEPARTMENTS[0])
    st.session_state.setdefault("login_failed_attempts", 0)
    st.session_state.setdefault("login_lockout_until", None)

    if not st.session_state.authenticated:
        _render_login_screen(st.session_state.get("initial_totp_secret"))
        return

    if st.sidebar.button("Abmelden"):
        for key in ("authenticated", "user_email", "is_admin", "login_error"):
            st.session_state[key] = False if key == "authenticated" else "" if key != "is_admin" else False
        st.session_state["department"] = DEPARTMENTS[0]
        st.session_state["login_department"] = DEPARTMENTS[0]
        st.session_state["login_failed_attempts"] = 0
        st.session_state["login_lockout_until"] = None
        _rerun_app()

    department = st.session_state.get("department") or DEPARTMENTS[0]
    user_email = st.session_state.user_email or "-"
    st.title("EW Live – Monitoring Dashboard")
    _render_user_context()
    _inject_dashboard_styles()
    st.markdown(
        """
> **Haftungsausschluss**  
> Dieses Dashboard dient ausschließlich zu privaten Analysezwecken. Es stellt **keine Anlageberatung, keine
> Aufforderung zum Kauf/Verkauf** von Finanzinstrumenten und keine Zusicherung zukünftiger Ergebnisse dar. Alle Daten
> stammen aus einem inoffiziellen Projekt-Setup; Nutzung nur auf eigenes Risiko.
        """
    )
    if not st.checkbox("Ich bestätige, dass ich den Hinweis gelesen habe und trotzdem fortfahre.", value=False):
        st.warning("Bitte bestätige zuerst den Hinweis, um das Dashboard zu verwenden.")
        return

    log_path = str(_default_segment_dir())
    st.sidebar.header("Logquelle")
    st.sidebar.caption("Quelle ist fest auf den Segment-Ordner eingestellt.")
    st.sidebar.code(log_path, language="text")
    st.sidebar.caption("Alle verfügbaren Dateien im Ordner werden automatisch analysiert.")
    force_refresh = st.sidebar.button("Neu laden")
    if force_refresh:
        load_log_entries.clear()

    st.sidebar.markdown("---")
    st.sidebar.caption(f"Angemeldet als: {user_email}")
    st.sidebar.caption(f"Abteilung: {department}")

    if st.session_state.is_admin:
        _render_admin_panel()

    try:
        df = load_log_entries(log_path, None)
    except FileNotFoundError as exc:
        st.error(str(exc))
        return
    except Exception as exc:
        st.exception(exc)
        return

    if df.empty:
        st.info("Noch keine Logeinträge gefunden.")
        return

    df.sort_values("timestamp", inplace=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")

    timestamp_series = df["timestamp"].dropna()
    st.sidebar.header("Filter")
    date_selection = None
    if not timestamp_series.empty:
        min_date = timestamp_series.min().date()
        max_date = timestamp_series.max().date()
        date_selection = st.sidebar.date_input(
            "Zeitraum",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
        )
    symbol_filter = st.sidebar.multiselect(
        "Symbole filtern",
        options=sorted(df["symbol"].dropna().unique()),
        default=[],
    )
    category_filter = st.sidebar.multiselect(
        "Kategorien filtern",
        options=sorted(df["category"].dropna().unique()),
        default=[],
    )

    filtered = df.copy()
    if date_selection:
        if isinstance(date_selection, tuple) or isinstance(date_selection, list):
            if len(date_selection) == 2:
                start_date, end_date = date_selection
            else:
                start_date = end_date = date_selection[0]
        else:
            start_date = end_date = date_selection

        if start_date and end_date:
            start_dt = pd.Timestamp(start_date).tz_localize("UTC")
            end_dt = pd.Timestamp(end_date).tz_localize("UTC") + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
            filtered = filtered[(filtered["timestamp"] >= start_dt) & (filtered["timestamp"] <= end_dt)]

    if symbol_filter:
        filtered = filtered[filtered["symbol"].isin(symbol_filter)]
    if category_filter:
        filtered = filtered[filtered["category"].isin(category_filter)]

    entry_signals = _prepare_entry_signals(filtered)
    segment_dir = Path(log_path)
    segment_count, last_segment_ts = _segment_file_stats(segment_dir)
    cycles = filtered[filtered["category"] == "Cycle"]
    cycle_durations = cycles["cycle_duration"].dropna()
    last_cycle_duration = cycle_durations.iloc[-1] if not cycle_durations.empty else None
    avg_cycle_duration = cycle_durations.mean() if not cycle_durations.empty else None
    last_cycle_time = (
        cycles["timestamp"].dropna().max() if not cycles["timestamp"].dropna().empty else None
    )
    cycles_snapshot = _cycle_snapshot(cycles)
    insight_metrics = _insight_metrics(filtered)
    freshness_label = "unbekannt"
    if last_segment_ts:
        delta = pd.Timestamp.now(tz="UTC") - last_segment_ts
        freshness_label = f"{int(delta.total_seconds() // 60)}m alt"
    st.subheader("Pipeline-Status")
    pipeline_col, status_col = st.columns([2, 1], gap="large")
    with pipeline_col:
        _render_pipeline_hero(
            segment_count,
            last_segment_ts,
            freshness_label,
            last_cycle_time,
            cycles_snapshot,
            insight_metrics,
        )
    status_cards = [
        ("Letzte Logaktualisierung", _format_timestamp(last_segment_ts), freshness_label),
        ("Ø Cycle-Dauer", _format_duration(avg_cycle_duration), "berechnet"),
        ("Letzte Cycle-Dauer", _format_duration(last_cycle_duration), "beendet"),
    ]
    with status_col:
        _render_status_cards(status_cards)
    if last_segment_ts:
        st.caption(f"Logdaten zuletzt aktualisiert vor {freshness_label}.")
    if last_cycle_time is not None:
        st.caption(f"Letzter Cycle endet um {_format_timestamp(last_cycle_time)}.")
    if filtered.empty:
        st.warning("Für die aktuellen Filter liegen keine Einträge vor.")

    overview_tab, insights_tab, signal_tab, raw_tab = st.tabs(["Übersicht", "Insights", "MT5 Signale", "Rohdaten"])

    with overview_tab:
        profit_rejects = (filtered["category"] == "Profit-Faktor").sum()
        broker_blocks = (filtered["category"] == "Broker untersagt").sum()
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Cycles", int(len(cycles)))
        col2.metric("Symbole", int(filtered["symbol"].nunique()))
        col3.metric("Profit-Faktor Rejects", int(profit_rejects))
        total_events = len(filtered)
        block_ratio = f"{(broker_blocks / total_events * 100):.1f}%" if total_events else "0%"
        col4.metric("Broker-Blocks", block_ratio)

        symbol_series = filtered[filtered["symbol"].notna()]["symbol"]
        if symbol_series.empty:
            top_symbol = "-"
            top_symbol_count = "0"
        else:
            most_common = symbol_series.value_counts()
            top_symbol = most_common.idxmax()
            top_symbol_count = int(most_common.max())
        st.metric("Top-Symbol (Skips)", top_symbol, f"{top_symbol_count} Ereignisse")

        exposure_series = filtered["exposure_pct"] if "exposure_pct" in filtered else pd.Series(dtype=float)
        limit_series = filtered["exposure_limit_pct"] if "exposure_limit_pct" in filtered else pd.Series(dtype=float)
        basis_series = filtered["exposure_basis"] if "exposure_basis" in filtered else pd.Series(dtype=str)
        exposure_values = exposure_series.dropna()
        limit_values = limit_series.dropna()
        basis_values = basis_series.dropna()
        st.subheader("Expositionslimits")
        if exposure_values.empty:
            st.write("Keine Expositionslimit-Skips vorhanden.")
        else:
            avg_pct = exposure_values.mean()
            max_pct = exposure_values.max()
            exp_col1, exp_col2 = st.columns(2)
            exp_col1.metric("Ø Exponierung", f"{avg_pct:.2f}% vom Konto")
            exp_col2.metric("Max. Exponierung", f"{max_pct:.2f}% vom Konto", f"{len(exposure_values)} Ereignisse")
            captions: List[str] = []
            if not limit_values.empty:
                captions.append(f"Limit durchschnittlich {limit_values.mean():.2f}% vom Konto")
            if not basis_values.empty:
                mode_basis = basis_values.mode()
                basis_hint = mode_basis.iloc[0] if not mode_basis.empty else basis_values.iloc[0]
                captions.append(f"Basis '{basis_hint}'")
            if captions:
                st.caption(". ".join(captions) + ".")

        st.subheader("Skip-Gründe")
        tracked_categories = {
            "Profit-Faktor",
            "Broker untersagt",
            "Preisschutz",
            "Mock-Daten",
            "Stop-Distanz",
            "Confidence-Filter",
            "Cooldown",
            "Trade-Limit",
            "Symbolinfo",
            "Preisdaten",
        }
        reason_counts = (
            filtered[filtered["category"].isin(tracked_categories)]["category"].value_counts().rename_axis("Kategorie").reset_index(
                name="Anzahl"
            )
        )
        if reason_counts.empty:
            st.write("Keine Skip-Gründe für die Auswahl.")
        else:
            reason_counts = reason_counts.sort_values("Anzahl", ascending=True)
            reason_chart = (
                alt.Chart(reason_counts)
                .mark_bar()
                .encode(
                    x=alt.X("Anzahl:Q", title="Anzahl Skips"),
                    y=alt.Y("Kategorie:N", sort="-x", title="Kategorie"),
                    tooltip=["Kategorie", "Anzahl:Q"],
                    color=alt.Color("Kategorie:N", legend=None),
                )
                .properties(height=260)
            )
            reason_chart = _apply_chart_theme(reason_chart)
            st.markdown('<div class="insight-panel chart-panel">', unsafe_allow_html=True)
            st.altair_chart(reason_chart, width="stretch")
            st.markdown('</div>', unsafe_allow_html=True)

    with insights_tab:
        cycle_stats = filtered[filtered["symbol"] == "cycle"]
        cycle_metrics = cycle_stats[cycle_stats["cycle_signals"].notna()]
        cycle_duration_df = cycle_stats[cycle_stats["cycle_duration"].notna()]
        efficiency_snapshot = _execution_efficiency_snapshot(cycle_metrics)
        efficiency_series = _execution_efficiency_timeseries(cycle_metrics)
        total_entries = len(filtered)
        total_symbols = int(filtered["symbol"].nunique()) if not filtered.empty else 0
        active_categories = int(filtered["category"].nunique()) if not filtered.empty else 0
        cycle_percentage = 0.0
        if total_entries:
            cycle_percentage = (len(cycle_stats) / total_entries) * 100
        last_cycle_signals = efficiency_snapshot.get("signals")
        overview_cols = st.columns(4)
        overview_cols[0].metric("Logeinträge", f"{total_entries}", "nach Filtern")
        overview_cols[1].metric("Aktive Symbole", f"{total_symbols}", "im aktuellen Satz")
        overview_cols[2].metric("Aktive Kategorien", f"{active_categories}")
        overview_cols[3].metric("Cycle-Anteil", f"{cycle_percentage:.0f}%", f"Letzte Signale: {last_cycle_signals or '-'}")
        st.subheader("Skips im Zeitverlauf")
        timeline_source = filtered[filtered["timestamp"].notna()].copy()
        detail_source: pd.DataFrame
        if timeline_source.empty:
            st.write("Keine Zeitinformationen vorhanden.")
            detail_source = pd.DataFrame()
        else:
            hours_back = st.slider(
                "Blick zurück (Stunden)",
                min_value=6,
                max_value=168,
                value=48,
                step=6,
                key="insights_hours",
            )
            window_start = pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=hours_back)
            detail_source = timeline_source[timeline_source["timestamp"] >= window_start]
            if detail_source.empty:
                st.warning("Für den gewählten Zeitraum liegen keine Einträge vor.")
            else:
                span = detail_source["timestamp"].max() - detail_source["timestamp"].min()
                freq = _timeline_frequency(span)
                timeline = (
                    detail_source.set_index("timestamp")
                    .assign(count=1)
                    .groupby([pd.Grouper(freq=freq), "category"])
                    .size()
                    .reset_index(name="Anzahl")
                )
                chart = (
                    alt.Chart(timeline)
                    .mark_line(point=True)
                    .encode(
                        x=alt.X("timestamp:T", title="Zeit"),
                        y=alt.Y("Anzahl:Q", title="Skip-Anzahl"),
                        color=alt.Color("category:N", title="Kategorie"),
                        tooltip=["timestamp:T", "category:N", "Anzahl:Q"],
                    )
                    .properties(height=320)
                )
                chart = _apply_chart_theme(chart)
                st.markdown('<div class="insight-panel chart-panel">', unsafe_allow_html=True)
                st.altair_chart(chart, width="stretch")
                st.markdown('</div>', unsafe_allow_html=True)

        st.subheader("Stündliche Skip-Verteilung")
        if detail_source.empty:
            st.write("Nicht genug Zeitdaten für die stündliche Auswertung.")
        else:
            hourly = (
                detail_source.assign(hour=detail_source["timestamp"].dt.hour)
                .groupby("hour")
                .size()
                .reset_index(name="Anzahl")
            )
            hourly_chart = (
                alt.Chart(hourly)
                .mark_bar()
                .encode(
                    x=alt.X("hour:O", title="Stunde"),
                    y=alt.Y("Anzahl:Q", title="Anzahl Skips"),
                    tooltip=["hour:O", "Anzahl:Q"],
                )
                .properties(height=260)
            )
            hourly_chart = _apply_chart_theme(hourly_chart)
            st.markdown('<div class="insight-panel chart-panel">', unsafe_allow_html=True)
            st.altair_chart(hourly_chart, width="stretch")
            st.markdown('</div>', unsafe_allow_html=True)

        st.subheader("Execution Efficiency")
        if cycle_metrics.empty:
            st.write("Keine Zyklus-Metriken für Validierungen/Executions vorhanden.")
        else:
            summary_cards = f"""
            <div class="insight-summary">
                <div class="insight-summary-card">
                    <span>Signale pro Cycle</span>
                    <strong>{efficiency_snapshot.get('signals') or '-'}</strong>
                </div>
                <div class="insight-summary-card">
                    <span>Validierungsrate</span>
                    <strong>{_format_rate(efficiency_snapshot.get('validation_rate'))}</strong>
                </div>
                <div class="insight-summary-card">
                    <span>Executionrate</span>
                    <strong>{_format_rate(efficiency_snapshot.get('execution_rate'))}</strong>
                </div>
                <div class="insight-summary-card">
                    <span>Duplikate</span>
                    <strong>{_format_rate(efficiency_snapshot.get('duplicate_rate'))}</strong>
                </div>
            </div>
            """
            st.markdown(summary_cards, unsafe_allow_html=True)
            melted = (
                efficiency_series
                .melt(
                    id_vars="timestamp",
                    value_vars=["cycle_validated", "cycle_executed", "cycle_duplicates"],
                    var_name="Metric",
                    value_name="Anzahl",
                )
                .query("Anzahl > 0")
            )
            if not melted.empty:
                efficiency_chart = (
                    alt.Chart(melted)
                    .mark_line(point=True)
                    .encode(
                        x=alt.X("timestamp:T", title="Zeit"),
                        y=alt.Y("Anzahl:Q", title="Zahl der Signale"),
                        color=alt.Color("Metric:N", title="Metric"),
                        tooltip=["timestamp:T", "Metric:N", "Anzahl:Q"],
                    )
                    .properties(height=320)
                )
                efficiency_chart = _apply_chart_theme(efficiency_chart)
                st.markdown('<div class="insight-panel chart-panel">', unsafe_allow_html=True)
                st.altair_chart(efficiency_chart, width="stretch")
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.write("Keine Trenddaten für die Execution-Metriken verfügbar.")
            rate_trends = efficiency_series.copy()
            rate_trends = rate_trends[rate_trends["cycle_signals"] > 0]
            rate_trends = rate_trends.assign(
                validation_rate=rate_trends["cycle_validated"] / rate_trends["cycle_signals"],
                execution_rate=rate_trends["cycle_executed"] / rate_trends["cycle_validated"].replace(0, 1),
                duplicate_rate=rate_trends["cycle_duplicates"] / rate_trends["cycle_signals"],
            )
            rate_melted = (
                rate_trends
                .melt(
                    id_vars="timestamp",
                    value_vars=["validation_rate", "execution_rate", "duplicate_rate"],
                    var_name="Rate",
                    value_name="Wert",
                )
                .dropna()
            )
            if not rate_melted.empty:
                rate_chart = (
                    alt.Chart(rate_melted)
                    .mark_line(point=True)
                    .encode(
                        x=alt.X("timestamp:T", title="Zeit"),
                        y=alt.Y("Wert:Q", title="Quote", axis=alt.Axis(format=".0%")),
                        color=alt.Color("Rate:N", title="Quote"),
                        tooltip=["timestamp:T", "Rate:N", alt.Tooltip("Wert:Q", format=".2f")],
                    )
                    .properties(height=280)
                )
                rate_chart = _apply_chart_theme(rate_chart)
                st.markdown('<div class="insight-panel chart-panel">', unsafe_allow_html=True)
                st.altair_chart(rate_chart, width="stretch")
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.write("Keine Rate-Daten verfügbar.")

        st.subheader("Exposure-Verlauf")
        exposure_timeline = filtered[filtered["exposure_pct"].notna() & filtered["timestamp"].notna()].copy()
        if exposure_timeline.empty:
            st.write("Keine Exposure-Daten aus den Logs verfügbar.")
        else:
            exposure_timeline = exposure_timeline.sort_values("timestamp")
            exposure_chart = (
                alt.Chart(exposure_timeline)
                .mark_line(point=True)
                .encode(
                    x=alt.X("timestamp:T", title="Zeit"),
                    y=alt.Y("exposure_pct:Q", title="Exponierung (% vom Konto)"),
                    color=alt.Color("symbol:N", title="Symbol", legend=alt.Legend(orient="bottom")),
                    tooltip=["timestamp:T", alt.Tooltip("exposure_pct:Q", format=".2f"), "symbol"],
                )
                .properties(height=250)
            )
            exposure_chart = _apply_chart_theme(exposure_chart)
            st.markdown('<div class="insight-panel chart-panel">', unsafe_allow_html=True)
            st.altair_chart(exposure_chart, width="stretch")
            st.markdown('</div>', unsafe_allow_html=True)
            latest_exposure = exposure_timeline.iloc[-1]
            exp_cols = st.columns(3)
            exp_balance = latest_exposure.get("balance")
            exp_exposure = latest_exposure.get("exposure")
            exp_drawdown = latest_exposure.get("drawdown")
            exp_cols[0].metric("Letzte Balance", f"{exp_balance:.2f}" if pd.notna(exp_balance) else "-")
            exp_cols[1].metric("Exponierung", f"{exp_exposure:.2f}" if pd.notna(exp_exposure) else "-", f"{latest_exposure.get('exposure_pct'):.2f}%" if pd.notna(latest_exposure.get('exposure_pct')) else "")
            exp_cols[2].metric("Drawdown", f"{exp_drawdown:.2f}%" if pd.notna(exp_drawdown) else "-")

        st.subheader("Confidence vs. Price Gap")
        correlation_df = filtered[(filtered["confidence_gap"].notna()) & (filtered["price_gap"].notna())]
        if correlation_df.empty:
            st.write("Keine ausreichenden Confidence/Price Gap-Daten verfügbar.")
        else:
            correlation_chart = (
                alt.Chart(correlation_df)
                .mark_circle(size=70, opacity=0.7)
                .encode(
                    x=alt.X("confidence_gap:Q", title="Confidence Gap"),
                    y=alt.Y("price_gap:Q", title="Price Gap"),
                    color=alt.Color("category:N", title="Kategorie"),
                    tooltip=["symbol", "category", alt.Tooltip("confidence_gap:Q", format=".3f"), alt.Tooltip("price_gap:Q", format=".3f")],
                )
                .properties(height=320)
            )
            correlation_chart = _apply_chart_theme(correlation_chart)
            st.markdown('<div class="insight-panel chart-panel">', unsafe_allow_html=True)
            st.altair_chart(correlation_chart, width="stretch")
            st.markdown('</div>', unsafe_allow_html=True)

        st.subheader("Kategorie-Anteile")
        category_counts = filtered[filtered["category"].notna()]["category"].fillna("Unbekannt").astype(str)
        if category_counts.empty:
            st.write("Keine Kategorien für diese Auswahl verfügbar.")
        else:
            cat_summary = (
                category_counts.value_counts().rename_axis("Kategorie").reset_index(name="Anzahl")
            )
            cat_summary["Kategorie"] = cat_summary["Kategorie"].astype(str)
            cat_chart = (
                alt.Chart(cat_summary)
                .mark_arc(innerRadius=40)
                .encode(
                    theta=alt.Theta("Anzahl:Q", title="Anteil"),
                    color=alt.Color("Kategorie:N", title="Kategorie"),
                    tooltip=["Kategorie", "Anzahl:Q"],
                )
                .properties(height=320)
            )
            cat_chart = _apply_chart_theme(cat_chart)
            st.markdown('<div class="insight-panel chart-panel">', unsafe_allow_html=True)
            st.altair_chart(cat_chart, width="stretch")
            st.markdown('</div>', unsafe_allow_html=True)

        st.subheader("Cycle-Dauer")
        if cycle_duration_df.empty:
            st.write("Keine Cycle-Daten mit Dauerangabe verfügbar.")
        else:
            cycle_chart = (
                alt.Chart(cycle_duration_df)
                .mark_line(point=True)
                .encode(
                    x=alt.X("timestamp:T", title="Zeit"),
                    y=alt.Y("cycle_duration:Q", title="Sekunden"),
                    tooltip=["timestamp:T", alt.Tooltip("cycle_duration:Q", title="Sekunden", format=".2f")],
                )
                .properties(height=250)
            )
            cycle_chart = _apply_chart_theme(cycle_chart)
            st.altair_chart(cycle_chart, width="stretch")

        st.subheader("Top-Symbole nach Skips")
        with st.expander("Top-Symbole und Kategorien anzeigen", expanded=True):
            top_symbols = (
                filtered.groupby(["symbol", "category"])
                .size()
                .reset_index(name="Anzahl")
                .sort_values("Anzahl", ascending=False)
                .head(15)
            )
            if top_symbols.empty:
                st.write("Keine Daten für die aktuelle Auswahl.")
            else:
                st.dataframe(top_symbols, width="stretch", hide_index=True)

        st.subheader("Symbol/Kategorie Heatmap")
        heat_limit = st.slider("Heatmap - max. Einträge", min_value=50, max_value=400, value=200, step=50, key="heatmap_limit")
        heat_data = (
            filtered.groupby(["symbol", "category"]).size().reset_index(name="Anzahl").sort_values("Anzahl", ascending=False).head(heat_limit)
        )
        if heat_data.empty:
            st.write("Keine Daten verfügbar.")
        else:
            heat_chart = (
                alt.Chart(heat_data)
                .mark_rect()
                .encode(
                    x=alt.X("symbol:N", title="Symbol", sort="-y"),
                    y=alt.Y("category:N", title="Kategorie"),
                    color=alt.Color("Anzahl:Q", title="Anzahl"),
                    tooltip=["symbol", "category", "Anzahl"],
                )
                .properties(height=320)
            )
            heat_chart = _apply_chart_theme(heat_chart)
            st.altair_chart(heat_chart, width="stretch")

        st.subheader("Profit-Faktor Insights")
        profit_df = filtered[(filtered["category"] == "Profit-Faktor") & filtered["cr_ratio"].notna()]
        if profit_df.empty:
            st.write("Keine Profit-Faktor-Einträge vorhanden.")
        else:
            avg_cr = profit_df["cr_ratio"].mean()
            worst_cr = profit_df["cr_ratio"].min()
            below_threshold = (profit_df["min_factor"].notna() & (profit_df["cr_ratio"] < profit_df["min_factor"])).sum()
            c1, c2, c3 = st.columns(3)
            c1.metric("Ø Chance/Risiko", f"{avg_cr:.2f}")
            c2.metric("Schlechtester Wert", f"{worst_cr:.2f}")
            c3.metric("Unter Mindestfaktor", int(below_threshold))

            hist = (
                alt.Chart(profit_df)
                .mark_bar()
                .encode(
                    x=alt.X("cr_ratio:Q", bin=alt.Bin(maxbins=25), title="Chance/Risiko"),
                    y=alt.Y("count()", title="Anzahl"),
                    tooltip=["count()"],
                )
                .properties(height=250)
            )
            hist = _apply_chart_theme(hist)
            st.altair_chart(hist, width="stretch")

        st.subheader("Confidence-Filter")
        confidence_df = filtered[filtered["confidence_gap"].notna()]
        if confidence_df.empty:
            st.write("Keine Confidence-Filter gefunden.")
        else:
            avg_gap = confidence_df["confidence_gap"].mean()
            max_gap = confidence_df["confidence_gap"].max()
            c1, c2 = st.columns(2)
            c1.metric("Ø Abstand zum Threshold", f"{avg_gap:.3f}")
            c2.metric("Max. Abstand", f"{max_gap:.3f}")

        st.subheader("Cooldown-Dauer")
        cooldown_df = filtered[filtered["cooldown_minutes"].notna()]
        if cooldown_df.empty:
            st.write("Keine aktiven Cooldowns in den Daten.")
        else:
            avg_cd = cooldown_df["cooldown_minutes"].mean()
            st.metric("Ø verbleibende Minuten", f"{avg_cd:.1f}")

        st.subheader("Stop-Distanz Puffer")
        stop_df = filtered[filtered["stop_shortfall"].notna()]
        if stop_df.empty:
            st.write("Keine Stop-Distanz-Verletzungen.")
        else:
            mean_shortfall = stop_df["stop_shortfall"].mean()
            st.metric("Ø fehlender Abstand", f"{mean_shortfall:.5f}")

        st.subheader("Preisschutz-Distanz")
        price_df = filtered[(filtered["category"] == "Preisschutz") & filtered["price_gap"].notna()]
        if price_df.empty:
            st.write("Keine Preisschutz-Signale erkannt.")
        else:
            avg_gap = price_df["price_gap"].abs().mean()
            st.metric("Ø Distanz (Punkte)", f"{avg_gap:.2f}")
            relation_counts = (
                price_df["price_relation"].fillna("Unbekannt").value_counts().rename_axis("Barriere").reset_index(name="Anzahl")
            )
            st.bar_chart(relation_counts, x="Barriere", y="Anzahl")

    with signal_tab:
        st.subheader("MT5 Entry-Signale")
        st.caption("Die folgenden Signale werden direkt aus den MT5-Logs über LastEntry=EntrySignal gewonnen.")
        if entry_signals.empty:
            st.info("Noch keine MT5-Entry-Signale im aktuellen Filter.")
        else:
            symbols = ["Alle"] + sorted(entry_signals["symbol"].dropna().unique())
            selected_symbol = st.selectbox("Symbol auswählen", symbols)
            directions = sorted(entry_signals["direction"].dropna().unique())
            direction_filter = st.multiselect("Richtung", directions, default=directions)
            max_limit = max(5, len(entry_signals))
            limit = st.slider(
                "Anzahl der Einträge",
                min_value=1,
                max_value=max_limit,
                value=min(max_limit, 20),
                step=1,
                key="signal_limit",
            )
            chart_source = entry_signals
            if selected_symbol != "Alle":
                chart_source = chart_source[chart_source["symbol"] == selected_symbol]
            if direction_filter:
                chart_source = chart_source[chart_source["direction"].isin(direction_filter)]
            chart_source = chart_source.sort_values("entry_time")
            chart_source = chart_source.tail(limit)
            if chart_source.empty:
                st.warning("Keine Signale für die aktuelle Kombination.")
            else:
                melt = (
                    chart_source[
                        ["entry_time", "entry_price", "stop_loss", "take_profit"]
                    ]
                    .melt(id_vars="entry_time", var_name="type", value_name="price")
                    .dropna()
                )
                signal_chart = (
                    alt.Chart(melt)
                    .mark_line(point=True)
                    .encode(
                        x=alt.X("entry_time:T", title="Entry-Zeit"),
                        y=alt.Y("price:Q", title="Preis"),
                        color=alt.Color("type:N", title="Typ"),
                        tooltip=["type:N", "price:Q", "entry_time:T"],
                    )
                    .properties(height=360)
                    .interactive()
                )
                signal_chart = _apply_chart_theme(signal_chart)
                st.markdown('<div class="insight-panel chart-panel">', unsafe_allow_html=True)
                st.altair_chart(signal_chart, width="stretch")
                st.markdown('</div>', unsafe_allow_html=True)
                metric_cols = st.columns(3)
                avg_entry = chart_source["entry_price"].mean()
                avg_stop = chart_source["stop_loss"].mean()
                avg_tp = chart_source["take_profit"].mean()
                metric_cols[0].metric("Ø Entry-Preis", f"{avg_entry:.2f}" if pd.notna(avg_entry) else "-")
                metric_cols[1].metric("Ø Stop-Loss", f"{avg_stop:.2f}" if pd.notna(avg_stop) else "-")
                metric_cols[2].metric("Ø Take-Profit", f"{avg_tp:.2f}" if pd.notna(avg_tp) else "-")
                st.subheader("Letzte Signale")
                display_df = chart_source.copy()
                display_df["entry_time"] = display_df["entry_time"].dt.strftime("%Y-%m-%d %H:%M")
                st.dataframe(
                    display_df[
                        ["symbol", "direction", "entry_time", "entry_price", "stop_loss", "take_profit", "confidence"]
                    ]
                    .rename(columns={
                        "symbol": "Symbol",
                        "direction": "Richtung",
                        "entry_time": "Entry-Zeit",
                        "entry_price": "Entry",
                        "stop_loss": "Stop",
                        "take_profit": "TP",
                        "confidence": "Confidence",
                    }),
                    hide_index=True,
                )
                st.caption("Nutze diese Übersicht, um genau zu sehen, welche MT5-Signale zuletzt berechnet wurden und welche Preisziele der Bot im Blick hatte.")

    with raw_tab:
        st.subheader("Letzte Ereignisse")
        columns = ["timestamp", "symbol", "category", "message"]
        if "source" in filtered.columns:
            columns.append("source")
        table = filtered.tail(500)[columns].rename(
            columns={
                "timestamp": "Zeit",
                "symbol": "Symbol",
                "category": "Kategorie",
                "message": "Nachricht",
                "source": "Datei",
            }
        )
        st.dataframe(table, width="stretch", hide_index=True)
        csv_data = table.to_csv(index=False).encode("utf-8")
        st.download_button("CSV exportieren", csv_data, file_name="ew_live_logs.csv", mime="text/csv")

    st.caption(
        "Die Daten stammen direkt aus den Logdateien. Nutze Filter und Insights, um Muster zu erkennen und die Handelslogik anzupassen."
    )
    st.caption("Tipp: sync_segments.py kopiert die neuesten VPS-Segmente in den lokalen logs/segments-Ordner.")


if __name__ == "__main__":
    main()
