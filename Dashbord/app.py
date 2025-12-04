from __future__ import annotations

import hashlib
import io
import re
import secrets
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import altair as alt
import pandas as pd
import pyotp
import qrcode
import streamlit as st

# Configure Streamlit page early so the layout matches the dashboard use case.
st.set_page_config(page_title="EW Live Dashboard", layout="wide")

DEFAULT_LOG = Path(__file__).resolve().parents[1] / "logs" / "live_execution.txt"
REMOTE_SEGMENT_DIR = Path(r"C:\Users\Administrator\Documents\EW-Livev2.1\logs\segments")
LOCAL_SEGMENT_DIR = Path(__file__).resolve().parents[1] / "logs" / "segments"
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
    r"Exponierung\s+[0-9.,]+\s+>\s+Limit\s+[0-9.,]+\s+\(max\s+([0-9.,]+)%\s+vom\s+Konto,\s+aktuell\s+([0-9.,]+)%\s+vom\s+Konto\)"
)
COOLDOWN_PATTERN = re.compile(r"Cooldown aktiv\s+\(([0-9.,]+)m verbleibend\)")
CYCLE_DURATION_PATTERN = re.compile(r"Dauer=([0-9.,]+)s")
ENTRY_SIGNAL_PATTERN = re.compile(r"LastEntry=EntrySignal\((?P<payload>[^)]*)\)")
ENTRY_TIME_PATTERN = re.compile(r"entry_time=Timestamp\('(?P<entry_time>[^']+)'\)")
ENTRY_DIRECTION_PATTERN = re.compile(r"direction=<Dir\.\w+:\s*'(?P<direction>[A-Z]+)'\>")
ENTRY_ZONE_PATTERN = re.compile(r"entry_zone=\((?P<low>[0-9.,+-]+),\s*(?P<high>[0-9.,+-]+)\)")


def _default_segment_dir() -> Path:
    try:
        REMOTE_SEGMENT_DIR.mkdir(parents=True, exist_ok=True)
        return REMOTE_SEGMENT_DIR
    except (OSError, PermissionError):
        LOCAL_SEGMENT_DIR.mkdir(parents=True, exist_ok=True)
        return LOCAL_SEGMENT_DIR


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
    _create_user(EMERGENCY_EMAIL, EMERGENCY_PASSWORD, is_admin=True)


def _verify_credentials(email: str, password: str, totp_code: str) -> tuple[bool, str, Optional[sqlite3.Row]]:
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

    exposure_match = EXPOSURE_PATTERN.search(message)
    if exposure_match:
        limit_pct = _safe_float(exposure_match.group(1))
        actual_pct = _safe_float(exposure_match.group(2))
        if limit_pct is not None:
            data["exposure_limit_pct"] = limit_pct
        if actual_pct is not None:
            data["exposure_pct"] = actual_pct

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

    with st.form("login_form"):
        email = st.text_input("E-Mail-Adresse")
        password = st.text_input("Passwort", type="password")
        totp_code = st.text_input("2FA-Code", max_chars=6)
        submitted = st.form_submit_button("Anmelden")

    if submitted:
        success, message, row = _verify_credentials(email, password, totp_code)
        if success and row:
            st.session_state.authenticated = True
            st.session_state.user_email = email
            st.session_state.is_admin = bool(row["is_admin"])
            st.session_state.login_error = ""
            _rerun_app()
        else:
            st.session_state.login_error = message

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

    if not st.session_state.authenticated:
        _render_login_screen(st.session_state.get("initial_totp_secret"))
        return

    if st.sidebar.button("Abmelden"):
        for key in ("authenticated", "user_email", "is_admin", "login_error"):
            st.session_state[key] = False if key == "authenticated" else "" if key != "is_admin" else False
        _rerun_app()

    st.title("EW Live – Monitoring Dashboard")
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

    if filtered.empty:
        st.warning("Für die aktuellen Filter liegen keine Einträge vor.")

    overview_tab, insights_tab, signal_tab, raw_tab = st.tabs(["Übersicht", "Insights", "MT5 Signale", "Rohdaten"])

    with overview_tab:
        cycles = filtered[filtered["category"] == "Cycle"]
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
        exposure_values = exposure_series.dropna()
        limit_values = limit_series.dropna()
        st.subheader("Expositionslimits")
        if exposure_values.empty:
            st.write("Keine Expositionslimit-Skips vorhanden.")
        else:
            avg_pct = exposure_values.mean()
            max_pct = exposure_values.max()
            limit_pct = limit_values.mean() if not limit_values.empty else None
            if limit_pct is not None:
                avg_display = min(avg_pct, limit_pct)
                max_display = min(max_pct, limit_pct)
            else:
                avg_display = avg_pct
                max_display = max_pct
            exp_col1, exp_col2 = st.columns(2)
            exp_col1.metric("Ø Exponierung", f"{avg_display:.2f}% vom Konto")
            exp_col2.metric("Max. Exponierung", f"{max_display:.2f}% vom Konto", f"{len(exposure_values)} Ereignisse")
            if limit_pct is not None:
                st.caption(f"Limit durchschnittlich {limit_pct:.2f}% vom Konto. Die dargestellten Werte werden nicht über dem Limit angezeigt (tatsächliche Werte siehe Logs).")

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
            st.bar_chart(reason_counts, x="Kategorie", y="Anzahl")

    with insights_tab:
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
                st.markdown('<div class="animated-panel chart-panel">', unsafe_allow_html=True)
                st.altair_chart(chart, use_container_width=True)
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
            st.markdown('<div class="animated-panel chart-panel">', unsafe_allow_html=True)
            st.altair_chart(hourly_chart, use_container_width=True)
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
            st.markdown('<div class="animated-panel chart-panel">', unsafe_allow_html=True)
            st.altair_chart(cat_chart, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        st.subheader("Cycle-Dauer")
        cycle_df = filtered[(filtered["symbol"] == "cycle") & filtered["cycle_duration"].notna()]
        if cycle_df.empty:
            st.write("Keine Cycle-Daten mit Dauerangabe verfügbar.")
        else:
            cycle_chart = (
                alt.Chart(cycle_df)
                .mark_line(point=True)
                .encode(
                    x=alt.X("timestamp:T", title="Zeit"),
                    y=alt.Y("cycle_duration:Q", title="Sekunden"),
                    tooltip=["timestamp:T", alt.Tooltip("cycle_duration:Q", title="Sekunden", format=".2f")],
                )
                .properties(height=250)
            )
            st.altair_chart(cycle_chart, use_container_width=True)

        st.subheader("Top-Symbole nach Skips")
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
        heat_data = (
            filtered.groupby(["symbol", "category"]).size().reset_index(name="Anzahl").sort_values("Anzahl", ascending=False).head(200)
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
            st.altair_chart(heat_chart, use_container_width=True)

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
            st.altair_chart(hist, use_container_width=True)

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
                st.markdown('<div class="animated-panel chart-panel">', unsafe_allow_html=True)
                st.altair_chart(signal_chart, use_container_width=True)
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
