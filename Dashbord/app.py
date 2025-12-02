from __future__ import annotations

import hashlib
import re
import secrets
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import altair as alt
import pandas as pd
import pyotp
import streamlit as st

# Configure Streamlit page early so the layout matches the dashboard use case.
st.set_page_config(page_title="EW Live Dashboard", layout="wide")

DEFAULT_LOG = Path(__file__).resolve().parents[1] / "logs" / "live_execution.txt"
REMOTE_SEGMENT_DIR = Path(r"C:\Users\Administrator\Documents\EW-Livev2.1\logs\segments")
LOCAL_SEGMENT_DIR = Path(__file__).resolve().parents[1] / "logs" / "segments"
AUTH_DB_PATH = Path(__file__).resolve().parents[1] / "auth.db"
DEFAULT_ADMIN_EMAIL = "vossebuerger@fmmuc.com"
DEFAULT_ADMIN_PASSWORD = "mimiKatze1!"
TIMESTAMP_OFFSET_PATTERN = re.compile(r"([+-]\d{2})(\d{2})$")
CR_PATTERN = re.compile(r"Chance/Risiko\s+([0-9.,]+)")
MIN_FACTOR_PATTERN = re.compile(r"Mindestfaktor\s+([0-9.,]+)")
PRICE_GUARD_PATTERN = re.compile(
    r"aktueller Preis\s+([0-9.,]+)\s+(<=|>=)\s+(Stop|TP)\s+([0-9.,]+)",
    re.IGNORECASE,
)
STOP_DIST_PATTERN = re.compile(r"Stop-Distanz\s+([0-9.,]+)\s+<\s+Mindestabstand\s+([0-9.,]+)")
CONFIDENCE_PATTERN = re.compile(r"Confidence\s+([0-9.,]+)\s+<\s+Threshold\s+([0-9.,]+)")
COOLDOWN_PATTERN = re.compile(r"Cooldown aktiv\s+\(([0-9.,]+)m verbleibend\)")
CYCLE_DURATION_PATTERN = re.compile(r"Dauer=([0-9.,]+)s")


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


def _create_user(email: str, password: str, is_admin: bool = False) -> str:
    salt = secrets.token_bytes(16)
    password_hash = _hash_password(password, salt)
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
    return _create_user(DEFAULT_ADMIN_EMAIL, DEFAULT_ADMIN_PASSWORD, is_admin=True)


def _verify_credentials(email: str, password: str, totp_code: str) -> tuple[bool, str, Optional[sqlite3.Row]]:
    row = _get_user(email)
    if not row:
        return False, "Benutzer nicht gefunden.", None

    salt = bytes.fromhex(row["password_salt"])
    candidate_hash = _hash_password(password, salt)
    if not secrets.compare_digest(candidate_hash, row["password_hash"]):
        return False, "Ungültiges Passwort.", None

    totp = pyotp.TOTP(row["totp_secret"])
    if not totp.verify(totp_code, valid_window=1):
        return False, "Ungültiger 2FA-Code.", None

    return True, "", row


def _format_totp_info(email: str, secret: str) -> str:
    totp = pyotp.TOTP(secret)
    uri = totp.provisioning_uri(email, issuer_name="EW Live Dashboard")
    return f"TOTP-Secret: {secret}\nProvisioning URI: {uri}"


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

    return data


def _safe_float(value: str) -> Optional[float]:
    try:
        return float(value.replace(",", "."))
    except (AttributeError, ValueError):
        return None


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
        st.session_state.initial_totp_secret_shown = True

    with st.form("login_form"):
        email = st.text_input("E-Mail-Adresse", value=DEFAULT_ADMIN_EMAIL)
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
            st.experimental_rerun()
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

        if users:
            st.markdown("---")
            selected_user = st.selectbox("2FA zurücksetzen", options=[user["email"] for user in users], key="reset_user")
            if st.button("TOTP regenerieren", key="reset_totp"):
                new_secret = _reset_totp_secret(selected_user)
                st.success("Neues TOTP-Secret erstellt.")
                st.code(_format_totp_info(selected_user, new_secret), language="text")


def main() -> None:
    initial_totp_secret = _ensure_default_admin()
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
        st.experimental_rerun()

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

    if filtered.empty:
        st.warning("Für die aktuellen Filter liegen keine Einträge vor.")

    overview_tab, insights_tab, raw_tab = st.tabs(["Übersicht", "Insights", "Rohdaten"])

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
        if timeline_source.empty:
            st.write("Keine Zeitinformationen vorhanden.")
        else:
            span = timeline_source["timestamp"].max() - timeline_source["timestamp"].min()
            freq = _timeline_frequency(span)
            timeline = (
                timeline_source.set_index("timestamp")
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
            st.altair_chart(chart, use_container_width=True)

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


def _render_login_screen(initial_secret: Optional[str]) -> None:
    st.header("Gesicherter Zugriff")
    st.caption("Bitte melde dich mit deinem @fmmuc.com-Account, Passwort und TOTP-Code an.")
    if initial_secret and not st.session_state.get("initial_totp_secret_shown"):
        st.warning(
            "Initialer TOTP-Secret (nur einmal anzeigen):\n" + _format_totp_info(DEFAULT_ADMIN_EMAIL, initial_secret)
        )
        st.session_state.initial_totp_secret_shown = True

    with st.form("login_form"):
        email = st.text_input("E-Mail-Adresse", value=DEFAULT_ADMIN_EMAIL)
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
            st.experimental_rerun()
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

        if users:
            st.markdown("---")
            selected_user = st.selectbox("2FA zurücksetzen", options=[user["email"] for user in users], key="reset_user")
            if st.button("TOTP regenerieren", key="reset_totp"):
                new_secret = _reset_totp_secret(selected_user)
                st.success("Neues TOTP-Secret erstellt.")
                st.code(_format_totp_info(selected_user, new_secret), language="text")
