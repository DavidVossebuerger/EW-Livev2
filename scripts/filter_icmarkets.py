"""Filter ICMarkets symbol export by tradability and maximum spread."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable, List, Tuple


def _parse_bool(value: str) -> bool:
    text = (value or "").strip().lower()
    return text in {"ja", "yes", "true", "1"}


def _parse_float(value: str) -> float:
    text = (value or "").strip().replace(",", ".")
    return float(text) if text else 0.0


def _read_header(first_line: str) -> list[str]:
    cleaned = first_line.strip().lstrip("\ufeff")
    if cleaned.startswith("\""):
        parts = cleaned.split("\"")
        if len(parts) >= 2:
            cleaned = parts[1]
    if cleaned:
        return [part.strip() for part in cleaned.split(";") if part.strip()]
    return []


def _filter_rows(rows: Iterable[dict[str, str]], max_spread: float) -> list[dict[str, str]]:
    filtered: list[dict[str, str]] = []
    for row in rows:
        if not row:
            continue
        tradable = _parse_bool(row.get("Handel_erlaubt", ""))
        if not tradable:
            continue
        spread_value = _parse_float(row.get("Spread", "0"))
        if spread_value <= max_spread and not _is_etf(row):
            filtered.append(row)
    return filtered


def _is_etf(row: dict[str, str]) -> bool:
    group = (row.get("Gruppe") or "").upper()
    description = (row.get("Beschreibung") or "").upper()
    symbol = (row.get("Symbol") or "").upper()
    return "ETF" in group or "ETF" in description or symbol.endswith(".ETF")


def _load_rows(path: Path, preferred_encoding: str | None) -> Tuple[List[str], List[dict[str, str]]]:
    encodings = [preferred_encoding] if preferred_encoding else []
    encodings += ["utf-8-sig", "cp1252", "latin-1"]
    last_error: UnicodeDecodeError | None = None
    for enc in encodings:
        if not enc:
            continue
        try:
            with path.open("r", encoding=enc) as fh:
                first_line = fh.readline()
                headers = _read_header(first_line)
                reader = csv.DictReader(fh, fieldnames=headers, delimiter=";")
                return headers, list(reader)
        except UnicodeDecodeError as exc:
            last_error = exc
            continue
    raise last_error if last_error else RuntimeError("CSV konnte nicht gelesen werden")


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter ICMarkets CSV export.")
    parser.add_argument("input", type=Path, help="Pfad zur ICMarkets CSV-Datei")
    parser.add_argument("output", type=Path, help="Pfad zur Ausgabedatei")
    parser.add_argument("--max-spread", type=float, default=5.0, help="Maximal erlaubter Spread")
    parser.add_argument("--encoding", help="Explizites Encoding der Eingabedatei")
    args = parser.parse_args()

    headers, rows = _load_rows(args.input, args.encoding)
    filtered = _filter_rows(rows, args.max_spread)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=headers, delimiter=";")
        writer.writeheader()
        writer.writerows(filtered)

    print(f"Gefilterte Symbole: {len(filtered)} gespeichert in {args.output}")


if __name__ == "__main__":
    main()
