"""Hilfsskript, um segmentierte Logs zwischen VPS und lokalem Dashboard zu synchronisieren."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

DEFAULT_SOURCE = Path(r"C:\Users\Administrator\Documents\EW-Livev2.0\logs\segments")
DEFAULT_DEST = Path(__file__).resolve().parent / "logs" / "segments"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Synchronisiere Log-Segmente in einen lokalen Ordner")
    parser.add_argument("--source", default=str(DEFAULT_SOURCE), help="Quellordner mit Logsegmenten")
    parser.add_argument(
        "--destination",
        default=str(DEFAULT_DEST),
        help="Zielordner für lokale Auswertung (Standard: logs/segments)",
    )
    parser.add_argument("--max-files", type=int, default=200, help="Maximal zu kopierende Dateien (neueste zuerst)")
    parser.add_argument(
        "--prune",
        action="store_true",
        help="Nicht mehr vorhandene Dateien im Ziel entfernen, damit beide Verzeichnisse identisch bleiben",
    )
    return parser.parse_args()


def sync_segments(source: Path, destination: Path, max_files: int, prune: bool) -> None:
    if not source.exists():
        raise FileNotFoundError(f"Quellordner nicht gefunden: {source}")
    destination.mkdir(parents=True, exist_ok=True)

    files = sorted(p for p in source.glob("*.txt") if p.is_file())
    if max_files > 0:
        files = files[-max_files:]

    copied = 0
    for file in files:
        target = destination / file.name
        shutil.copy2(file, target)
        copied += 1

    if prune:
        keep = {f.name for f in files}
        for file in destination.glob("*.txt"):
            if file.name not in keep:
                file.unlink()

    print(f"Kopiert: {copied} Dateien → {destination}")
    if prune:
        print("Pruning aktiviert – Zielordner bereinigt.")


def main() -> None:
    args = parse_args()
    sync_segments(Path(args.source), Path(args.destination), args.max_files, args.prune)


if __name__ == "__main__":
    main()
