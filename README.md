# EW Live-System (MT5 + Vantage)

Dieses Repo enthält das Grundgerüst, um deine Elliott-Wave-Strategie ohne EMA/ML direkt auf MT5 (Vantage) als Live-System zu betreiben.

## Architektur

- `live_core/config.py`: Zustandsloser Config-Container mit Lade-Hooks (JSON + Env).
- `live_core/signals.py`: Regelbasierte Signalgenerierung über vereinfachte ZigZag-Impulse, ganz ohne ML.
- `live_core/mt5_adapter.py`: Adapter zu MetaTrader5 inklusive Mock-Modus für lokale Tests.
- `live_core/execution.py`: OrderManager, der Signale mit MT5-Aufträgen verknüpft.
- `main.py`: Laufzeit-Loop, der Konfiguration, Adapter und Signale zusammenschnürt.

## Erste Schritte

1. Installiere Abhängigkeiten:

```bash
pip install -r requirements.txt
```

2. Testlauf im DryRun-Modus (kein MT5 nötig):

```bash
python main.py --once --dry-run
```

3. Richte `Symbols.txt` mit deinen Zielinstrumenten ein (ein Symbol pro Zeile).
4. Für echten Livebetrieb stelle sicher, dass MetaTrader5 installiert und `EW_MT5_*` Environment-Variablen gesetzt sind. Dann starte ohne `--dry-run`.

## Live-Workflow

1. `main.py` lädt Konfigurationen und verbindet sich via `MetaTrader5Adapter`.
2. `SignalEngine` verarbeitet die aktuellen Kursdaten und liefert Entry-Signale.
3. `OrderManager` sendet Market-Orders mit Stop/TP und verfolgt offene Positionen.

## Nächste Schritte

- Erweiterte Risikosteuerung (vol-target, Drawdown-Protektion).
- Logging/Monitoring (z.B. Equity-Graph, Notifications).
- Integration mit Vantage-REST-API für sekundäre Checks oder Absicherungen.

## Dynamic Position Sizing

- `dynamic_trend_scaling` erlaubt es, Größe auf Basis von Setup- und Zeitrahmen-Stärke selbst anzupassen.
- Die Mappings `setup_size_factors` und `tf_size_factors` definieren individuelle Skalierungsfaktoren für W3/C/W5-Setups oder etwaige H1/M30-Zyklen.
- Dadurch reagiert das Live-System automatisch etwas zurückhaltender auf schwächere Signale.