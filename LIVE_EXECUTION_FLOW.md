 # Live Execution Flow und Vergleich zum Backtest

 ## 1. Gesamtübersicht
1. `main.py` verbindet `LiveConfig`, `MetaTrader5Adapter`, `SignalEngine`, `OrderManager` und einen `CycleRunner`.
2. Jede Ausführungsschleife läuft über `Symbols.txt`, holt sich über `MetaTrader5Adapter.get_rates` aktuelle Kerzen, erstellt und verarbeitet ein DataFrame und lässt den `SignalEngine` `EntrySignal`-Instanzen erzeugen.
3. Die Signale gehen in `OrderManager.evaluate_signals`: Für jedes Symbol greift dieselbe `ElliottEngine`, erzeugt Setups/Bestätigungen und überlässt dem Live-Order-Manager die Ausführungsvorbehalte (Cooldowns, Richtungsbeschränkungen, Filling-Mode-Retries) und die Risikogrößen aus `LiveConfig`.
4. Jeder Marktauftrag passt Stops an das Broker-`trade_stops_level` an, probiert Filling-Modi (IOC/FOK/RETURN), loggt ausführliche Metadaten und speichert funktionierende Filling-Modi für zukünftige Zyklen.

 ## 2. Signallogik (`live_core/signals.py`)
- Der `SignalEngine` nutzt dieselbe `ElliottEngine` wie der Backtest für Zig-Zag-Pivots, Impulse und ABCs. Standardwerte (`entry_zone_*`, `tp1`, `tp2`, ATR-Puffer) entsprechen dem aggressiven Profil in `EW_Backtester.py`.
- Die Bestätigungsregeln (Durchbruch vorheriger Extreme, optional EMA-Kreuzung) sowie die ATR-basierten Volumenfilter spiegeln die `CONFIRM_RULES` und `ATR_PCT`-Grenzen des Backtests wider.
- Touch-Fenster (`entry_window_h1/m30`) und Lookback-Einstellungen sind deckungsgleich mit den Standards des aggressiven Profils, sodass der Live-Lauf dieselbe Preisstruktur untersucht wie der Backtest.

 ## 3. Ausführung & Risiko (`live_core/execution.py`, `live_core/mt5_adapter.py`, `live_core/config.py`)
- `LiveConfig` definiert Risiko-Parameter (`risk_per_trade`, `lot_size_per_risk_point`, `min_lot`, `max_lot`, `order_cooldown_seconds`) und neue Flags (`dynamic_dd_risk`, `dd_risk_steps`, `risk_per_trade_min/max`, `use_vol_target`, `target_annual_vol`, `vol_window_trades`, `size_short_factor`, `use_ml_filters`, `ml_probability_threshold`, `ml_threshold_shift`, `size_by_prob`), um die adaptiven Backtest-Regeln nachzubilden.
- Zusätzlich kann `use_pending_orders` aktiviert werden. Dann wählt der OrderManager aus der `EntrySignal`-Zone den passenden Limit-Preis, berechnet Risiko und Lotgröße am geplanten Ausführungspreis und platziert eine MT5-Pending-Limit-Order, die nach `pending_order_expiry_minutes` verfällt. So bleiben die übrigen Filterschichten (Profit-Factor, Cooldown, Exposure) unangetastet, während die Entry-Zone näher an den Backtest heranreicht.
- Zusätzlich erlaubt `webhook_url` eine Discord-Benachrichtigung pro erfolgreichem Live-Trade, falls ein Webhook konfiguriert ist.
- `_calculate_volume` errechnet Lotgrößen aus Kontostand × `risk_per_trade`, Stop-Abstand, Tick-/Kontrakt-Metadaten und `lot_size_per_risk_point`. Zusätzlich wird jetzt der aktuelle Balance-Drawdown über `_risk_multiplier_for_dd` skaliert, eine optionale Volatilitätsziel-Skalierung (`_vol_multiplier`) durch feste R-Multiples angewendet sowie Short-Positionen und ML-Wahrscheinlichkeiten (`confidence`) über Multiplikatoren angepasst, bevor `min_lot`/`max_lot` gelten.
- `SignalEngine` akzeptiert eine optionale `MLProbabilityProvider`-Datei (`ml_probability_path`), die CSV/JSON-Zeilen (Symbol, Setup, Richtung, Entry-Time, Probability) läd und damit die `confidence`-Werte für `use_ml_filters`/`size_by_prob` füllt. Ohne Datei bleibt das Confidence-Feld auf `ml_default_probability`, die ebenfalls vom Config abhängt.
- Der Adapter kapselt MT5 mit Stop-Validierung/-Anpassung, ticksensitiven Risikometriken und Fallback-Strategien über IOC/FOK/RETURN. Die Logs zeigen Stop-Anpassungen, Versuchszusammenfassungen und Retcode-Infos, damit sich Dry Runs ähnlich zu einem Live-Run verhalten.

 ## 4. Wie nah ist das am Backtest?
| Feature | Backtest | Live-Executor | Anmerkungen |
| --- | --- | --- | --- |
| Signal-Modell | `ElliottEngine` + Bestätigungsregeln | dieselbe `ElliottEngine` + Regeln | Beide nutzen identische Pivot-/Impuls-Erkennung, daher sollten dieselben Setups entstehen.
| Risiko pro Trade | Dynamisch (DD, Vol-Ziel, ML-Größenanpassung, min/max) | optional dynamischer DD-Multiplikator, Volatilitätsziel und min/max-Clamps aus `LiveConfig` plus Short-Faktor | Aktivierbar via `dynamic_dd_risk`, `use_vol_target`, `risk_per_trade_min/max` (§3) und ML-Sizer; default bewusst konservativ eingestellt.
| ML-/Wahrscheinlichkeitsfilter | standardmäßig aktiv (`USE_ML`, `PROFILES`) | konfigurierbare Confidence-Filter (`use_ml_filters`, `ml_probability_threshold`) und Wahrscheinlichkeitsscaling (`size_by_prob`, `prob_size_*`) | Signal-Confidence stammt aktuell aus heuristischen Metriken; echte Wahrscheinlichkeiten müssten eingespeist werden, um ML-ähnliche Schwellen zu reproduzieren.
| Ausführungsrealismus | simulierte Orders (Gebühren/Slippage) und Offline-Reporting | echte MT5-Aufträge mit Filling-Mode-Retries und Stop-Validierung | Live fügt Broker-Realismus hinzu, während der Backtest auf simulierte Ausführungen setzt.
| Logging/Diagnostik | PDF/CSV-Reports, Equity-Kurven | Konsolenlogs, `CycleSummary`, Filling-Mode-/Error-Details | Unterschiedliche Formate, aber beide liefern pro Zyklus/Symbol ähnliche Zusammenfassungen.

### Fazit
Der Live-Executor behält die **same Struktur**: Elliott-Wave-Setups, ATR-/Bestätigungsfilter sowie die Platzierung von SL/TP folgen den Backtest-Standards. Die **Lücke** liegt in der Risikosteuerung: Live nutzt aktuell nur `risk_per_trade` und `lot_size_per_risk_point`, während Drawdown-Skalierung, ML-Wahrscheinlichkeitsfilter und Volatilitätsziele der Backtest-`PROFILES` fehlen. Diese Regeln müssten in `LiveConfig` und `OrderManager` übernommen werden (dynamisches DD-Scaling, ML-Filter/Größenmultiplikatoren), damit die Live-Größen dieselbe Wirkung erzielen wie im Backtest.

 ## 5. Nächste Schritte für Parität
1. die adaptiven Risiko-Skalierer aus dem Backtest (DD-Stufen, Vol-Ziel, ML-Wahrscheinlichkeitsfaktoren) in `LiveConfig` verfügbar machen und vor `_calculate_volume` den tatsächlichen `risk_per_trade` berechnen.
2. die ML-Kalibrierung/-Threshold-Logik aus `EW_Backtester.py` in den Live-Signalprozessor einfließen lassen (oder über gespeicherte Wahrscheinlichkeiten nachbauen), indem `ml_probability_path` mit den bereitgestellten Wahrscheinlichkeiten befüllt wird, sodass `MLProbabilityProvider`, `use_ml_filters` und `size_by_prob` dasselbe Verhalten liefern.
3. `CycleSummary`-Kennzahlen kontinuierlich mit Backtest-Metriken (Signale pro Sekunde, Retcode-Statistiken) vergleichen, um sicherzustellen, dass die Live-Schleife weiterhin dem historischen Verhalten entspricht.
