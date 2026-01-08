# HYDRA PRICER – Institutional Options Analytics

Hydra Pricer ist ein mehrphasiges Quant-Finance-Projekt, das einen vollständigen Workflow von Options-Pricing, Risikoanalysen bis hin zum interaktiven Streamlit-Dashboard abbildet. Dieses Repository dient aktuell als Grundgerüst – Implementierungen folgen entlang der Meilensteine aus der Projektspezifikation (`Angabe.txt`).

## Projektstatus
- ✅ Projektstruktur angelegt
- ⏳ Pricing-Modelle (BSM, Binomial, Monte Carlo)
- ⏳ Greeks-Engine und Risikoanalysen
- ⏳ Datenanbindung, Dashboard & Deployment

## Entwicklung
```bash
# Abhängigkeiten installieren
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt

# Tests ausführen
pytest
```

## Struktur
```
hydra-pricer/
├── app/                # Streamlit UI (Phase 3)
├── src/
│   ├── data/           # Marktdaten-Adapter
│   ├── greeks/         # Sensitivitätsberechnungen
│   ├── models/         # Pricing-Engines
│   └── utils/          # Gemeinsame Helferfunktionen
├── tests/              # pytest-Suite
├── requirements.txt
├── README.md
└── TODO.md             # Erweiterungs-Backlog
```

## Nächste Schritte
1. Black-Scholes-Engine mit Pricing & Greeks implementieren
2. Binomial-Modell (amerikanische Optionen) ergänzen
3. Monte-Carlo-Pricer samt Payoff-Bibliothek aufsetzen
4. Datenebene, Dashboard und Deployment abrunden
