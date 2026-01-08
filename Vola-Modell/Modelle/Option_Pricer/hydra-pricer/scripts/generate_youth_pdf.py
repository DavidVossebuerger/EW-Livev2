"""Generate a youth-friendly PDF explainer for HYDRA PRICER."""
from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path
from typing import List

from fpdf import FPDF

PROJECT_NAME = "HYDRA PRICER"
OUTPUT_DEFAULT = "HYDRA_PRICER_Erklaerung.pdf"

SECTION_DATA = [
    {
        "title": "Level 1: Options 101",
        "paragraphs": [
            "Stell dir eine Option wie ein Konzertticket vor: du sicherst dir heute das Recht, spaeter zu einem festen Preis reinzukommen. Ein Call setzt auf steigende Preise, ein Put auf fallende Preise.",
            "Damit wir wissen, ob der Ticketpreis fair ist, brauchen wir Pricing-Modelle. Sie kombinieren Spot (aktueller Kurs), Strike (Ticketpreis), Restlaufzeit, Zins und Volatilitaet (wie wild der Markt tanzt).",
        ],
        "bullets": [
            "Call = Recht zu kaufen, Put = Recht zu verkaufen",
            "European bedeutet Ausuebung nur am Ende, American erlaubt frueher",
            "Volatilitaet ist das Zittern im Kursverlauf: je mehr Drama, desto teurer die Option",
        ],
        "fun_fact": "Fun Fact: Trader nennen Volatilitaet gerne einfach 'vol', weil alles cooler klingt, wenn man es abkuerzt.",
    },
    {
        "title": "Level 2: Modelle in HYDRA",
        "paragraphs": [
            "Unser Toolkit kombiniert mehrere Perspektiven, damit du Preise vergleichen kannst.",
        ],
        "bullets": [
            "Black-Scholes: Mathe-Klassiker mit geschlossener Formel. Ideal, wenn der Markt ruhig bleibt.",
            "Binomial Tree: Bauplan aus vielen kleinen Zeitschritten. Kann American-Features und unterschiedliche Volas abbilden.",
            "Monte Carlo: Computerspiel mit tausenden Zufallspfaden. Perfekt fuer exotische Payoffs wie Asian, Lookback oder Barrier.",
        ],
    },
    {
        "title": "Level 3: Analytics & Skills",
        "paragraphs": [
            "Modelle liefern Preise, aber die echten Pros tracken Sensitivitaeten und testen Strategien.",
        ],
        "bullets": [
            "Greeks: Delta (Spot-Boost), Gamma (wie schnell Delta dreht), Vega (Vol-Schock), Theta (Zeitverfall), Rho (Zinsen).",
            "Implied-Vol-Kalibrierung: Newton-Raphson sucht die Vol, die zum Marktpreis passt.",
            "Delta-Hedge-Backtesting: Wir laufen durch echte Preisreihen und schauen, wie gut unser Hedge PnL stabilisiert.",
        ],
    },
    {
        "title": "Level 4: Streamlit Dashboard",
        "paragraphs": [
            "Alles landet in einer App, die du wie ein Gaming-HUD bedienen kannst.",
        ],
        "bullets": [
            "Pricing & Greeks Tab mit Modellvergleich und Delta/Gamma etc.",
            "Payoff-, Greeks- und Vol-Surface-Plots zum direkten Visualisieren.",
            "Calibration Tab fuer implied Vol, Backtesting Tab fuer den Delta-Hedge-Lauf.",
            "Education Mode mit Presets, Lexikon und Playbook, damit das Lernen nicht trocken ist.",
        ],
    },
]

CHECKLIST = [
    "Starte das Dashboard: streamlit run app/streamlit_app.py",
    "Vergleiche Call vs. Put und beobachte das Payoff-Diagramm",
    "Kalibriere eine implied Vol aus einem Beispielpreis",
    "Fuehre einen Backtest mit deinem Lieblingsticker durch",
    "Spiele die Education-Presets durch und passe danach die Inputs an",
]


class YouthPDF(FPDF):
    """Thin wrapper so we can pre-set fonts and spacing."""

    def header(self) -> None:  # pragma: no cover - visual output
        self.set_font("Helvetica", "B", 16)
        self.cell(0, 10, PROJECT_NAME, new_x="LMARGIN", new_y="NEXT")
        self.set_font("Helvetica", size=11)
        subtitle = "Options-Analytics erklaert fuer Teens"
        self.cell(0, 8, subtitle, new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

    def footer(self) -> None:  # pragma: no cover - visual output
        self.set_y(-15)
        self.set_font("Helvetica", size=9)
        self.cell(0, 10, f"Seite {self.page_no()} / {PROJECT_NAME}", align="C")


def _content_width(pdf: YouthPDF) -> float:
    return pdf.w - pdf.l_margin - pdf.r_margin


def _wrap_paragraph(pdf: YouthPDF, text: str) -> None:
    pdf.set_font("Helvetica", size=11)
    pdf.multi_cell(_content_width(pdf), 6, text)
    pdf.ln(1)


def _draw_bullets(pdf: YouthPDF, bullets: List[str]) -> None:
    pdf.set_font("Helvetica", size=11)
    for item in bullets:
        pdf.multi_cell(_content_width(pdf), 6, f"- {item}")
    pdf.ln(1)


def _draw_fun_fact(pdf: YouthPDF, message: str) -> None:
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 6, "Fun Fact", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", size=10)
    pdf.multi_cell(_content_width(pdf), 6, message)
    pdf.ln(1)


def build_pdf(output_path: Path, author: str) -> Path:
    pdf = YouthPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.set_font("Helvetica", size=11)
    pdf.multi_cell(_content_width(pdf), 6, f"Erstellt am {date.today().isoformat()} von {author}.")
    pdf.ln(3)

    for section in SECTION_DATA:
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 8, section["title"], new_x="LMARGIN", new_y="NEXT")
        pdf.ln(1)
        for paragraph in section.get("paragraphs", []):
            _wrap_paragraph(pdf, paragraph)
        _draw_bullets(pdf, section.get("bullets", []))
        fun_fact = section.get("fun_fact")
        if fun_fact:
            _draw_fun_fact(pdf, fun_fact)
        pdf.ln(2)

    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, "Next Steps Checklist", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(1)
    _draw_bullets(pdf, CHECKLIST)

    pdf.output(str(output_path))
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a youth-friendly PDF for HYDRA PRICER.")
    parser.add_argument("--output", type=str, default=OUTPUT_DEFAULT, help="Pfad zur PDF-Datei")
    parser.add_argument("--author", type=str, default="HYDRA Crew", help="Name, der im Dokument erscheinen soll")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output).resolve()
    build_pdf(output_path=output_path, author=args.author)
    print(f"PDF gespeichert unter: {output_path}")


if __name__ == "__main__":
    main()
