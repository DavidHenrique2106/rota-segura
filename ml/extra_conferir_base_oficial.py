"""EXTRA - CONFERE SE TODAS AS FOTOS DO DRIVE ESTÃO NA BASE OFICIAL"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

PHOTO = {".jpg", ".jpeg", ".png", ".heic", ".webp"}


def main() -> None:
    drive = {p.stem for p in Path("data/maquete/drive_oficial").iterdir() if p.suffix.lower() in PHOTO}
    base = {p.stem for p in Path("data/maquete/oficial").glob("*.jpg")}
    labels_path = Path("data/rotulos_oficial.csv")
    labels = {}
    if labels_path.exists():
        with labels_path.open(newline="", encoding="utf-8") as handle:
            labels = {Path(r["arquivo"]).stem: r["rotulo"] for r in csv.DictReader(handle)}
    used = set()
    division_path = Path("data/divisao.json")
    if division_path.exists():
        division = json.loads(division_path.read_text(encoding="utf-8")).get("oficial", {})
        used = {Path(f).stem for items in division.values() for f, _ in items}

    print(f"Fotos no Drive (baixadas): {len(drive)}")
    print(f"Na base oficial:           {len(base)}")
    print(f"Rotuladas:                 {len(labels)}  {dict(Counter(labels.values()))}")
    if used:
        print(f"Usadas na divisão:         {len(used)}")
    problems = {
        "baixada mas não convertida (arquivo inválido/incompleto)": drive - base,
        "sem rótulo (rode a etapa 2)": base - set(labels),
        "descartada pelo rotulador (objeto ambíguo, entre 3% e 6% da altura)":
            {s for s, r in labels.items() if r == "descartada"},
    }
    if used:
        problems["rotulada mas fora da divisão (rode a etapa 3)"] = {
            s for s, r in labels.items() if r != "descartada"} - used
    for reason, names in problems.items():
        if names:
            print(f"\n{len(names)} foto(s) - {reason}:")
            for name in sorted(names):
                print(f"   {name}")
    if not any(problems.values()):
        print("\nTodas as fotos do Drive estão na base oficial.")


if __name__ == "__main__":
    main()
