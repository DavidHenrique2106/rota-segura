"""ETAPA 3 - DIVISÃO DA BASE EM TREINO, VALIDAÇÃO E TESTE"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from pathlib import Path

from sklearn.model_selection import train_test_split

SEED = 42
CLASSES = ("sem_obstaculo", "com_obstaculo")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".heic"}
SOURCES = {
    "fundos": "fundos/*.jpg",
    "cone": "recortes/cone/*.png",
    "caixa": "recortes/caixa/*.png",
}


def split(files: list[str]) -> dict[str, list[str]]:
    train, rest = train_test_split(files, test_size=0.30, random_state=SEED, shuffle=True)
    validation, test = train_test_split(rest, test_size=0.50, random_state=SEED, shuffle=True)
    return {"treino": sorted(train), "validacao": sorted(validation), "teste": sorted(test)}


BURST_CHUNK = 10


def group_of_official(name: str) -> str:
    """Rajadas (IMG_..._BURST12.jpg) são divididas em trechos de 10 fotos seguidas; cada trecho é um
    grupo que fica inteiro no treino, na validação ou no teste (fotos vizinhas são quase iguais)."""
    stem = Path(name).stem
    match = re.search(r"_BURST(\d+)", stem)
    if not match:
        return stem
    return f"{stem[:match.start()]}_t{(int(match.group(1)) - 1) // BURST_CHUNK}"


def group_split(items: list[tuple[str, str]], group_fn) -> dict[str, list[list[str]]]:
    """Divide por GRUPO (rajada / trecho de vídeo) em 70/15/15, garantindo pelo menos um grupo na
    validação e no teste e, se possível, as duas classes em cada parte."""
    groups = sorted({group_fn(f) for f, _ in items})
    overall = sum(label == CLASSES[1] for _, label in items) / max(len(items), 1)
    best, best_score = None, float("inf")
    for attempt in range(500):
        shuffled = groups[:]
        random.Random(SEED + attempt).shuffle(shuffled)
        n = len(shuffled)
        n_val = max(1, round(n * 0.15)) if n >= 3 else 0
        n_test = max(1, round(n * 0.15)) if n >= 3 else 0
        n_train = n - n_val - n_test
        part_of = {g: "treino" for g in shuffled[:n_train]}
        part_of.update({g: "validacao" for g in shuffled[n_train:n_train + n_val]})
        part_of.update({g: "teste" for g in shuffled[n_train + n_val:]})
        parts = {"treino": [], "validacao": [], "teste": []}
        for f, label in items:
            parts[part_of[group_fn(f)]].append([f, label])
        both = all({label for _, label in parts[p]} >= set(CLASSES) for p in ("validacao", "teste"))
        score = sum(abs(sum(l == CLASSES[1] for _, l in parts[p]) / max(len(parts[p]), 1) - overall)
                    for p in ("validacao", "teste")) + (0 if both else 10)
        if score < best_score:
            best, best_score = parts, score
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fontes", type=Path, default=Path("data/fontes"))
    parser.add_argument("--saida", type=Path, default=Path("data/divisao.json"))
    args = parser.parse_args()

    division: dict[str, dict[str, list[str]]] = {}
    for name, pattern in SOURCES.items():
        files = sorted(str(path.relative_to(args.fontes)) for path in args.fontes.glob(pattern))
        if len(files) < 10:
            raise SystemExit(f"Poucos arquivos em '{name}' ({len(files)}). Rode antes: python ml/1_coleta_dados.py")
        division[name] = split(files)

    labels_csv = Path("data/rotulos_automaticos.csv")
    if not labels_csv.exists():
        raise SystemExit("Faltam os rótulos das fotos reais. Rode antes: python ml/2_rotulagem_automatica.py")
    with labels_csv.open(newline="", encoding="utf-8") as handle:
        real = [(f"fotos_reais/{r['arquivo']}", r["rotulo"]) for r in csv.DictReader(handle) if r["rotulo"] in CLASSES]
    files, labels = [f for f, _ in real], [label for _, label in real]
    train, rest, _, rest_labels = train_test_split(files, labels, test_size=0.30, random_state=SEED, stratify=labels)
    validation, test = train_test_split(rest, test_size=0.50, random_state=SEED, stratify=rest_labels)
    division["fotos_reais"] = {"treino": sorted(train), "validacao": sorted(validation), "teste": sorted(test)}

    official_csv = Path("data/rotulos_oficial.csv")
    if official_csv.exists():
        with official_csv.open(newline="", encoding="utf-8") as handle:
            official = [(f"oficial/{r['arquivo']}", r["rotulo"]) for r in csv.DictReader(handle) if r["rotulo"] in CLASSES]
        division["oficial"] = group_split(official, group_of_official)
        n_groups = len({group_of_official(f) for f, _ in official})
        print(f"Base oficial: {len(official)} fotos em {n_groups} rajadas/grupos")

    maquete = Path("data/maquete")
    own_photos: list[tuple[str, str]] = []
    for label in CLASSES:
        if (maquete / label).exists():
            own_photos += [(str(p.relative_to(maquete)), label) for p in sorted((maquete / label).rglob("*"))
                           if p.suffix.lower() in IMAGE_EXTENSIONS]
    auto_csv = Path("data/rotulos_maquete.csv")
    if auto_csv.exists():
        with auto_csv.open(newline="", encoding="utf-8") as handle:
            own_photos += [(f"sem_rotulo/{r['arquivo']}", r["rotulo"]) for r in csv.DictReader(handle)
                           if r["rotulo"] in CLASSES]
    own = {"treino": [], "validacao": [], "teste": []}

    frames = [(f, label) for f, label in own_photos if re.match(r"sem_rotulo/video\d+_t\d+_q", f)]
    own_photos = [item for item in own_photos if item not in frames]
    if frames:
        segment = lambda f: re.match(r"sem_rotulo/(video\d+_t\d+)_", f).group(1)
        for part, chosen in group_split(frames, segment).items():
            own[part] += chosen
        print(f"Quadros de vídeo: {len(frames)} em {len({segment(f) for f, _ in frames})} trechos de 5 s")

    if own_photos:
        by_label = {label: [p for p in own_photos if p[1] == label] for label in CLASSES}
        for label, items in by_label.items():
            if len(items) >= 7:
                for part, chosen in split([f for f, _ in items]).items():
                    own[part] += [[f, label] for f in chosen]
            elif items:
                own["treino"] += [[f, label] for f, _ in items]
                print(f"Aviso: só {len(items)} fotos da maquete com rótulo '{label}' (todas no treino).")
    division["maquete"] = own

    args.saida.parent.mkdir(parents=True, exist_ok=True)
    args.saida.write_text(json.dumps(division, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"{'fonte':<12} {'treino':>7} {'validacao':>10} {'teste':>6}")
    for name, parts in division.items():
        if not any(parts.values()):
            continue
        print(f"{name:<12} {len(parts['treino']):>7} {len(parts['validacao']):>10} {len(parts['teste']):>6}")
    print(f"Divisão salva em {args.saida}")


if __name__ == "__main__":
    main()
