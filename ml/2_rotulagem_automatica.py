"""ETAPA 2 - ROTULAGEM AUTOMÁTICA DAS FOTOS REAIS"""

from __future__ import annotations

import argparse
import csv
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

try:
    from pillow_heif import register_heif_opener

    register_heif_opener()
except ImportError:
    pass
from transformers import Owlv2ForObjectDetection, Owlv2Processor

DETECTOR = "google/owlv2-base-patch16-ensemble"
OBSTACLES = {
    "a traffic cone": "cone", "a cardboard box": "caixa", "a box": "caixa", "a person": "pessoa",
    "a chair": "cadeira", "a bench": "banco", "a table": "mesa", "a bicycle": "bicicleta",
    "a trash can": "lixeira", "a fire hydrant": "hidrante", "a pole": "poste", "a potted plant": "vaso",
    "a suitcase": "mala", "a barrel": "barril", "a sofa": "sofa", "a dog": "cachorro",
    "a car": "carro", "a motorcycle": "moto", "a stroller": "carrinho", "a shopping cart": "carrinho",
    "a pillar": "pilar", "a barrier": "barreira", "a sign post": "placa",
    "a toy traffic cone": "cone", "a road barrier": "barreira", "a can": "lata", "a jar": "pote",
    "a toy figure": "boneco", "a toy car": "carro", "a toy gas station": "posto", "a black jar": "pote",
}
QUERIES = list(OBSTACLES)
MAQUETE_QUERIES = ["a toy traffic cone", "a traffic cone", "a road barrier", "a can", "a jar", "a toy figure",
                   "a toy car", "a toy gas station", "a black jar"]
SCORE_MIN = 0.28
CORRIDOR = (0.25, 0.75)
CLASSES = ("sem_obstaculo", "com_obstaculo")
MIN_OBSTACLE = 0.15
MIN_VISIBLE = 0.08


def square(image: Image.Image, size: int = 256) -> Image.Image:
    """Quadrado central (o que o app analisa). Fotos já quadradas só são reduzidas."""
    from PIL import ImageOps

    image = ImageOps.exif_transpose(image).convert("RGB")
    side = min(image.size)
    left, top = (image.width - side) // 2, (image.height - side) // 2
    return image.crop((left, top, left + side, top + side)).resize((size, size), Image.LANCZOS)


def mostly_tactile_strip(image: Image.Image, box: tuple[float, float, float, float]) -> bool:
    """Na maquete, o detector às vezes confunde a FAIXA AMARELA TÁTIL com uma barreira/cone.
    Descarta caixas em que mais de 35% dos pixels são amarelo-vivo (a cor da faixa)."""
    w, h = image.size
    x0, y0, x1, y1 = box
    if (y1 - y0) < 0.35 and (x1 - x0) < 0.35:
        return False
    crop = image.crop((int(x0 * w), int(y0 * h), max(int(x1 * w), int(x0 * w) + 1), max(int(y1 * h), int(y0 * h) + 1)))
    hsv = np.asarray(crop.convert("HSV")).astype(np.int16)
    yellow = (hsv[..., 0] >= 28) & (hsv[..., 0] <= 50) & (hsv[..., 1] >= 110) & (hsv[..., 2] >= 120)
    return float(yellow.mean()) > 0.35


def object_level(box: tuple[float, float, float, float]) -> str:
    """Classifica UM objeto detectado (coordenadas normalizadas 0..1)."""
    x0, y0, x1, y1 = box
    overlap = max(0.0, min(x1, CORRIDOR[1]) - max(x0, CORRIDOR[0]))
    if overlap < 0.5 * max(x1 - x0, 1e-6) and overlap < 0.25 * (CORRIDOR[1] - CORRIDOR[0]):
        return "fora"
    height = y1 - y0
    if height >= MIN_OBSTACLE:
        return "obstaculo"
    if height >= MIN_VISIBLE:
        return "ambiguo"
    return "pequeno"


def image_label(levels: list[str]) -> str | None:
    if "obstaculo" in levels:
        return "com_obstaculo"
    if "ambiguo" in levels:
        return None
    return "sem_obstaculo"


def relabel(csv_path: Path) -> list[dict]:
    """Recalcula os rótulos a partir das detecções guardadas no CSV (sem rodar o detector)."""
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        levels, details = [], []
        for item in filter(None, row["deteccoes"].split(";")):
            name, old_level, coords = item.split("|")
            box = tuple(float(v) for v in coords.split(","))
            is_strip = name == "faixa_amarela" and not ((box[3] - box[1]) < 0.35 and (box[2] - box[0]) < 0.35)
            if name == "faixa_amarela" and not is_strip:
                name = "objeto"
            level = "fora" if is_strip else object_level(box)
            if not is_strip:
                levels.append(level)
            details.append(f"{name}|{level}|{coords}")
        row["rotulo"] = image_label(levels) or "descartada"
        row["deteccoes"] = ";".join(details)
    return rows


def detect(model, processor, images: list[Image.Image], device: str, queries: list[str] = QUERIES) -> list[list[tuple]]:
    inputs = processor(text=[queries] * len(images), images=images, return_tensors="pt").to(device)
    with torch.inference_mode():
        outputs = model(**inputs)
    sizes = torch.tensor([[1.0, 1.0]] * len(images))
    results = processor.post_process_object_detection(outputs, threshold=SCORE_MIN, target_sizes=sizes)
    found = []
    for result in results:
        boxes = []
        for box, label, score in zip(result["boxes"].tolist(), result["labels"].tolist(), result["scores"].tolist()):
            x0, y0, x1, y1 = (min(max(v, 0.0), 1.0) for v in box)
            if x1 - x0 > 0.95 and y1 - y0 > 0.95:
                continue
            boxes.append((OBSTACLES[queries[label]], round(score, 3), (x0, y0, x1, y1)))
        found.append(boxes)
    return found


def review_sheet(rows: list[dict], photos: Path, label: str, output: Path, count: int = 40) -> None:
    chosen = [r for r in rows if r["rotulo"] == label]
    random.Random(1).shuffle(chosen)
    chosen = chosen[:count]
    if not chosen:
        return
    cell, columns = 192, 8
    sheet = Image.new("RGB", (columns * cell, ((len(chosen) + columns - 1) // columns) * cell), "white")
    for n, row in enumerate(chosen):
        image = square(Image.open(photos / row["arquivo"]), cell)
        draw = ImageDraw.Draw(image)
        draw.line([(CORRIDOR[0] * cell, 0), (CORRIDOR[0] * cell, cell)], fill=(216, 237, 97), width=1)
        draw.line([(CORRIDOR[1] * cell, 0), (CORRIDOR[1] * cell, cell)], fill=(216, 237, 97), width=1)
        for item in filter(None, row["deteccoes"].split(";")):
            name, level, coords = item.split("|")
            x0, y0, x1, y1 = (float(v) * cell for v in coords.split(","))
            color = {"obstaculo": "red", "ambiguo": "orange", "pequeno": "lime"}.get(level, "white")
            draw.rectangle((x0, y0, x1, y1), outline=color, width=2)
            draw.text((x0 + 2, y0 + 2), name, fill=color)
        sheet.paste(image, ((n % columns) * cell, (n // columns) * cell))
    sheet.save(output, quality=88)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fotos", type=Path, default=Path("data/fontes/fotos_reais"))
    parser.add_argument("--saida", type=Path, default=Path("data/rotulos_automaticos.csv"))
    parser.add_argument("--lote", type=int, default=4)
    parser.add_argument("--recalcular", type=Path, help="CSV já rotulado: refaz só os rótulos com a regra atual")
    parser.add_argument("--limite", type=int, default=600,
                        help="máximo de fotos por execução; rode de novo para continuar (evita acúmulo de memória da GPU)")
    args = parser.parse_args()

    if args.recalcular:
        rows = relabel(args.recalcular)
        with args.recalcular.open("w", newline="", encoding="utf-8") as out:
            writer = csv.DictWriter(out, fieldnames=["arquivo", "rotulo", "deteccoes"])
            writer.writeheader()
            writer.writerows(rows)
        photos = Path("data/fontes/fotos_reais") if "automaticos" in args.recalcular.name else \
            Path("data/maquete/sem_rotulo") if "maquete" in args.recalcular.name else Path("data/maquete/oficial")
        prefix = "rotulagem_" + args.recalcular.stem.replace("rotulos_", "")
        for label in CLASSES:
            review_sheet(rows, photos, label, Path("ml/artefatos") / f"{prefix}_{label}.jpg")
        counts = {label: sum(r["rotulo"] == label for r in rows) for label in (*CLASSES, "descartada")}
        print(f"Rótulos recalculados em {args.recalcular}: {counts}")
        return

    files = sorted(p.name for p in args.fotos.iterdir()
                   if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".heic"})
    if not files:
        raise SystemExit("Nenhuma foto real encontrada. Rode antes: python ml/1_coleta_dados.py")
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Carregando o detector {DETECTOR} ({device})...")
    processor = Owlv2Processor.from_pretrained(DETECTOR)
    model = Owlv2ForObjectDetection.from_pretrained(DETECTOR).to(device).eval()

    is_maquete = "maquete" in str(args.fotos)
    fields = ["arquivo", "rotulo", "deteccoes"]
    partial = args.saida.with_suffix(".parcial.csv")
    args.saida.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    if partial.exists():
        with partial.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        print(f"Retomando: {len(rows)} fotos já rotuladas.")
    done = {r["arquivo"] for r in rows}
    pending = [f for f in files if f not in done][: args.limite]
    handle = partial.open("a", newline="", encoding="utf-8")
    writer = csv.DictWriter(handle, fieldnames=fields)
    if not rows:
        writer.writeheader()

    start = time.time()
    for i in range(0, len(pending), args.lote):
        batch = pending[i:i + args.lote]
        images = [square(Image.open(args.fotos / f)) for f in batch]
        for name, image, boxes in zip(batch, images, detect(model, processor, images, device, MAQUETE_QUERIES if is_maquete else QUERIES)):
            levels, details = [], []
            for obstacle, score, box in boxes:
                if is_maquete and mostly_tactile_strip(image, box):
                    details.append(f"faixa_amarela|fora|{','.join(f'{v:.3f}' for v in box)}")
                    continue
                level = object_level(box)
                levels.append(level)
                details.append(f"{obstacle}|{level}|{','.join(f'{v:.3f}' for v in box)}")
            label = image_label(levels)
            row = {"arquivo": name, "rotulo": label or "descartada", "deteccoes": ";".join(details)}
            rows.append(row)
            writer.writerow(row)
        handle.flush()
        if device == "mps":
            torch.mps.empty_cache()
        if (i // args.lote) % 25 == 0:
            elapsed = time.time() - start
            print(f"  {len(rows)}/{len(files)} fotos ({elapsed:.0f}s)", flush=True)
    handle.close()
    if len(rows) < len(files):
        print(f"Parcial: {len(rows)}/{len(files)} fotos. Rode de novo para continuar.")
        raise SystemExit(3)

    rows.sort(key=lambda r: r["arquivo"])
    with args.saida.open("w", newline="", encoding="utf-8") as out:
        final = csv.DictWriter(out, fieldnames=fields)
        final.writeheader()
        final.writerows(rows)
    partial.unlink()

    artifacts = Path("ml/artefatos")
    artifacts.mkdir(parents=True, exist_ok=True)
    prefix = "rotulagem_" + args.saida.stem.replace("rotulos_", "")
    for label in CLASSES:
        review_sheet(rows, args.fotos, label, artifacts / f"{prefix}_{label}.jpg")
    counts = {label: sum(r["rotulo"] == label for r in rows) for label in (*CLASSES, "descartada")}
    print(f"Rótulos: {counts}")
    print(f"Salvo em {args.saida}. Confira as amostras em ml/artefatos/rotulagem_*.jpg")


if __name__ == "__main__":
    main()
