"""ETAPA 2 - ROTULAGEM AUTOMÁTICA DAS FOTOS REAIS

Rotula as fotos reais coletadas na etapa 1 (data/fontes/fotos_reais) como
seguro / atencao / perigo, usando um modelo PRÉ-TREINADO de detecção de objetos
como "rotulador":

    OWLv2 (Google, google/owlv2-base-patch16-ensemble) - detector que entende
    texto: procuramos por "traffic cone", "cardboard box", "person", "chair",
    "bench", "bicycle", "trash can"... qualquer coisa que possa estar no caminho.

IMPORTANTE: o OWLv2 é usado SOMENTE aqui, para criar os rótulos da base. Ele não
vai para o app. O modelo do projeto (etapa 5) é uma CNN treinada do zero que
aprende com esses rótulos. É a técnica de "professor-aluno" (destilação):
um modelo grande rotula os dados e um modelo pequeno, nosso, aprende com eles.

Regra (a mesma usada na montagem da etapa 4), olhando cada obstáculo detectado
que esteja no corredor central da imagem (por onde a pessoa vai passar):

    perigo   obstáculo grande: altura >= 48% da imagem e base na parte de baixo
             (>= 80%), ou objeto largo (>= 35%) encostado na borda de baixo
    atencao  obstáculo médio: altura entre 22% e 36%, base entre 60% e 92%
    seguro   nenhum obstáculo no corredor, ou só obstáculos pequenos (< 13%)

Fotos com obstáculos entre as faixas (ambíguos) são descartadas, para não
ensinar rótulos duvidosos.

Também rotula as fotos da MAQUETE tiradas pelo grupo, sem precisar rotular à
mão: coloque-as em data/maquete/sem_rotulo/ e rode
    python ml/2_rotulagem_automatica.py --fotos data/maquete/sem_rotulo --saida data/rotulos_maquete.csv
(o executar_tudo.sh faz isso sozinho quando a pasta tem fotos). As fotos são
recortadas no quadrado central antes da detecção, igual ao que o app analisa.

Saída:
    data/rotulos_automaticos.csv
    ml/artefatos/rotulagem_<classe>.jpg   (amostras com as detecções desenhadas,
                                           para conferência visual)

Uso (precisa de ml/requirements-rotulagem.txt):
    python ml/2_rotulagem_automatica.py
"""

from __future__ import annotations

import argparse
import csv
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

try:  # permite abrir fotos HEIC (padrão do iPhone)
    from pillow_heif import register_heif_opener

    register_heif_opener()
except ImportError:
    pass
from transformers import Owlv2ForObjectDetection, Owlv2Processor

DETECTOR = "google/owlv2-base-patch16-ensemble"
# Consulta em inglês (o detector foi treinado em inglês) -> nome em português.
OBSTACLES = {
    "a traffic cone": "cone", "a cardboard box": "caixa", "a box": "caixa", "a person": "pessoa",
    "a chair": "cadeira", "a bench": "banco", "a table": "mesa", "a bicycle": "bicicleta",
    "a trash can": "lixeira", "a fire hydrant": "hidrante", "a pole": "poste", "a potted plant": "vaso",
    "a suitcase": "mala", "a barrel": "barril", "a sofa": "sofa", "a dog": "cachorro",
    "a car": "carro", "a motorcycle": "moto", "a stroller": "carrinho", "a shopping cart": "carrinho",
    "a pillar": "pilar", "a barrier": "barreira", "a sign post": "placa",
    # Objetos em miniatura da maquete (testado nas fotos de data/maquete/visao_geral).
    "a toy traffic cone": "cone", "a road barrier": "barreira", "a can": "lata", "a jar": "pote",
    "a toy figure": "boneco", "a toy car": "carro",
}
QUERIES = list(OBSTACLES)
SCORE_MIN = 0.28
CORRIDOR = (0.25, 0.75)  # faixa horizontal por onde a pessoa vai passar
CLASSES = ("seguro", "atencao", "perigo")


def square(image: Image.Image, size: int = 256) -> Image.Image:
    """Quadrado central (o que o app analisa). Fotos já quadradas só são reduzidas."""
    from PIL import ImageOps

    image = ImageOps.exif_transpose(image).convert("RGB")
    side = min(image.size)
    left, top = (image.width - side) // 2, (image.height - side) // 2
    return image.crop((left, top, left + side, top + side)).resize((size, size), Image.LANCZOS)


def object_level(box: tuple[float, float, float, float]) -> str | None:
    """Nível de risco de UM obstáculo (coordenadas normalizadas 0..1)."""
    x0, y0, x1, y1 = box
    overlap = max(0.0, min(x1, CORRIDOR[1]) - max(x0, CORRIDOR[0]))
    if overlap < 0.25 * (CORRIDOR[1] - CORRIDOR[0]):
        return None  # fora do caminho (nas laterais)
    height, width, bottom = y1 - y0, x1 - x0, y1
    if (height >= 0.48 and bottom >= 0.80) or (bottom >= 0.97 and width >= 0.35):
        return "perigo"
    if 0.22 <= height < 0.36 and 0.60 <= bottom < 0.92:
        return "atencao"
    if height < 0.13:
        return "seguro"
    return "ambiguo"


def image_label(levels: list[str]) -> str | None:
    if "perigo" in levels:
        return "perigo"
    if "ambiguo" in levels:
        return None
    if "atencao" in levels:
        return "atencao"
    return "seguro"


def detect(model, processor, images: list[Image.Image], device: str) -> list[list[tuple]]:
    inputs = processor(text=[QUERIES] * len(images), images=images, return_tensors="pt").to(device)
    with torch.inference_mode():
        outputs = model(**inputs)
    # OWLv2 trabalha numa versão quadrada da imagem; nossas fotos já são quadradas.
    sizes = torch.tensor([[1.0, 1.0]] * len(images))
    results = processor.post_process_object_detection(outputs, threshold=SCORE_MIN, target_sizes=sizes)
    found = []
    for result in results:
        boxes = []
        for box, label, score in zip(result["boxes"].tolist(), result["labels"].tolist(), result["scores"].tolist()):
            x0, y0, x1, y1 = (min(max(v, 0.0), 1.0) for v in box)
            if x1 - x0 > 0.95 and y1 - y0 > 0.95:
                continue  # "objeto" do tamanho da foto inteira: falso positivo típico
            boxes.append((OBSTACLES[QUERIES[label]], round(score, 3), (x0, y0, x1, y1)))
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
            color = {"perigo": "red", "atencao": "orange", "seguro": "lime"}.get(level, "white")
            draw.rectangle((x0, y0, x1, y1), outline=color, width=2)
            draw.text((x0 + 2, y0 + 2), name, fill=color)
        sheet.paste(image, ((n % columns) * cell, (n // columns) * cell))
    sheet.save(output, quality=88)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fotos", type=Path, default=Path("data/fontes/fotos_reais"))
    parser.add_argument("--saida", type=Path, default=Path("data/rotulos_automaticos.csv"))
    parser.add_argument("--lote", type=int, default=4)
    parser.add_argument("--limite", type=int, default=600,
                        help="máximo de fotos por execução; rode de novo para continuar (evita acúmulo de memória da GPU)")
    args = parser.parse_args()

    files = sorted(p.name for p in args.fotos.iterdir()
                   if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".heic"})
    if not files:
        raise SystemExit("Nenhuma foto real encontrada. Rode antes: python ml/1_coleta_dados.py")
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Carregando o detector {DETECTOR} ({device})...")
    processor = Owlv2Processor.from_pretrained(DETECTOR)
    model = Owlv2ForObjectDetection.from_pretrained(DETECTOR).to(device).eval()

    # Retoma de onde parou: o progresso é gravado a cada lote em um arquivo parcial.
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
        for name, boxes in zip(batch, detect(model, processor, images, device)):
            levels, details = [], []
            for obstacle, score, box in boxes:
                level = object_level(box)
                if level is not None:
                    levels.append(level)
                details.append(f"{obstacle}|{level or 'fora'}|{','.join(f'{v:.3f}' for v in box)}")
            label = image_label(levels)
            row = {"arquivo": name, "rotulo": label or "descartada", "deteccoes": ";".join(details)}
            rows.append(row)
            writer.writerow(row)
        handle.flush()
        if device == "mps":
            torch.mps.empty_cache()  # evita que a memória da GPU cresça a cada lote
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
    prefix = "rotulagem_maquete" if "maquete" in str(args.fotos) else "rotulagem"
    for label in CLASSES:
        review_sheet(rows, args.fotos, label, artifacts / f"{prefix}_{label}.jpg")
    counts = {label: sum(r["rotulo"] == label for r in rows) for label in (*CLASSES, "descartada")}
    print(f"Rótulos: {counts}")
    print(f"Salvo em {args.saida}. Confira as amostras em ml/artefatos/rotulagem_*.jpg")


if __name__ == "__main__":
    main()
