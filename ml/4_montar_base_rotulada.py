"""ETAPA 4 - MONTAGEM DA BASE ROTULADA (seguro / atencao / perigo)

A base junta dois tipos de exemplo:

A) FOTOS REAIS rotuladas automaticamente na etapa 2 (qualquer obstáculo:
   pessoas, cadeiras, bancos, bicicletas, cones, caixas...). Arquivos "real_*".

B) MONTAGENS (arquivos "montagem_*"), para garantir muitos exemplos de cone e
   caixa em todas as distâncias. Cada montagem cola um cone ou uma caixa REAIS
(recortados na etapa 1) sobre uma foto REAL de chão, em posição e tamanho
controlados. Como sabemos exatamente onde e de que tamanho o obstáculo ficou,
o rótulo é preciso.

Regra dos rótulos (simula a visão da câmera do celular apontada para frente):

    perigo   obstáculo GRANDE no quadro (ocupa 48-85% da altura), na parte de
             baixo e no centro -> está muito perto, no caminho.
    atencao  obstáculo MÉDIO (22-36% da altura), no meio do quadro -> está a
             alguns passos de distância.
    seguro   caminho livre, OU obstáculo pequeno e distante (5-13% da altura),
             perto do horizonte.

Deixamos "folgas" entre as faixas (13-22% e 36-48%) para que nenhum exemplo
fique ambíguo entre duas classes.

As montagens completam cada classe até o total desejado, deixando a base
equilibrada (mesmo número de seguro, atenção e perigo).

Cada partição (treino/validacao/teste) usa apenas as fotos de origem que a
etapa 3 reservou para ela. Fotos próprias da maquete (data/maquete/<classe>/),
se existirem, são copiadas para a partição definida na etapa 3.

Saída:
    data/base/<particao>/<classe>/*.jpg
    data/base/metadados.csv   (fonte, posição e tamanho de cada exemplo)

Uso:
    python ml/4_montar_base_rotulada.py
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

try:  # permite abrir fotos HEIC (padrão do iPhone)
    from pillow_heif import register_heif_opener

    register_heif_opener()
except ImportError:
    pass

SEED = 42
SIZE = 256  # imagens quadradas; o treino reduz para o tamanho da rede
CLASSES = ("seguro", "atencao", "perigo")
DEFAULT_COUNTS = {"treino": 1300, "validacao": 280, "teste": 280}  # total por classe
MAX_REAL_SHARE = 0.7  # no máximo 70% de cada classe vem de fotos reais; o resto é montagem

# (altura relativa mínima, máxima), (base do objeto: y mínimo, y máximo), (centro x: mín, máx)
RULES = {
    "seguro": ((0.05, 0.13), (0.42, 0.62), (0.10, 0.90)),
    "atencao": ((0.22, 0.36), (0.64, 0.84), (0.22, 0.78)),
    "perigo": ((0.48, 0.85), (0.90, 1.08), (0.32, 0.68)),
}


class Sources:
    def __init__(self, root: Path, division: dict, part: str) -> None:
        self.backgrounds = [Image.open(root / p).convert("RGB") for p in division["fundos"][part]]
        self.objects = {
            kind: [Image.open(root / p).convert("RGBA") for p in division[kind][part]]
            for kind in ("cone", "caixa")
        }
        self.names = {
            "fundos": division["fundos"][part],
            "cone": division["cone"][part],
            "caixa": division["caixa"][part],
        }


def random_background(rng: random.Random, src: Sources) -> tuple[Image.Image, int]:
    index = rng.randrange(len(src.backgrounds))
    image = src.backgrounds[index]
    side = int(min(image.size) * rng.uniform(0.65, 1.0))
    x = rng.randint(0, image.width - side)
    y = rng.randint(0, image.height - side)
    crop = image.crop((x, y, x + side, y + side)).resize((SIZE, SIZE), Image.BILINEAR)
    if rng.random() < 0.5:
        crop = crop.transpose(Image.FLIP_LEFT_RIGHT)
    crop = ImageEnhance.Brightness(crop).enhance(rng.uniform(0.7, 1.2))
    crop = ImageEnhance.Contrast(crop).enhance(rng.uniform(0.8, 1.2))
    crop = ImageEnhance.Color(crop).enhance(rng.uniform(0.7, 1.2))
    return crop, index


def paste_object(
    rng: random.Random, canvas: Image.Image, obj: Image.Image, rule: tuple
) -> dict[str, float]:
    (h_min, h_max), (b_min, b_max), (x_min, x_max) = rule
    if rng.random() < 0.5:
        obj = obj.transpose(Image.FLIP_LEFT_RIGHT)
    obj = obj.rotate(rng.uniform(-6, 6), resample=Image.BICUBIC, expand=True)
    obj = obj.crop(obj.getchannel("A").getbbox())

    rel_height = rng.uniform(h_min, h_max)
    height = rel_height * SIZE
    width = height * obj.width / obj.height
    if width > 0.95 * SIZE:  # caixas muito largas: limita a largura
        width = 0.95 * SIZE
        height = width * obj.height / obj.width
        rel_height = height / SIZE
    width, height = max(4, round(width)), max(4, round(height))
    obj = obj.resize((width, height), Image.LANCZOS)

    # Ajusta luz do objeto para combinar com o ambiente.
    rgb, alpha = obj.convert("RGB"), obj.getchannel("A")
    rgb = ImageEnhance.Brightness(rgb).enhance(rng.uniform(0.75, 1.1))
    rgb = ImageEnhance.Color(rgb).enhance(rng.uniform(0.8, 1.15))
    obj = Image.merge("RGBA", (*rgb.split(), alpha))

    center_x = rng.uniform(x_min, x_max) * SIZE
    bottom = rng.uniform(b_min, b_max) * SIZE
    left, top = round(center_x - width / 2), round(bottom - height)
    left = min(max(left, -width // 4), SIZE - 3 * width // 4)

    # Sombra suave no chão, embaixo do objeto.
    shadow = Image.new("L", (SIZE, SIZE), 0)
    shadow_layer = Image.new("L", (width, max(3, height // 6)), 0)
    ellipse = Image.new("L", shadow_layer.size, 0)
    from PIL import ImageDraw

    ImageDraw.Draw(ellipse).ellipse((0, 0, *shadow_layer.size), fill=int(255 * rng.uniform(0.25, 0.5)))
    shadow.paste(ellipse, (left + rng.randint(-width // 10, width // 10), round(bottom - shadow_layer.height * 0.6)))
    shadow = shadow.filter(ImageFilter.GaussianBlur(max(1, width / 12)))
    canvas.paste(Image.new("RGB", canvas.size, (0, 0, 0)), (0, 0), shadow)

    canvas.paste(obj, (left, top), obj)
    return {"altura_rel": round(rel_height, 3), "centro_x": round(center_x / SIZE, 3), "base_y": round(bottom / SIZE, 3)}


def camera_effects(rng: random.Random, image: Image.Image) -> Image.Image:
    if rng.random() < 0.35:
        image = image.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 1.2)))
    array = np.asarray(image).astype(np.float32)
    array += np.random.default_rng(rng.randrange(1 << 30)).normal(0, rng.uniform(0, 6), array.shape)
    return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8))


def make_example(rng: random.Random, src: Sources, label: str) -> tuple[Image.Image, dict]:
    canvas, bg_index = random_background(rng, src)
    meta: dict = {"fundo": src.names["fundos"][bg_index], "objeto": "", "tipo": "nenhum",
                  "altura_rel": 0.0, "centro_x": 0.0, "base_y": 0.0}

    empty_path = label == "seguro" and rng.random() < 0.5
    if not empty_path:
        kind = rng.choice(("cone", "caixa"))
        obj_index = rng.randrange(len(src.objects[kind]))
        meta.update(tipo=kind, objeto=src.names[kind][obj_index])
        meta.update(paste_object(rng, canvas, src.objects[kind][obj_index], RULES[label]))

    # Às vezes coloca um segundo obstáculo, pequeno e distante, como distrator.
    if label != "seguro" and rng.random() < 0.25:
        kind = rng.choice(("cone", "caixa"))
        far = rng.choice(src.objects[kind])
        paste_object(rng, canvas, far, RULES["seguro"])
    return camera_effects(rng, canvas), meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fontes", type=Path, default=Path("data/fontes"))
    parser.add_argument("--divisao", type=Path, default=Path("data/divisao.json"))
    parser.add_argument("--saida", type=Path, default=Path("data/base"))
    parser.add_argument("--por-classe-treino", type=int, default=DEFAULT_COUNTS["treino"])
    args = parser.parse_args()

    counts = dict(DEFAULT_COUNTS, treino=args.por_classe_treino)
    division = json.loads(args.divisao.read_text(encoding="utf-8"))
    if args.saida.exists():
        shutil.rmtree(args.saida)

    rows = []
    with Path("data/rotulos_automaticos.csv").open(newline="", encoding="utf-8") as handle:
        real_labels = {f"fotos_reais/{r['arquivo']}": r["rotulo"] for r in csv.DictReader(handle)}

    for part_index, (part, per_class) in enumerate(counts.items()):
        src = Sources(args.fontes, division, part)
        rng = random.Random(SEED + part_index)
        summary = []
        for label in CLASSES:
            folder = args.saida / part / label
            folder.mkdir(parents=True, exist_ok=True)

            real = [f for f in division["fotos_reais"][part] if real_labels.get(f) == label]
            random.Random(SEED).shuffle(real)
            real = sorted(real[: int(per_class * MAX_REAL_SHARE)])
            for relative in real:
                name = f"real_{Path(relative).stem}.jpg"
                Image.open(args.fontes / relative).convert("RGB").resize((SIZE, SIZE), Image.LANCZOS).save(
                    folder / name, quality=92)
                rows.append({"particao": part, "classe": label, "arquivo": f"{part}/{label}/{name}",
                             "fundo": relative, "objeto": "", "tipo": "foto_real_rotulada",
                             "altura_rel": "", "centro_x": "", "base_y": ""})

            for i in range(per_class - len(real)):
                image, meta = make_example(rng, src, label)
                name = f"montagem_{label}_{i:04d}.jpg"
                image.save(folder / name, quality=rng.randint(75, 95))
                rows.append({"particao": part, "classe": label, "arquivo": f"{part}/{label}/{name}", **meta})
            summary.append(f"{label}: {len(real)} reais + {per_class - len(real)} montagens")
        print(f"{part:<10} {'; '.join(summary)}")

    own = division.get("maquete", {})
    for part, items in own.items():
        for relative, label in items:
            image = Image.open(Path("data/maquete") / relative)
            image = ImageOps.exif_transpose(image).convert("RGB")
            side = min(image.size)
            left, top = (image.width - side) // 2, (image.height - side) // 2
            image = image.crop((left, top, left + side, top + side)).resize((SIZE, SIZE), Image.LANCZOS)
            name = f"maquete_{Path(relative).stem}.jpg"
            image.save(args.saida / part / label / name, quality=92)
            rows.append({"particao": part, "classe": label, "arquivo": f"{part}/{label}/{name}",
                         "fundo": "maquete", "objeto": relative, "tipo": "foto_maquete",
                         "altura_rel": "", "centro_x": "", "base_y": ""})
    if any(own.values()):
        print(f"Fotos da maquete incluídas: {sum(len(v) for v in own.values())}")

    with (args.saida / "metadados.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Base rotulada em {args.saida} ({len(rows)} imagens). Metadados: {args.saida / 'metadados.csv'}")


if __name__ == "__main__":
    main()
