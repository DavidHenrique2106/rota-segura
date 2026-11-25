"""ETAPA 4 - MONTAGEM DA BASE ROTULADA (sem_obstaculo / com_obstaculo)

Junta, em cada partição (treino/validacao/teste) definida na etapa 3:

  * BASE OFICIAL: fotos do grupo na maquete (Drive), rotuladas na etapa 2.
    Arquivos "oficial_*". É nelas que o desempenho oficial é medido.
  * Quadros dos vídeos da maquete. Arquivos "maquete_*".
  * Fontes PÚBLICAS, só se ajudarem (a etapa 5b testa cada combinação):
      - fotos reais da internet rotuladas na etapa 2 ("real_*");
      - montagens: um cone ou caixa real colado numa foto real de caminho, na
        região central ("montagem_*"); sem objeto = sem_obstaculo.

Saída:
    <saida>/<particao>/<classe>/*.jpg  e  <saida>/metadados.csv

Uso:
    python ml/4_montar_base_rotulada.py [--publicas nenhuma|reais|montagens|todas] [--saida data/base]
"""

from __future__ import annotations

import argparse
import os
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
CLASSES = ("sem_obstaculo", "com_obstaculo")
MONTAGENS_POR_CLASSE = {"treino": 600, "validacao": 120, "teste": 120}

# Montagem "com obstáculo": (altura relativa mín., máx.), (base do objeto y mín., máx.), (centro x mín., máx.)
# O objeto fica no corredor central (o caminho), de longe (8% da altura) até muito perto (85%).
RULE_OBSTACLE = ((0.08, 0.85), (0.55, 1.06), (0.30, 0.70))


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
    if label == "com_obstaculo":
        kind = rng.choice(("cone", "caixa"))
        obj_index = rng.randrange(len(src.objects[kind]))
        meta.update(tipo=kind, objeto=src.names[kind][obj_index])
        meta.update(paste_object(rng, canvas, src.objects[kind][obj_index], RULE_OBSTACLE))
    return camera_effects(rng, canvas), meta


def square_photo(path: Path) -> Image.Image:
    image = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    side = min(image.size)
    left, top = (image.width - side) // 2, (image.height - side) // 2
    return image.crop((left, top, left + side, top + side)).resize((SIZE, SIZE), Image.LANCZOS)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fontes", type=Path, default=Path("data/fontes"))
    parser.add_argument("--divisao", type=Path, default=Path("data/divisao.json"))
    parser.add_argument("--saida", type=Path, default=Path(os.environ.get("ROTA_BASE", "data/base")))
    parser.add_argument("--publicas", choices=("nenhuma", "reais", "montagens", "todas"),
                        default=os.environ.get("ROTA_PUBLICAS", "todas"),
                        help="quais fontes públicas entram (a etapa 5b escolhe a melhor)")
    args = parser.parse_args()

    use_real = args.publicas in ("reais", "todas")
    use_comp = args.publicas in ("montagens", "todas")
    division = json.loads(args.divisao.read_text(encoding="utf-8"))
    if args.saida.exists():
        shutil.rmtree(args.saida)
    for part in MONTAGENS_POR_CLASSE:
        for label in CLASSES:
            (args.saida / part / label).mkdir(parents=True, exist_ok=True)

    rows = []

    def add(part: str, label: str, name: str, image: Image.Image, kind: str, source: str, quality: int = 92):
        image.save(args.saida / part / label / name, quality=quality)
        rows.append({"particao": part, "classe": label, "arquivo": f"{part}/{label}/{name}", "fundo": source,
                     "objeto": "", "tipo": kind, "altura_rel": "", "centro_x": "", "base_y": ""})

    # Base oficial e quadros dos vídeos da maquete: sempre entram.
    for key, folder, prefix, kind in (("oficial", "", "oficial_", "foto_oficial"),
                                      ("maquete", "", "maquete_", "foto_maquete")):
        for part, items in division.get(key, {}).items():
            for relative, label in items:
                add(part, label, f"{prefix}{Path(relative).stem}.jpg", square_photo(Path("data/maquete") / relative),
                    kind, relative)

    real_labels = {}
    if use_real:
        with Path("data/rotulos_automaticos.csv").open(newline="", encoding="utf-8") as handle:
            real_labels = {f"fotos_reais/{r['arquivo']}": r["rotulo"] for r in csv.DictReader(handle)}

    for part_index, part in enumerate(MONTAGENS_POR_CLASSE):
        if use_real:
            for relative in division["fotos_reais"][part]:
                label = real_labels.get(relative)
                if label in CLASSES:
                    image = Image.open(args.fontes / relative).convert("RGB").resize((SIZE, SIZE), Image.LANCZOS)
                    add(part, label, f"real_{Path(relative).stem}.jpg", image, "foto_real_rotulada", relative)
        if use_comp:
            src = Sources(args.fontes, division, part)
            rng = random.Random(SEED + part_index)
            for label in CLASSES:
                for i in range(MONTAGENS_POR_CLASSE[part]):
                    image, meta = make_example(rng, src, label)
                    name = f"montagem_{label}_{i:04d}.jpg"
                    image.save(args.saida / part / label / name, quality=rng.randint(75, 95))
                    rows.append({"particao": part, "classe": label, "arquivo": f"{part}/{label}/{name}", **meta})

    with (args.saida / "metadados.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    from collections import Counter
    counts = Counter((r["particao"], r["tipo"].replace("caixa", "montagem").replace("cone", "montagem")
                      .replace("nenhum", "montagem"), r["classe"]) for r in rows)
    print(f"Base em {args.saida} (fontes públicas: {args.publicas}) - {len(rows)} imagens")
    for part in MONTAGENS_POR_CLASSE:
        detail = ", ".join(f"{kind}/{label.split('_')[0]}={n}" for (p_, kind, label), n in sorted(counts.items()) if p_ == part)
        print(f"  {part:<10} {detail}")


if __name__ == "__main__":
    main()
