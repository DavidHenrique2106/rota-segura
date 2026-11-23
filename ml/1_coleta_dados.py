"""ETAPA 1 - COLETA DE DADOS

Baixa fotos reais de fontes públicas e extrai os recortes dos obstáculos:

* Fundos (o caminho à frente): Places365 (MIT), conjunto de validação, só
  categorias de cena por onde se caminha (corredor, saguão, viela, calçadão,
  faixa de pedestres, garagem, trilha...).
* Caixas: Open Images V7, classe "Box", usando as máscaras de segmentação
  feitas por humanos para recortar só o objeto.
* Cones: base "Junhan0518/Traffic_Cone" (Hugging Face, Apache-2.0). Usamos as
  fotos de cone em fundo branco e recortamos o cone removendo o fundo.
* Fotos reais (sem montagem), rotuladas depois na etapa 2:
  - Places365: cenas onde há móveis, pessoas e objetos no caminho (rua, praça,
    estação, escritório, sala, cozinha...) - categorias diferentes das dos fundos;
  - Open Images: fotos com cones, caixas, cadeiras, bancos, bicicletas, lixeiras,
    hidrantes, pessoas... (as caixas usadas nos recortes ficam de fora);
  - Traffic_Cone: as fotos de cone em cenas reais (as de fundo branco ficam de fora).
  Todas recortadas no quadrado central, igual ao que o app analisa.

Saída:
    data/fontes/fundos/*.jpg
    data/fontes/recortes/caixa/*.png   (RGBA, fundo transparente)
    data/fontes/recortes/cone/*.png    (RGBA, fundo transparente)
    data/fontes/fotos_reais/*.jpg      (quadradas, 256x256)

Uso:
    python ml/1_coleta_dados.py
"""

from __future__ import annotations

import argparse
import csv
import io
import shutil
import tarfile
import urllib.request
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

SEED = 42
USER_AGENT = "projeto-academico-ia/1.0"
OI = "https://storage.googleapis.com/openimages"
OI_IMAGES = "https://s3.amazonaws.com/open-images-dataset"
CONE_TAR = "https://huggingface.co/datasets/Junhan0518/Traffic_Cone/resolve/main/cone_dataset.tar.gz"

BOX = "/m/025dyy"
PLACES_TAR = "https://data.csail.mit.edu/places/places365/val_256.tar"
PLACES_LIST = "https://data.csail.mit.edu/places/places365/filelist_places365-standard.tar"
# Categorias da Places365 que representam um caminho para pedestres.
PLACES_CATEGORIES = {
    "/a/alley", "/a/arcade", "/a/atrium/public", "/b/basement", "/b/boardwalk",
    "/c/corridor", "/c/courtyard", "/c/crosswalk", "/d/driveway", "/e/elevator_lobby",
    "/e/entrance_hall", "/f/forest_path", "/l/lobby",
    "/p/parking_garage/indoor", "/p/promenade",
}
# Categorias para as FOTOS REAIS (diferentes das dos fundos, para não repetir foto).
REAL_PLACES_CATEGORIES = {
    "/s/street", "/p/plaza", "/p/patio", "/p/parking_lot", "/b/bus_station/indoor",
    "/s/subway_station/platform", "/t/train_station/platform", "/m/museum/indoor",
    "/l/library/indoor", "/o/office", "/l/living_room", "/k/kitchen", "/d/dining_room",
    "/c/classroom", "/w/waiting_room", "/s/storage_room", "/s/shopping_mall/indoor",
    "/s/supermarket", "/c/campus", "/p/playground",
}
# Classes do Open Images que funcionam como obstáculos no caminho.
OI_OBSTACLES = {
    "/m/025dyy": "caixa", "/m/02zn6n": "barril", "/m/0cvnqh": "banco", "/m/0199g": "bicicleta",
    "/m/01mzpv": "cadeira", "/m/02crq1": "sofa", "/m/01y9k5": "mesa", "/m/04bcr3": "mesa",
    "/m/01pns0": "hidrante", "/m/0fm3zh": "vaso", "/m/03fp41": "planta", "/m/015qbp": "parquimetro",
    "/m/01g317": "pessoa", "/m/01s55n": "mala", "/m/0bjyj5": "lixeira", "/m/0bt9lr": "cachorro",
}
TRAFFIC_CONE = "/m/03sy7v"
PER_OBSTACLE_CLASS = 90
SPLITS = {"validation": "val", "test": "test"}

# Revisão manual (olhando as folhas de contato dos recortes): removidos os cones
# com marca d'água/pilhas caídas e as caixas com máscara incompleta.
REVISAO_EXCLUIR = {
    "cone": {"135", "2", "24", "25", "26", "4", "501", "57", "60", "65"},
    "caixa": {
        "21eac74cc306ee57_39c75455", "346cccdac97c2c50_7724495e", "35e80ba5130fe6fe_2bc6210c",
        "5c51474c5efecf0d_03b28c20", "5c51474c5efecf0d_568b0ffb", "5c51474c5efecf0d_9a19fc43",
        "90f57dad1ca73137_a3d7a82c", "90f57dad1ca73137_a6029d4d", "cba95582b3b15d63_700d46e0",
        "d2269508442873e0_5d007f67", "d2269508442873e0_68f8d192", "d2269508442873e0_92a506a0",
        "d2269508442873e0_a8336549", "d2269508442873e0_a9c1d45e", "d2269508442873e0_c3da14dc",
        "d2269508442873e0_e858bbf7", "d2269508442873e0_f5eb856a",
    },
}


def fetch(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def fetch_cached(url: str, path: Path) -> Path:
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"  baixando {url.rsplit('/', 1)[-1]} ...")
    partial = path.with_suffix(path.suffix + ".parcial")
    for attempt in range(1, 4):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=180) as response, partial.open("wb") as handle:
                shutil.copyfileobj(response, handle, 1 << 20)
            partial.rename(path)
            return path
        except OSError as error:
            print(f"  falha no download (tentativa {attempt}/3): {error}")
    raise SystemExit(f"Não foi possível baixar {url}. Verifique a internet e rode de novo.")


def shrink(image: Image.Image, max_side: int) -> Image.Image:
    scale = max_side / max(image.size)
    if scale < 1:
        image = image.resize((round(image.width * scale), round(image.height * scale)), Image.LANCZOS)
    return image


def tight_crop(rgba: Image.Image) -> Image.Image | None:
    bbox = rgba.getchannel("A").getbbox()
    return rgba.crop(bbox) if bbox else None


def save_clean(image: Image.Image, target: Path, **options) -> None:
    """Salva sem metadados (alguns perfis de cor ICC embutidos são enormes)."""
    image.info.clear()
    image.save(target, **options)


def center_square(image: Image.Image, size: int = 256) -> Image.Image:
    side = min(image.size)
    left, top = (image.width - side) // 2, (image.height - side) // 2
    return image.crop((left, top, left + side, top + side)).resize((size, size), Image.LANCZOS)


# --------------------------------------------------------------------- fundos
def collect_backgrounds(cache: Path, out_dir: Path, places: set[str] = PLACES_CATEGORIES, prefix: str = "") -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    lists = fetch_cached(PLACES_LIST, cache / "filelist_places365-standard.tar")
    with tarfile.open(lists) as tar:
        categories = dict(
            line.split() for line in tar.extractfile("categories_places365.txt").read().decode().splitlines()
        )
        wanted = {categories[name] for name in places}
        val_list = tar.extractfile("places365_val.txt").read().decode().splitlines()
    names = {}
    for line in val_list:
        file_name, category = line.split()
        if category in wanted:
            names[file_name] = category
    archive = fetch_cached(PLACES_TAR, cache / "places365_val_256.tar")
    category_names = {index: name for name, index in categories.items()}
    ok = 0
    with tarfile.open(archive) as tar:
        for member in tar:
            file_name = Path(member.name).name
            if file_name not in names:
                continue
            target = out_dir / f"{prefix}{category_names[names[file_name]].strip('/').replace('/', '-')}_{file_name}"
            if not target.exists():
                save_clean(Image.open(tar.extractfile(member)).convert("RGB"), target, quality=92)
            ok += 1
    print(f"Fotos Places365 salvas em {out_dir.name}: {ok} ({len(places)} categorias de cena)")


# --------------------------------------------------------------------- caixas
def collect_boxes(cache: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[tuple[str, dict[str, str]]] = []
    for split in SPLITS:
        seg_csv = fetch_cached(f"{OI}/v5/{split}-annotations-object-segmentation.csv", cache / f"{split}-seg.csv")
        with seg_csv.open(newline="") as handle:
            for row in csv.DictReader(handle):
                if row["LabelName"] != BOX:
                    continue
                x0, x1, y0, y1 = (float(row[k]) for k in ("BoxXMin", "BoxXMax", "BoxYMin", "BoxYMax"))
                touches_edge = x0 < 0.01 or y0 < 0.01 or x1 > 0.99 or y1 > 0.99
                if touches_edge or (x1 - x0) * (y1 - y0) < 0.02:
                    continue  # objeto cortado pela borda ou pequeno demais
                rows.append((split, row))
    print(f"Caixas: {len(rows)} máscaras candidatas")

    masks: dict[str, bytes] = {}
    needed = defaultdict(set)
    for split, row in rows:
        needed[(split, row["MaskPath"][0])].add(row["MaskPath"])
    for (split, prefix), names in sorted(needed.items()):
        archive = fetch_cached(f"{OI}/v5/{split}-masks/{split}-masks-{prefix}.zip", cache / "masks" / f"{split}-{prefix}.zip")
        with zipfile.ZipFile(archive) as zipped:
            for name in names:
                masks[name] = zipped.read(name)

    def cut(item: tuple[str, dict[str, str]]) -> bool:
        split, row = item
        target = out_dir / f"{row['ImageID']}_{row['BoxID']}.png"
        if target.exists():
            return True
        try:
            image = Image.open(io.BytesIO(fetch(f"{OI_IMAGES}/{split}/{row['ImageID']}.jpg"))).convert("RGB")
        except Exception:
            return False
        mask = Image.open(io.BytesIO(masks[row["MaskPath"]])).convert("L").resize(image.size, Image.NEAREST)
        mask_array = np.array(mask) > 127
        mask_array = ndimage.binary_opening(mask_array, iterations=2)
        labels, count = ndimage.label(mask_array)
        if count == 0:
            return False
        largest = labels == (np.argmax(np.bincount(labels.ravel())[1:]) + 1)
        width, height = image.size
        x0, x1 = float(row["BoxXMin"]) * width, float(row["BoxXMax"]) * width
        y0, y1 = float(row["BoxYMin"]) * height, float(row["BoxYMax"]) * height
        if largest.sum() < 0.45 * (x1 - x0) * (y1 - y0):
            return False  # máscara muito incompleta
        alpha = Image.fromarray((ndimage.gaussian_filter(largest.astype(float), 1.0) * 255).astype(np.uint8))
        rgba = image.copy()
        rgba.putalpha(alpha)
        cropped = tight_crop(rgba)
        if cropped is None or min(cropped.size) < 60:
            return False
        save_clean(shrink(cropped, 512), target)
        return True

    with ThreadPoolExecutor(16) as pool:
        ok = sum(pool.map(cut, rows))
    print(f"Recortes de caixa salvos: {ok}")


# ---------------------------------------------------------------------- cones
def white_background_cutout(image: Image.Image) -> Image.Image | None:
    """Remove fundo branco (foto de produto) e devolve o cone em RGBA."""
    rgb = np.asarray(image.convert("RGB")).astype(np.int16)
    border = np.concatenate([rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]])
    if (border.min(axis=1) > 225).mean() < 0.92:
        return None  # não é foto com fundo branco
    near_white = (rgb.min(axis=2) > 215) & (rgb.max(axis=2) - rgb.min(axis=2) < 25)
    regions, _ = ndimage.label(near_white)
    edge_ids = np.unique(np.concatenate([regions[0], regions[-1], regions[:, 0], regions[:, -1]]))
    background = np.isin(regions, edge_ids[edge_ids > 0])
    foreground = ndimage.binary_opening(~background, iterations=2)
    labels, count = ndimage.label(foreground)
    if count == 0:
        return None
    sizes = np.bincount(labels.ravel())[1:]
    keep = labels == (np.argmax(sizes) + 1)
    keep = ndimage.binary_fill_holes(keep)
    if keep.mean() < 0.05:
        return None
    # O objeto principal precisa ser laranja/vermelho (um cone de verdade).
    hsv = np.asarray(image.convert("HSV")).astype(np.int16)
    hue, sat = hsv[..., 0][keep], hsv[..., 1][keep]
    orange = (((hue < 25) | (hue > 240)) & (sat > 110)).mean()
    if orange < 0.30:
        return None
    alpha = Image.fromarray((ndimage.gaussian_filter(keep.astype(float), 0.8) * 255).astype(np.uint8))
    rgba = image.convert("RGB")
    rgba.putalpha(alpha)
    return tight_crop(rgba)


def collect_cones(cache: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    archive = fetch_cached(CONE_TAR, cache / "cone_dataset.tar.gz")
    ok = 0
    with tarfile.open(archive) as tar:
        for member in sorted(tar.getmembers(), key=lambda m: m.name):
            if not member.name.lower().endswith(".jpg"):
                continue
            image = Image.open(io.BytesIO(tar.extractfile(member).read()))
            cutout = white_background_cutout(image)
            if cutout is None or min(cutout.size) < 60:
                continue
            aspect = cutout.height / cutout.width
            if not 0.6 <= aspect <= 2.2:
                continue  # vários cones lado a lado ou recorte ruim
            save_clean(shrink(cutout, 512), out_dir / f"{Path(member.name).stem}.png")
            ok += 1
    print(f"Recortes de cone salvos: {ok}")


# --------------------------------------------------------------- fotos reais
def collect_real_photos(cache: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    collect_backgrounds(cache, out_dir, REAL_PLACES_CATEGORIES, prefix="places_")

    # Open Images: imagens com obstáculos anotados (caixas usadas nos recortes ficam de fora).
    used_for_cutouts: set[str] = set()
    for split in SPLITS:
        seg_csv = fetch_cached(f"{OI}/v5/{split}-annotations-object-segmentation.csv", cache / f"{split}-seg.csv")
        with seg_csv.open(newline="") as handle:
            used_for_cutouts |= {r["ImageID"] for r in csv.DictReader(handle) if r["LabelName"] == BOX}
    chosen: dict[tuple[str, str], str] = {}
    per_class: dict[str, int] = defaultdict(int)
    for split, short in SPLITS.items():
        bbox_csv = fetch_cached(f"{OI}/v5/{split}-annotations-bbox.csv", cache / f"{split}-bbox.csv")
        with bbox_csv.open(newline="") as handle:
            rows = sorted(csv.DictReader(handle), key=lambda r: (r["ImageID"], r["LabelName"]))
        for row in rows:
            name = OI_OBSTACLES.get(row["LabelName"])
            if not name or row["ImageID"] in used_for_cutouts or (split, row["ImageID"]) in chosen:
                continue
            if float(row["IsGroupOf"] or 0) or per_class[name] >= PER_OBSTACLE_CLASS:
                continue
            if float(row["YMax"]) - float(row["YMin"]) < 0.25:
                continue  # objeto pequeno: não acrescenta exemplos de obstáculo próximo
            chosen[(split, row["ImageID"])] = name
            per_class[name] += 1
        labels_csv = fetch_cached(f"{OI}/v7/oidv7-{short}-annotations-human-imagelabels.csv", cache / f"{short}-labels.csv")
        with labels_csv.open(newline="") as handle:
            for row in csv.DictReader(handle):
                if row["LabelName"] == TRAFFIC_CONE and row["Confidence"] == "1":
                    chosen[(split, row["ImageID"])] = "cone"
    print(f"Open Images: {len(chosen)} fotos com obstáculos ({dict(sorted(per_class.items()))})")

    def download(item: tuple[tuple[str, str], str]) -> bool:
        (split, image_id), name = item
        target = out_dir / f"oi_{name}_{image_id}.jpg"
        if target.exists():
            return True
        try:
            image = Image.open(io.BytesIO(fetch(f"{OI_IMAGES}/{split}/{image_id}.jpg"))).convert("RGB")
        except Exception:
            return False
        save_clean(center_square(image), target, quality=92)
        return True

    with ThreadPoolExecutor(16) as pool:
        ok = sum(pool.map(download, sorted(chosen.items())))
    print(f"Fotos Open Images salvas: {ok}")

    # Fotos de cone em cenas reais (as de fundo branco viraram recortes).
    archive = fetch_cached(CONE_TAR, cache / "cone_dataset.tar.gz")
    ok = 0
    with tarfile.open(archive) as tar:
        for member in sorted(tar.getmembers(), key=lambda m: m.name):
            if not member.name.lower().endswith(".jpg"):
                continue
            image = Image.open(io.BytesIO(tar.extractfile(member).read())).convert("RGB")
            rgb = np.asarray(image).astype(np.int16)
            border = np.concatenate([rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]])
            if (border.min(axis=1) > 225).mean() > 0.5 or min(image.size) < 200:
                continue  # foto de produto em fundo branco ou pequena demais
            save_clean(center_square(image), out_dir / f"hf_cone_{Path(member.name).stem}.jpg", quality=92)
            ok += 1
    print(f"Fotos reais de cone salvas: {ok}")


def apply_review(recortes: Path) -> None:
    for kind, stems in REVISAO_EXCLUIR.items():
        for stem in stems:
            (recortes / kind / f"{stem}.png").unlink(missing_ok=True)
    for kind in REVISAO_EXCLUIR:
        print(f"Recortes de {kind} após revisão manual: {len(list((recortes / kind).glob('*.png')))}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--saida", type=Path, default=Path("data/fontes"))
    parser.add_argument("--refazer", action="store_true", help="baixa tudo de novo mesmo se os dados já existirem")
    args = parser.parse_args()
    out = args.saida
    cache = out / "_cache"
    done = lambda folder, pattern: not args.refazer and any((out / folder).glob(pattern))  # noqa: E731
    if not done("recortes/cone", "*.png"):
        collect_cones(cache, out / "recortes" / "cone")
    if not done("recortes/caixa", "*.png"):
        collect_boxes(cache, out / "recortes" / "caixa")
    if not done("fundos", "*.jpg"):
        collect_backgrounds(cache, out / "fundos")
    apply_review(out / "recortes")
    if not done("fotos_reais", "*.jpg"):
        collect_real_photos(cache, out / "fotos_reais")
    print(f"Fotos reais: {len(list((out / 'fotos_reais').glob('*.jpg')))}")
    print("Coleta concluída (o que já existia foi mantido; use --refazer para baixar tudo de novo).")

if __name__ == "__main__":
    main()
