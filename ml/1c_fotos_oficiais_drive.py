"""ETAPA 1c - COLETA: BASE OFICIAL (FOTOS DA MAQUETE NO GOOGLE DRIVE)"""

from __future__ import annotations

import argparse
from pathlib import Path

import gdown
from PIL import Image, ImageOps

try:
    from pillow_heif import register_heif_opener

    register_heif_opener()
except ImportError:
    pass

URL = "https://drive.google.com/drive/folders/1Mpdb1Dk8PKUlx3vIiX9qZwu6G_Gm4WZG"
MIRROR = Path("data/maquete/drive_oficial")
OUTPUT = Path("data/maquete/oficial")
PHOTO = {".jpg", ".jpeg", ".png", ".heic", ".webp"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default=URL)
    parser.add_argument("--sem-download", action="store_true", help="só converte o que já foi baixado")
    args = parser.parse_args()

    MIRROR.mkdir(parents=True, exist_ok=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if not args.sem_download:
        for attempt in range(3):
            try:
                gdown.download_folder(args.url, output=str(MIRROR), quiet=True, resume=True, retries=3)
                break
            except Exception as error:
                print(f"  tentativa {attempt + 1}: {str(error).splitlines()[0]}")

    new = 0
    for path in sorted(MIRROR.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in PHOTO:
            continue
        target = OUTPUT / f"{path.stem}.jpg"
        if target.exists():
            continue
        try:
            image = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
        except Exception:
            print(f"  ignorada (arquivo incompleto ou inválido): {path.name}")
            continue
        image.thumbnail((1024, 1024), Image.LANCZOS)
        image.save(target, quality=90)
        new += 1
    total = len(list(OUTPUT.glob("*.jpg")))
    print(f"Base oficial: {total} fotos em {OUTPUT} ({new} novas)")


if __name__ == "__main__":
    main()
