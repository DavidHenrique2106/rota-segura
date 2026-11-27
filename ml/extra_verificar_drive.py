"""EXTRA - BUSCA FOTOS E VÍDEOS NOVOS DA MAQUETE NUMA PASTA DO GOOGLE DRIVE"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path

import gdown

VIDEO = {".mp4", ".mov", ".m4v", ".avi", ".3gp"}
PHOTO = {".jpg", ".jpeg", ".png", ".heic", ".webp"}
ROOT = Path("data/maquete")
MIRROR = ROOT / "drive"
DONE = ROOT / "drive_processados.txt"


def md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", required=True)
    args = parser.parse_args()

    MIRROR.mkdir(parents=True, exist_ok=True)
    try:
        gdown.download_folder(args.url, output=str(MIRROR), quiet=True, resume=True, retries=2)
    except Exception as error:
        print(f"ERRO ao acessar o Drive: {error}")
        sys.exit(2)

    done = set(DONE.read_text().split()) if DONE.exists() else set()
    for existing in (ROOT / "videos").glob("*"):
        if existing.is_file():
            done.add(md5(existing))

    new_photos = new_videos = 0
    for path in sorted(MIRROR.rglob("*")):
        if not path.is_file():
            continue
        ext = path.suffix.lower()
        if ext not in VIDEO | PHOTO:
            continue
        digest = md5(path)
        if digest in done:
            continue
        safe = path.name.replace(" ", "_")
        if ext in VIDEO:
            (ROOT / "videos").mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, ROOT / "videos" / f"drive_{digest[:8]}_{safe}")
            new_videos += 1
        else:
            (ROOT / "sem_rotulo").mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, ROOT / "sem_rotulo" / f"drive_{digest[:8]}_{safe}")
            new_photos += 1
        done.add(digest)

    DONE.write_text("\n".join(sorted(done)) + "\n")
    print(f"NOVOS: {new_photos} fotos, {new_videos} vídeos")
    sys.exit(0 if new_photos or new_videos else 10)


if __name__ == "__main__":
    main()
