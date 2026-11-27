"""ETAPA 1b - COLETA: QUADROS DOS VÍDEOS DA MAQUETE"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import cv2
import numpy as np

FRAMES_PER_SECOND = 3
SEGMENT_SECONDS = 5
MIN_SHARPNESS = 40.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--videos", type=Path, default=Path("data/maquete/videos"))
    parser.add_argument("--saida", type=Path, default=Path("data/maquete/sem_rotulo"))
    args = parser.parse_args()

    videos = sorted(p for p in args.videos.glob("*") if p.suffix.lower() in {".mp4", ".mov", ".m4v", ".avi"})
    if not videos:
        print(f"Nenhum vídeo em {args.videos}. Nada a fazer.")
        return
    args.saida.mkdir(parents=True, exist_ok=True)
    for old in args.saida.glob("video*_t*_q*.jpg"):
        old.unlink()

    seen: set[str] = set()
    number = 0
    total = 0
    for path in videos:
        digest = hashlib.md5(path.read_bytes()).hexdigest()
        if digest in seen:
            print(f"{path.name}: repetido, ignorado")
            continue
        seen.add(digest)
        number += 1
        capture = cv2.VideoCapture(str(path))
        fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        step = max(1, round(fps / FRAMES_PER_SECOND))
        index = kept = blurry = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if index % step == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
                if sharpness < MIN_SHARPNESS:
                    blurry += 1
                else:
                    segment = int(index / fps // SEGMENT_SECONDS)
                    name = f"video{number}_t{segment:03d}_q{index:04d}.jpg"
                    cv2.imwrite(str(args.saida / name), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
                    kept += 1
            index += 1
        capture.release()
        total += kept
        print(f"video{number} ({path.name}): {index / fps:.0f}s, {kept} quadros salvos, {blurry} borrados descartados")
    print(f"Total: {total} quadros em {args.saida}")


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
