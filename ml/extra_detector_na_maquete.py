"""EXTRA - TESTE DO ROTULADOR (OWLv2) NAS FOTOS DA MAQUETE"""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from transformers import Owlv2ForObjectDetection, Owlv2Processor

DETECTOR = "google/owlv2-base-patch16-ensemble"
QUERIES = ["a toy traffic cone", "a road barrier", "a can", "a jar", "a toy figure", "a toy car", "a plant"]
NAMES = ["cone", "barreira", "lata", "pote", "boneco", "carro", "planta"]


def main() -> None:
    files, seen = [], set()
    for path in sorted(Path("data/maquete/visao_geral").glob("*")):
        digest = hashlib.md5(path.read_bytes()).hexdigest()
        if digest not in seen:
            seen.add(digest)
            files.append(path)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    processor = Owlv2Processor.from_pretrained(DETECTOR)
    model = Owlv2ForObjectDetection.from_pretrained(DETECTOR).to(device).eval()

    cell, columns = 400, 3
    chosen = files[:6]
    sheet = Image.new("RGB", (columns * cell, ((len(chosen) + columns - 1) // columns) * cell), "white")
    for n, path in enumerate(chosen):
        image = Image.open(path).convert("RGB")
        side = max(image.size)
        inputs = processor(text=[QUERIES], images=image, return_tensors="pt").to(device)
        with torch.inference_mode():
            outputs = model(**inputs)
        result = processor.post_process_object_detection(outputs, threshold=0.2,
                                                         target_sizes=torch.tensor([[side, side]]))[0]
        draw = ImageDraw.Draw(image)
        width = max(2, image.width // 300)
        for box, label in zip(result["boxes"].tolist(), result["labels"].tolist()):
            draw.rectangle(box, outline="red", width=width)
            draw.text((box[0] + 3, box[1] + 3), NAMES[label], fill="red")
        image.thumbnail((cell - 6, cell - 6))
        sheet.paste(image, ((n % columns) * cell + 3, (n // columns) * cell + 3))
        print(path.name, sorted({NAMES[label] for label in result["labels"].tolist()}))
    output = Path("ml/artefatos/maquete_deteccao.jpg")
    sheet.save(output, quality=88)
    print(f"Figura salva em {output}")


if __name__ == "__main__":
    main()
