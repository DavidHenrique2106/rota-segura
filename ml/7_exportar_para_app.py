"""ETAPA 7 - EXPORTAR O MODELO PARA O APP DO CELULAR"""

from __future__ import annotations

import json
import shutil
import warnings
from pathlib import Path

from comum import ARTIFACTS_DIR, CLASSES, IMAGE_SIZE, MODEL_PATH, tf

EXPORT_DIR = Path("public/model")


def main() -> None:
    if not MODEL_PATH.exists():
        raise SystemExit(f"Modelo {MODEL_PATH} não encontrado. Rode antes: python ml/5_treinamento.py")
    warnings.filterwarnings("ignore")
    import tensorflowjs as tfjs

    model = tf.keras.models.load_model(MODEL_PATH)
    if EXPORT_DIR.exists():
        shutil.rmtree(EXPORT_DIR)
    EXPORT_DIR.mkdir(parents=True)
    tfjs.converters.save_keras_model(model, str(EXPORT_DIR))

    info = {"classes": list(CLASSES), "tamanho_imagem": IMAGE_SIZE}
    evaluation = ARTIFACTS_DIR / "avaliacao.json"
    if evaluation.exists():
        data = json.loads(evaluation.read_text(encoding="utf-8"))
        info.update(
            acuracia_teste=data.get("acuracia_teste"),
            acuracia_balanceada=data.get("acuracia_balanceada_teste"),
            revocacao_obstaculo=data.get("revocacao_obstaculo"),
            amostras_teste=data.get("amostras_teste"),
            limiar_obstaculo=data.get("limiar_obstaculo"),
        )
    (EXPORT_DIR / "info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    size = sum(f.stat().st_size for f in EXPORT_DIR.iterdir()) / 1024
    print(f"Modelo exportado para {EXPORT_DIR}/ ({size:.0f} KB). Rode o app com: npm run dev")


if __name__ == "__main__":
    main()
