"""TREINO V2 - MODELO QUE OLHA SÓ O CORREDOR, SEM COR E COM AUTOCONTRASTE

Por que existe: o primeiro modelo aprendia a "cara da cena da maquete" (muita grama, banco, latas = obstáculo;
faixa amarela limpa = livre) e errava quando o fundo mudava. Aqui o modelo:
  1. recorta só o corredor central (colunas 25% a 75% do quadrado central) - a mesma região usada na rotulagem;
  2. tira a cor (tons de cinza);
  3. normaliza brilho e contraste de cada imagem (autocontraste: média 0, desvio 1);
  4. treina com rotação de ±10° (inclinação do celular), deslocamento lateral, brilho/contraste, espelhamento
     e imagens lisas rotuladas como "sem obstáculo".
O app (src/App.tsx) faz exatamente o mesmo pré-processamento antes de chamar o modelo.

Uso (na raiz do projeto, com o ambiente do README):  python ml/treino_corredor.py
Saídas: ml/artefatos/modelo_corredor.keras, ml/artefatos/avaliacao_corredor.json e public/model/ (TensorFlow.js).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps
from scipy import ndimage

L = tf.keras.layers
SQUARE = 192                 # quadrado central
CORRIDOR = slice(48, 144)    # colunas do corredor (192x96)
SEED = 2
THRESHOLD = 0.35             # limiar de "com obstáculo" (mais sensível que 0,5, por segurança)
ART = Path("ml/artefatos")
EXPORT = Path("public/model")


def square_192(path: Path) -> np.ndarray:
    image = np.asarray(ImageOps.exif_transpose(Image.open(path)).convert("RGB"))
    side = min(image.shape[:2])
    top, left = (image.shape[0] - side) // 2, (image.shape[1] - side) // 2
    image = image[top:top + side, left:left + side].astype("float32")
    resized = tf.image.resize(image[None], [SQUARE, SQUARE], method="bilinear", antialias=False)[0].numpy()
    return np.clip(np.round(resized), 0, 255).astype("uint8")


def load_data():
    division = json.loads(Path("data/divisao.json").read_text(encoding="utf-8"))
    out = {part: ([], []) for part in ("treino", "validacao", "teste")}
    for key in ("oficial", "maquete"):
        for part, items in division[key].items():
            for relative, label in items:
                out[part][0].append(square_192(Path("data/maquete") / relative))
                out[part][1].append(int(label == "com_obstaculo"))
    return {p: (np.stack(x), np.array(y)) for p, (x, y) in out.items()}


def preprocess(x):
    """Igual ao app: corredor -> tons de cinza -> autocontraste."""
    x = tf.cast(x, tf.float32)[:, :, CORRIDOR, :]
    gray = tf.reduce_sum(x * tf.constant([0.299, 0.587, 0.114]), axis=-1, keepdims=True)
    mean = tf.reduce_mean(gray, axis=[1, 2, 3], keepdims=True)
    std = tf.math.reduce_std(gray, axis=[1, 2, 3], keepdims=True)
    return (gray - mean) / (std + 8.0)


rotation = L.RandomRotation(10 / 360, fill_mode="reflect")
shift = L.RandomTranslation(0.0, 0.06, fill_mode="reflect")


def augment(x, y):
    x = tf.cast(x, tf.float32)
    x = tf.image.random_flip_left_right(x)
    x = rotation(x, training=True)
    x = shift(x, training=True)
    x = tf.image.random_brightness(x, 0.125 * 255)
    x = tf.image.random_contrast(x, 0.6, 1.5)
    return preprocess(tf.clip_by_value(x, 0, 255)), y


def flat_images(n: int, rng) -> np.ndarray:
    """Imagens lisas (uma cor, às vezes com degradê e ruído) = nada no caminho."""
    images = []
    for _ in range(n):
        image = np.ones((SQUARE, SQUARE, 3)) * rng.integers(0, 256, 3)
        if rng.random() < 0.5:
            ramp = np.linspace(-1, 1, SQUARE)[None, :, None] * rng.uniform(0, 40)
            image = image + (ramp if rng.random() < 0.5 else ramp.transpose(1, 0, 2))
        image = image + rng.normal(0, rng.uniform(0, 4), image.shape)
        images.append(np.clip(image, 0, 255).astype("uint8"))
    return np.stack(images)


def build_model():
    inputs = L.Input(shape=(SQUARE, 96, 1), name="imagem")
    x = inputs
    for filters in (16, 32, 64, 96):
        x = L.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = L.BatchNormalization(momentum=0.9)(x)
        x = L.ReLU()(x)
        x = L.MaxPooling2D()(x)
    x = L.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = L.GlobalAveragePooling2D()(x)
    x = L.Dropout(0.3)(x)
    x = L.Dense(64, activation="relu")(x)
    outputs = L.Dense(2, activation="softmax", name="risco")(x)
    model = tf.keras.Model(inputs, outputs, name="classificador_corredor")
    model.compile(tf.keras.optimizers.Adam(1e-3), "categorical_crossentropy", metrics=["accuracy"])
    return model


def metrics(p, y, th):
    pred = (p >= th).astype(int)
    rec = ((pred == 1) & (y == 1)).sum() / max((y == 1).sum(), 1)
    spec = ((pred == 0) & (y == 0)).sum() / max((y == 0).sum(), 1)
    return {"acuracia": float((pred == y).mean()), "acuracia_balanceada": float((rec + spec) / 2),
            "obstaculos_detectados": float(rec), "alarmes_falsos": float(1 - spec)}


def main() -> None:
    tf.keras.utils.set_random_seed(SEED)
    rng = np.random.default_rng(SEED)
    data = load_data()
    (xtr, ytr), (xva, yva), (xte, yte) = data["treino"], data["validacao"], data["teste"]
    flats = flat_images(30, rng)
    xtr = np.concatenate([xtr, flats])
    ytr = np.concatenate([ytr, np.zeros(len(flats), int)])
    weights = {i: len(ytr) / (2 * (ytr == i).sum()) for i in (0, 1)}
    ds = (tf.data.Dataset.from_tensor_slices((xtr, tf.keras.utils.to_categorical(ytr, 2)))
          .shuffle(len(xtr), seed=SEED).batch(32).map(augment).prefetch(2))
    model = build_model()
    model.fit(ds, validation_data=(preprocess(xva), tf.keras.utils.to_categorical(yva, 2)), epochs=40, verbose=2,
              class_weight=weights, callbacks=[
                  tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, start_from_epoch=5),
                  tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-5)])

    prob = lambda x: model.predict(preprocess(x), verbose=0)[:, 1]
    result = {"limiar_obstaculo": THRESHOLD, "amostras_teste": int(len(yte)), "teste": metrics(prob(xte), yte, THRESHOLD)}

    # Testes de robustez (o que o modelo antigo errava)
    rs = np.random.default_rng(0)
    solid = np.stack([np.ones((SQUARE, SQUARE, 3), np.uint8) * rs.integers(0, 256, 3).astype("uint8") for _ in range(60)])
    free, clutter = xte[yte == 0], xtr[ytr == 1]
    side = free.copy()
    for i in range(len(side)):  # laterais bagunçadas, corredor intacto
        side[i, :, :48] = clutter[rs.integers(len(clutter))][:, :48]
        side[i, :, 144:] = clutter[rs.integers(len(clutter))][:, 144:]
    robust = {"imagem_lisa_livre": float((prob(solid) < THRESHOLD).mean()),
              "alarme_falso_sem_laterais_bagunçadas": float((prob(free) >= THRESHOLD).mean()),
              "alarme_falso_com_laterais_bagunçadas": float((prob(side) >= THRESHOLD).mean())}
    for name, fn in (("escuro", lambda a: a * 0.5), ("claro", lambda a: np.clip(a * 1.5 + 10, 0, 255)),
                     ("baixo_contraste", lambda a: (a - 128) * 0.5 + 128)):
        robust[f"acuracia_{name}"] = metrics(prob(np.clip(fn(xte.astype(float)), 0, 255).astype("uint8")), yte, THRESHOLD)["acuracia"]
    for angle in (-8, 8):
        rotated = np.stack([ndimage.rotate(a, angle, axes=(0, 1), reshape=False, mode="reflect", order=1) for a in xte])
        robust[f"acuracia_rotacao_{angle}"] = metrics(prob(rotated), yte, THRESHOLD)["acuracia"]
    result["robustez"] = robust
    print(json.dumps(result, indent=2, ensure_ascii=False))

    ART.mkdir(parents=True, exist_ok=True)
    model.save(ART / "modelo_corredor.keras")
    (ART / "avaliacao_corredor.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    import shutil
    import tensorflowjs as tfjs
    if EXPORT.exists():
        shutil.rmtree(EXPORT)
    EXPORT.mkdir(parents=True)
    tfjs.converters.save_keras_model(model, str(EXPORT))
    t = result["teste"]
    (EXPORT / "info.json").write_text(json.dumps({
        "classes": ["sem_obstaculo", "com_obstaculo"], "tamanho_imagem": [SQUARE, 96],
        "pre_processamento": "corredor central + tons de cinza + autocontraste",
        "acuracia_teste": round(t["acuracia"], 3), "acuracia_balanceada": round(t["acuracia_balanceada"], 3),
        "revocacao_obstaculo": round(t["obstaculos_detectados"], 3), "amostras_teste": result["amostras_teste"],
        "limiar_obstaculo": THRESHOLD}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Modelo exportado para {EXPORT}/")


if __name__ == "__main__":
    main()
