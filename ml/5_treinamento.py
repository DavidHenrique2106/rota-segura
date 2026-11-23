"""ETAPA 5 - TREINAMENTO DO MODELO

Treina DO ZERO (pesos aleatórios, sem modelo pré-treinado) uma rede neural
convolucional (CNN) que recebe a imagem da câmera e devolve a probabilidade de
cada classe: seguro, atencao, perigo.

Arquitetura (≈ 200 mil parâmetros, leve para rodar no celular):

    entrada 160x160x3
    4 blocos [Conv 3x3 -> BatchNorm -> ReLU -> MaxPool]  (16, 32, 64, 96 filtros)
    Conv 3x3 (128) -> GlobalAveragePooling -> Dropout 0.3
    Dense 64 ReLU -> Dense 3 Softmax

Durante o treino as imagens recebem pequenas variações aleatórias (espelhamento,
brilho, contraste, saturação), que imitam celulares e iluminações diferentes. Usamos o conjunto de VALIDAÇÃO
para parar o treino na melhor época (EarlyStopping). O conjunto de TESTE não é
usado aqui.

Saída:
    ml/artefatos/modelo.keras
    ml/artefatos/historico_treino.csv
    ml/artefatos/curvas_treino.png

Uso:
    python ml/5_treinamento.py [--epocas 40]
"""

from __future__ import annotations

import argparse
import csv
import random
import time

import numpy as np

from comum import ARTIFACTS_DIR, CLASSES, IMAGE_SIZE, MODEL_PATH, SEED, load_split, tf


def augment(image: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    # Só variações que NÃO mudam o rótulo: espelhar para os lados e mexer na
    # cor/luz. Zoom ou deslocamento vertical mudariam o tamanho/posição do
    # obstáculo, que é justamente o que define a classe.
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, 0.12)
    image = tf.image.random_contrast(image, 0.8, 1.2)
    image = tf.image.random_saturation(image, 0.75, 1.25)
    return tf.clip_by_value(image, 0.0, 1.0), label


def build_model() -> tf.keras.Model:
    layers = tf.keras.layers
    inputs = layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3), name="imagem")
    x = inputs
    for filters in (16, 32, 64, 96):
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation="relu")(x)
    outputs = layers.Dense(len(CLASSES), activation="softmax", name="risco")(x)
    model = tf.keras.Model(inputs, outputs, name="classificador_de_risco")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def save_curves(history_csv, output_png) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with open(history_csv, newline="") as handle:
        rows = list(csv.DictReader(handle))
    epochs = [int(r["epoch"]) + 1 for r in rows]
    figure, (left, right) = plt.subplots(1, 2, figsize=(10, 4))
    left.plot(epochs, [float(r["loss"]) for r in rows], label="treino")
    left.plot(epochs, [float(r["val_loss"]) for r in rows], label="validação")
    left.set(title="Perda (loss)", xlabel="época")
    right.plot(epochs, [float(r["accuracy"]) for r in rows], label="treino")
    right.plot(epochs, [float(r["val_accuracy"]) for r in rows], label="validação")
    right.set(title="Acurácia", xlabel="época", ylim=(0, 1))
    for axis in (left, right):
        axis.grid(alpha=0.3)
        axis.legend()
    figure.tight_layout()
    figure.savefig(output_png, dpi=140)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--epocas", type=int, default=40)
    parser.add_argument("--lote", type=int, default=32)
    args = parser.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)
    tf.keras.utils.set_random_seed(SEED)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    train = (
        load_split("treino", args.lote, shuffle=True)
        .map(augment, num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE)
    )
    validation = load_split("validacao", args.lote).cache().prefetch(tf.data.AUTOTUNE)

    model = build_model()
    model.summary()
    history_csv = ARTIFACTS_DIR / "historico_treino.csv"
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5),
        tf.keras.callbacks.CSVLogger(str(history_csv)),
    ]
    start = time.time()
    history = model.fit(train, validation_data=validation, epochs=args.epocas, callbacks=callbacks, verbose=2)
    best = int(np.argmin(history.history["val_loss"]))
    print(
        f"Treino concluído em {time.time() - start:.0f}s. Melhor época: {best + 1} "
        f"(acurácia de validação {history.history['val_accuracy'][best]:.3f})"
    )

    model.save(MODEL_PATH)
    save_curves(history_csv, ARTIFACTS_DIR / "curvas_treino.png")
    print(f"Modelo salvo em {MODEL_PATH}")


if __name__ == "__main__":
    main()
