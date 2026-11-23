"""Configurações compartilhadas pelas etapas de treino, avaliação e exportação."""

from __future__ import annotations

import os

# O conversor TensorFlow.js funciona melhor com o Keras 2 (pacote tf_keras).
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from pathlib import Path  # noqa: E402

import tensorflow as tf  # noqa: E402

CLASSES = ("seguro", "atencao", "perigo")  # mesma ordem usada no app (src/App.tsx)
IMAGE_SIZE = 160  # o app recorta o centro da câmera em quadrado e reduz para 160x160
SEED = 42

BASE_DIR = Path("data/base")
ARTIFACTS_DIR = Path("ml/artefatos")
MODEL_PATH = ARTIFACTS_DIR / "modelo.keras"


def load_split(split: str, batch_size: int = 32, shuffle: bool = False) -> tf.data.Dataset:
    """Lê data/base/<split>/<classe>/*.jpg e devolve (imagem 0..1, rótulo one-hot)."""
    directory = BASE_DIR / split
    if not directory.exists():
        raise SystemExit(f"Pasta {directory} não encontrada. Rode as etapas 1 a 3 antes.")
    dataset = tf.keras.utils.image_dataset_from_directory(
        directory,
        labels="inferred",
        label_mode="categorical",
        class_names=list(CLASSES),
        image_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=batch_size,
        shuffle=shuffle,
        seed=SEED,
    )
    normalized = dataset.map(lambda x, y: (x / 255.0, y), num_parallel_calls=tf.data.AUTOTUNE)
    normalized.file_paths = dataset.file_paths  # mesma ordem quando shuffle=False
    return normalized
