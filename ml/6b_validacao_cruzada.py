"""ETAPA 6b - VALIDAÇÃO CRUZADA EM 5 PARTES (POR GRUPOS DE RAJADA)"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from sklearn.metrics import accuracy_score, balanced_accuracy_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).parent))
from comum import ARTIFACTS_DIR, CLASSES, IMAGE_SIZE, POSITIVE, SEED, tf

spec = importlib.util.spec_from_file_location("treino", Path(__file__).parent / "5_treinamento.py")
treino = importlib.util.module_from_spec(spec)
spec.loader.exec_module(treino)
spec = importlib.util.spec_from_file_location("divisao", Path(__file__).parent / "3_divisao_treino_teste.py")
divisao = importlib.util.module_from_spec(spec)
spec.loader.exec_module(divisao)


def load(path: Path) -> np.ndarray:
    image = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    side = min(image.size)
    image = image.crop(((image.width - side) // 2, (image.height - side) // 2,
                        (image.width + side) // 2, (image.height + side) // 2)).resize((IMAGE_SIZE, IMAGE_SIZE))
    return np.asarray(image, dtype=np.float32) / 255.0


def labeled(csv_path: Path, folder: Path) -> list[tuple[Path, int]]:
    with csv_path.open(newline="", encoding="utf-8") as handle:
        return [(folder / r["arquivo"], CLASSES.index(r["rotulo"])) for r in csv.DictReader(handle) if r["rotulo"] in CLASSES]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--epocas", type=int, default=40)
    parser.add_argument("--partes", type=int, default=5)
    args = parser.parse_args()

    official = labeled(Path("data/rotulos_oficial.csv"), Path("data/maquete/oficial"))
    videos = labeled(Path("data/rotulos_maquete.csv"), Path("data/maquete/sem_rotulo")) \
        if Path("data/rotulos_maquete.csv").exists() else []
    x_off = np.stack([load(p) for p, _ in official])
    y_off = np.array([y for _, y in official])
    groups = np.array([divisao.group_of_official(p.name) for p, _ in official])
    x_vid = np.stack([load(p) for p, _ in videos]) if videos else np.zeros((0, IMAGE_SIZE, IMAGE_SIZE, 3), np.float32)
    y_vid = np.array([y for _, y in videos], dtype=int)
    print(f"Fotos oficiais: {len(y_off)} em {len(set(groups))} grupos; quadros de vídeo (sempre no treino): {len(y_vid)}")

    folds = []
    splitter = StratifiedGroupKFold(n_splits=args.partes, shuffle=True, random_state=SEED)
    for k, (train_idx, test_idx) in enumerate(splitter.split(x_off, y_off, groups), start=1):
        tf.keras.utils.set_random_seed(SEED + k)
        train_groups = sorted(set(groups[train_idx]))
        rng = np.random.default_rng(SEED + k)
        val_groups = set(rng.choice(train_groups, size=max(1, round(len(train_groups) * 0.15)), replace=False))
        val_mask = np.isin(groups[train_idx], list(val_groups))
        fit_idx, val_idx = train_idx[~val_mask], train_idx[val_mask]

        x_fit = np.concatenate([x_off[fit_idx], x_vid])
        y_fit = np.concatenate([y_off[fit_idx], y_vid])
        counts = np.bincount(y_fit, minlength=len(CLASSES))
        weights = {i: len(y_fit) / (len(CLASSES) * n) for i, n in enumerate(counts) if n}
        onehot = lambda y: tf.one_hot(y, len(CLASSES))
        fit_ds = (tf.data.Dataset.from_tensor_slices((x_fit, onehot(y_fit))).shuffle(len(y_fit), seed=SEED)
                  .batch(32).map(treino.augment).prefetch(tf.data.AUTOTUNE))
        val_ds = tf.data.Dataset.from_tensor_slices((x_off[val_idx], onehot(y_off[val_idx]))).batch(64)

        model = treino.build_model()
        model.fit(fit_ds, validation_data=val_ds, epochs=args.epocas, verbose=0, class_weight=weights, callbacks=[
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True,
                                             start_from_epoch=5),
            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-5)])

        def scores(idx):
            prob = model.predict(x_off[idx], verbose=0)[:, POSITIVE]
            pred = np.where(prob >= 0.5, POSITIVE, 1 - POSITIVE)
            y = y_off[idx]
            return {"amostras": int(len(idx)), "acuracia": float(accuracy_score(y, pred)),
                    "acuracia_balanceada": float(balanced_accuracy_score(y, pred)),
                    "revocacao_obstaculo": float(recall_score(y, pred, pos_label=POSITIVE, zero_division=0)),
                    "alarmes_falsos": float((pred[y != POSITIVE] == POSITIVE).mean()) if (y != POSITIVE).any() else None}

        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        feats = lambda x: tf.image.resize(tf.image.rgb_to_grayscale(x), (24, 24)).numpy().reshape(len(x), -1)
        simple = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.05))
        simple.fit(feats(x_fit), y_fit)
        y_test = y_off[test_idx]
        simple_pred = simple.predict(feats(x_off[test_idx]))
        fold = {"parte": k, "treino": scores(fit_idx), "teste": scores(test_idx),
                "modelo_simples_teste": {"acuracia": float(accuracy_score(y_test, simple_pred)),
                                         "acuracia_balanceada": float(balanced_accuracy_score(y_test, simple_pred)),
                                         "revocacao_obstaculo": float(recall_score(y_test, simple_pred, pos_label=POSITIVE,
                                                                                    zero_division=0)),
                                         "alarmes_falsos": None}}
        folds.append(fold)
        print(f"Parte {k}: treino {fold['treino']['acuracia']:.3f} | teste {fold['teste']['acuracia']:.3f} "
              f"(balanceada {fold['teste']['acuracia_balanceada']:.3f}, obstáculos detectados "
              f"{fold['teste']['revocacao_obstaculo']:.3f}, {fold['teste']['amostras']} fotos)", flush=True)

    def stat(part: str, key: str) -> dict:
        values = [f[part][key] for f in folds if f[part][key] is not None]
        if not values:
            return {"media": None, "desvio": None, "min": None, "max": None}
        return {"media": round(float(np.mean(values)), 4), "desvio": round(float(np.std(values)), 4),
                "min": round(float(np.min(values)), 4), "max": round(float(np.max(values)), 4)}

    summary = {"partes": args.partes, "limiar": 0.5,
               "resumo": {part: {key: stat(part, key) for key in
                                 ("acuracia", "acuracia_balanceada", "revocacao_obstaculo", "alarmes_falsos")}
                          for part in ("treino", "teste", "modelo_simples_teste")},
               "por_parte": folds}
    (ARTIFACTS_DIR / "validacao_cruzada.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                                                          encoding="utf-8")
    r = summary["resumo"]
    print(f"\nVALIDAÇÃO CRUZADA ({args.partes} partes): acurácia de treino {r['treino']['acuracia']['media']:.3f} "
          f"± {r['treino']['acuracia']['desvio']:.3f} | acurácia de teste {r['teste']['acuracia']['media']:.3f} "
          f"± {r['teste']['acuracia']['desvio']:.3f} | obstáculos detectados "
          f"{r['teste']['revocacao_obstaculo']['media']:.3f} ± {r['teste']['revocacao_obstaculo']['desvio']:.3f} | "
          f"modelo simples: acurácia {r['modelo_simples_teste']['acuracia']['media']:.3f}")


if __name__ == "__main__":
    main()
