"""ETAPA 6 - AVALIAÇÃO DO MODELO (com_obstaculo × sem_obstaculo)"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix,
                             recall_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from comum import ARTIFACTS_DIR, CLASSES, MODEL_PATH, POSITIVE, load_split, tf

NAMES_PT = {"sem_obstaculo": "sem obstáculo", "com_obstaculo": "com obstáculo"}
NEGATIVE = 1 - POSITIVE
MIN_RECALL = 0.85


def to_arrays(dataset) -> tuple[np.ndarray, np.ndarray, list[str]]:
    images, labels = [], []
    for x, y in dataset:
        images.append(x.numpy())
        labels.append(np.argmax(y.numpy(), axis=1))
    names = [Path(p).name for p in dataset.file_paths]
    return np.concatenate(images), np.concatenate(labels), names


def origin_masks(names: list[str]) -> dict[str, np.ndarray]:
    starts = lambda prefix: np.array([n.startswith(prefix) for n in names])
    return {"oficial": starts("oficial_"), "videos_maquete": starts("maquete_"),
            "fotos_publicas": starts("real_"), "montagens": starts("montagem_")}


def decide(prob_obstacle: np.ndarray, threshold: float) -> np.ndarray:
    return np.where(prob_obstacle >= threshold, POSITIVE, NEGATIVE)


def choose_threshold(model) -> tuple[float, list[dict], str]:
    """Escolhe o limiar de P(com_obstaculo) na VALIDAÇÃO, de preferência só com fotos oficiais:
    entre os limiares que detectam >= 85% dos obstáculos, o de maior acurácia balanceada."""
    val_x, val_y, names = to_arrays(load_split("validacao", 64))
    official = origin_masks(names)["oficial"]
    use_official = official.any() and len(set(val_y[official])) == 2
    mask = official if use_official else np.ones_like(val_y, dtype=bool)
    probs = model.predict(val_x[mask], verbose=0)[:, POSITIVE]
    table = []
    for threshold in np.round(np.arange(0.10, 0.91, 0.05), 2):
        pred = decide(probs, threshold)
        table.append({"limiar": float(threshold),
                      "revocacao_obstaculo": round(float(recall_score(val_y[mask], pred, pos_label=POSITIVE)), 4),
                      "acuracia_balanceada": round(float(balanced_accuracy_score(val_y[mask], pred)), 4)})
    allowed = [r for r in table if r["revocacao_obstaculo"] >= MIN_RECALL] or table
    best = max(allowed, key=lambda r: (r["acuracia_balanceada"], -abs(r["limiar"] - 0.5)))
    return best["limiar"], table, "fotos oficiais da validação" if use_official else "validação completa"


def summarize(y: np.ndarray, pred: np.ndarray) -> dict:
    if len(y) == 0:
        return {}
    has_pos, has_neg = (y == POSITIVE).any(), (y == NEGATIVE).any()
    return {
        "amostras": int(len(y)),
        "acuracia": round(float(accuracy_score(y, pred)), 4),
        "acuracia_balanceada": round(float(balanced_accuracy_score(y, pred)), 4) if has_pos and has_neg else None,
        "revocacao_obstaculo": round(float((pred[y == POSITIVE] == POSITIVE).mean()), 4) if has_pos else None,
        "alarmes_falsos": round(float((pred[y == NEGATIVE] == POSITIVE).mean()), 4) if has_neg else None,
        "matriz_confusao": confusion_matrix(y, pred, labels=range(len(CLASSES))).tolist(),
    }


def baseline_accuracy(train_x, train_y, test_x, test_y) -> float:
    """Regressão logística sobre a imagem reduzida para 24x24 em tons de cinza."""

    def features(images: np.ndarray) -> np.ndarray:
        small = tf.image.resize(tf.image.rgb_to_grayscale(images), (24, 24)).numpy()
        return small.reshape(len(images), -1)

    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.05))
    model.fit(features(train_x), train_y)
    return float(balanced_accuracy_score(test_y, model.predict(features(test_x))))


def save_confusion_matrix(matrix: np.ndarray, path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [NAMES_PT[c] for c in CLASSES]
    figure, axis = plt.subplots(figsize=(5, 4.3))
    image = axis.imshow(matrix, cmap="Greens")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    n = len(CLASSES)
    axis.set(xticks=range(n), yticks=range(n), xticklabels=labels, yticklabels=labels,
             xlabel="Classe prevista pelo modelo", ylabel="Classe real", title=title)
    for row in range(n):
        for col in range(n):
            value = matrix[row, col]
            axis.text(col, row, str(value), ha="center", va="center",
                      color="white" if value > matrix.max() / 2 else "black", fontsize=13)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def save_error_examples(images, true, predicted, prob, path, priority_mask, limit=24) -> None:
    """Erros por tipo, começando pelo mais grave (obstáculo previsto como 'sem obstáculo'), e
    preferindo fotos da base oficial."""
    from PIL import Image, ImageDraw

    rng = np.random.default_rng(0)
    chosen = []
    for real_class in (POSITIVE, NEGATIVE):
        wrong = np.flatnonzero((true == real_class) & (predicted != real_class))
        ordered = list(rng.permutation(wrong[priority_mask[wrong]])) + list(rng.permutation(wrong[~priority_mask[wrong]]))
        chosen += ordered[: limit // 2]
    if not chosen:
        return
    cell, columns = 160, 6
    rows = (len(chosen) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * cell, rows * (cell + 30)), "white")
    draw = ImageDraw.Draw(sheet)
    for n, index in enumerate(chosen):
        x, y = (n % columns) * cell, (n // columns) * (cell + 30)
        sheet.paste(Image.fromarray((images[index] * 255).astype(np.uint8)), (x, y))
        draw.text((x + 4, y + cell + 2), f"real: {NAMES_PT[CLASSES[true[index]]]}", fill="black")
        draw.text((x + 4, y + cell + 15), f"previu: {NAMES_PT[CLASSES[predicted[index]]]} (P obst. {prob[index]:.0%})",
                  fill="red")
    sheet.save(path, quality=90)


def main() -> None:
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    if not MODEL_PATH.exists():
        raise SystemExit(f"Modelo {MODEL_PATH} não encontrado. Rode antes: python ml/5_treinamento.py")
    model = tf.keras.models.load_model(MODEL_PATH)

    test_x, test_y, names = to_arrays(load_split("teste", 64))
    masks = origin_masks(names)
    prob = model.predict(test_x, verbose=0)[:, POSITIVE]
    threshold, table, chosen_on = choose_threshold(model)
    predicted = decide(prob, threshold)
    argmax_pred = decide(prob, 0.5)

    official = masks["oficial"] if masks["oficial"].any() else np.ones_like(test_y, dtype=bool)
    report = classification_report(test_y[official], predicted[official], labels=range(len(CLASSES)),
                                   target_names=list(CLASSES), output_dict=True, zero_division=0)
    report_text = classification_report(test_y[official], predicted[official], labels=range(len(CLASSES)),
                                        target_names=list(CLASSES), digits=3, zero_division=0)
    matrix = confusion_matrix(test_y[official], predicted[official], labels=range(len(CLASSES)))

    print("Treinando modelo de comparação (regressão logística)...")
    train_x, train_y, train_names = to_arrays(load_split("treino", 64))
    train_prob = model.predict(train_x, verbose=0)[:, POSITIVE]
    train_pred = decide(train_prob, threshold)
    train_official = origin_masks(train_names)["oficial"]
    val_x, val_y, val_names = to_arrays(load_split("validacao", 64))
    val_pred = decide(model.predict(val_x, verbose=0)[:, POSITIVE], threshold)
    val_official = origin_masks(val_names)["oficial"]
    splits = {
        "treino": {"oficial": summarize(train_y[train_official], train_pred[train_official]),
                   "tudo": summarize(train_y, train_pred)},
        "validacao": {"oficial": summarize(val_y[val_official], val_pred[val_official]),
                      "tudo": summarize(val_y, val_pred)},
        "teste": {"oficial": summarize(test_y[official], predicted[official]), "tudo": summarize(test_y, predicted)},
    }
    baseline = baseline_accuracy(train_x, train_y, test_x[official], test_y[official])

    oficial = summarize(test_y[official], predicted[official])
    results = {
        "classes": list(CLASSES),
        "base_oficial_teste": oficial,
        "acuracia_teste": oficial.get("acuracia"),
        "acuracia_balanceada_teste": oficial.get("acuracia_balanceada"),
        "revocacao_obstaculo": oficial.get("revocacao_obstaculo"),
        "alarmes_falsos": oficial.get("alarmes_falsos"),
        "amostras_teste": oficial.get("amostras"),
        "acuracia_balanceada_modelo_simples": round(baseline, 4),
        "relatorio_por_classe": report,
        "limiar_obstaculo": threshold,
        "limiar_escolhido_em": chosen_on,
        "limiar_tabela_validacao": table,
        "sem_limiar_0_5": summarize(test_y[official], argmax_pred[official]),
        "por_origem": {name: summarize(test_y[m], predicted[m]) for name, m in masks.items() if m.any()},
        "parametros_modelo": int(model.count_params()),
        "treino_validacao_teste": splits,
    }
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS_DIR / "avaliacao.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    save_confusion_matrix(matrix, ARTIFACTS_DIR / "matriz_confusao.png", "Matriz de confusão — base oficial (teste)")
    save_error_examples(test_x, test_y, predicted, prob, ARTIFACTS_DIR / "exemplos_erros.jpg", masks["oficial"])

    fmt = lambda v: "—" if v is None else f"{v:.3f}"
    summary = (
        f"AVALIAÇÃO NO TESTE — BASE OFICIAL ({oficial.get('amostras', 0)} fotos de rajadas nunca vistas no treino)\n"
        f"Regra: 'com obstáculo' quando P(obstáculo) >= {threshold:.2f} (limiar escolhido nas {chosen_on}).\n\n"
        f"{report_text}\n"
        f"Matriz de confusão (linhas = real, colunas = previsto; ordem {', '.join(CLASSES)}):\n{matrix}\n\n"
        f"Acurácia:                         {fmt(oficial.get('acuracia'))}\n"
        f"Acurácia balanceada:              {fmt(oficial.get('acuracia_balanceada'))}\n"
        f"Obstáculos detectados (revocação): {fmt(oficial.get('revocacao_obstaculo'))}\n"
        f"Alarmes falsos (sem obst. -> com): {fmt(oficial.get('alarmes_falsos'))}\n"
        f"Modelo simples (acur. balanceada): {baseline:.3f}\n"
    )
    summary += "\nTreino × validação × teste (base oficial):\n"
    for split_name, data in splits.items():
        o = data["oficial"]
        summary += f"  {split_name:<10} acurácia {fmt(o.get('acuracia'))} ({o.get('amostras', 0)} fotos)\n"
    for name, data in results["por_origem"].items():
        summary += (f"\n{name} ({data['amostras']} imagens): acurácia {fmt(data['acuracia'])}, "
                    f"obstáculos detectados {fmt(data['revocacao_obstaculo'])}, alarmes falsos {fmt(data['alarmes_falsos'])}")
    (ARTIFACTS_DIR / "relatorio_avaliacao.txt").write_text(summary + "\n", encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
