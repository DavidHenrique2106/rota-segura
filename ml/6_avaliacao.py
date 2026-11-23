"""ETAPA 6 - AVALIAÇÃO DO MODELO

Avalia o modelo treinado SOMENTE no conjunto de TESTE (fotos de chão, cones e
caixas que a rede nunca viu no treino).

Gera:
    * acurácia, precisão, revocação (recall) e F1 por classe;
    * matriz de confusão;
    * comparação com um modelo de base simples (regressão logística sobre os
      pixels), para mostrar quanto a CNN acrescenta;
    * folha com exemplos de erros, para análise qualitativa.

A métrica mais importante para a segurança é a REVOCAÇÃO DE "perigo": de todas
as situações realmente perigosas, quantas o modelo detectou.

Saída (ml/artefatos/):
    avaliacao.json, relatorio_avaliacao.txt, matriz_confusao.png, exemplos_erros.jpg

Uso:
    python ml/6_avaliacao.py
"""

from __future__ import annotations

import json
import warnings

from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from comum import ARTIFACTS_DIR, CLASSES, MODEL_PATH, load_split, tf

NAMES_PT = {"seguro": "seguro", "atencao": "atenção", "perigo": "perigo"}


def to_arrays(dataset: tf.data.Dataset) -> tuple[np.ndarray, np.ndarray]:
    images, labels = [], []
    for x, y in dataset:
        images.append(x.numpy())
        labels.append(np.argmax(y.numpy(), axis=1))
    return np.concatenate(images), np.concatenate(labels)


def baseline_accuracy(train_x, train_y, test_x, test_y) -> float:
    """Regressão logística sobre a imagem reduzida para 24x24 em tons de cinza."""

    def features(images: np.ndarray) -> np.ndarray:
        small = tf.image.resize(tf.image.rgb_to_grayscale(images), (24, 24)).numpy()
        return small.reshape(len(images), -1)

    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.05))
    model.fit(features(train_x), train_y)
    return float(accuracy_score(test_y, model.predict(features(test_x))))


PERIGO = CLASSES.index("perigo")


def apply_threshold(probabilities: np.ndarray, threshold: float | None) -> np.ndarray:
    """Avisa 'perigo' quando P(perigo) >= limiar; senão escolhe a mais provável entre as outras."""
    if threshold is None:
        return probabilities.argmax(axis=1)
    others = probabilities.copy()
    others[:, PERIGO] = -1
    return np.where(probabilities[:, PERIGO] >= threshold, PERIGO, others.argmax(axis=1))


def choose_threshold(model) -> tuple[float, list]:
    """Escolhe o limiar de 'perigo' no conjunto de VALIDAÇÃO (nunca no teste).

    Critério: entre os limiares que detectam pelo menos 75% dos perigos (revocação >= 0,75), o de maior F1
    médio das três classes. Assim priorizamos não deixar passar perigos sem transformar tudo em alarme falso."""
    val_x, val_y = to_arrays(load_split("validacao", 64))
    probs = model.predict(val_x, verbose=0)
    table = []
    for threshold in np.round(np.arange(0.15, 0.61, 0.05), 2):
        pred = apply_threshold(probs, threshold)
        recall = recall_score(val_y, pred, labels=[PERIGO], average="macro", zero_division=0)
        macro_f1 = f1_score(val_y, pred, average="macro")
        table.append((float(threshold), round(float(recall), 4), round(float(macro_f1), 4),
                      round(float(accuracy_score(val_y, pred)), 4)))
    allowed = [row for row in table if row[1] >= 0.75] or table
    best = max(allowed, key=lambda row: row[2])
    return best[0], table


def save_confusion_matrix(matrix: np.ndarray, path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [NAMES_PT[c] for c in CLASSES]
    figure, axis = plt.subplots(figsize=(5.6, 4.8))
    image = axis.imshow(matrix, cmap="Greens")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    axis.set(
        xticks=range(3), yticks=range(3), xticklabels=labels, yticklabels=labels,
        xlabel="Classe prevista pelo modelo", ylabel="Classe real",
        title="Matriz de confusão — conjunto de teste",
    )
    for row in range(3):
        for col in range(3):
            value = matrix[row, col]
            axis.text(col, row, str(value), ha="center", va="center",
                      color="white" if value > matrix.max() / 2 else "black", fontsize=12)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def save_error_examples(images, true, predicted, confidence, path, limit=24) -> None:
    from PIL import Image, ImageDraw

    wrong = np.flatnonzero(true != predicted)[:limit]
    if len(wrong) == 0:
        return
    cell = 160
    columns = 6
    rows = (len(wrong) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * cell, rows * (cell + 30)), "white")
    draw = ImageDraw.Draw(sheet)
    for n, index in enumerate(wrong):
        x, y = (n % columns) * cell, (n // columns) * (cell + 30)
        sheet.paste(Image.fromarray((images[index] * 255).astype(np.uint8)), (x, y))
        draw.text((x + 4, y + cell + 2), f"real: {CLASSES[true[index]]}", fill="black")
        draw.text((x + 4, y + cell + 15), f"previu: {CLASSES[predicted[index]]} ({confidence[index]:.0%})", fill="red")
    sheet.save(path, quality=90)


def main() -> None:
    # Avisos numéricos espúrios do numpy/Accelerate no macOS durante a regressão logística.
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    if not MODEL_PATH.exists():
        raise SystemExit(f"Modelo {MODEL_PATH} não encontrado. Rode antes: python ml/5_treinamento.py")
    model = tf.keras.models.load_model(MODEL_PATH)

    test_data = load_split("teste", 64)
    test_x, test_y = to_arrays(test_data)
    names = [Path(p).name for p in test_data.file_paths]
    is_real = np.array([n.startswith("real_") for n in names])
    is_maquete = np.array([n.startswith("maquete_") for n in names])
    probabilities = model.predict(test_x, verbose=0)
    argmax_pred = probabilities.argmax(axis=1)
    threshold, threshold_table = choose_threshold(model)
    predicted = apply_threshold(probabilities, threshold)
    confidence = probabilities.max(axis=1)
    argmax_report = classification_report(test_y, argmax_pred, target_names=list(CLASSES), output_dict=True,
                                          zero_division=0)

    accuracy = float(accuracy_score(test_y, predicted))
    report = classification_report(test_y, predicted, target_names=list(CLASSES), output_dict=True, zero_division=0)
    report_text = classification_report(test_y, predicted, target_names=list(CLASSES), digits=3, zero_division=0)
    matrix = confusion_matrix(test_y, predicted, labels=range(3))

    print("Treinando modelo de base (regressão logística) para comparação...")
    train_x, train_y = to_arrays(load_split("treino", 64))
    baseline = baseline_accuracy(train_x, train_y, test_x, test_y)

    # Erro mais grave: dizer "seguro" quando era "perigo".
    danger_as_safe = int(matrix[CLASSES.index("perigo"), CLASSES.index("seguro")])

    def subset(mask: np.ndarray) -> dict:
        if not mask.any():
            return {}
        sub_report = classification_report(test_y[mask], predicted[mask], labels=range(3), target_names=list(CLASSES),
                                           output_dict=True, zero_division=0)
        return {
            "amostras": int(mask.sum()),
            "acuracia": round(float(accuracy_score(test_y[mask], predicted[mask])), 4),
            "revocacao_perigo": round(sub_report["perigo"]["recall"], 4),
            "matriz_confusao": confusion_matrix(test_y[mask], predicted[mask], labels=range(3)).tolist(),
        }

    by_origin = {"fotos_reais": subset(is_real), "maquete": subset(is_maquete),
                 "montagens": subset(~is_real & ~is_maquete)}

    results = {
        "amostras_teste": int(len(test_y)),
        "acuracia_teste": round(accuracy, 4),
        "acuracia_modelo_base_regressao_logistica": round(baseline, 4),
        "revocacao_perigo": round(report["perigo"]["recall"], 4),
        "perigo_classificado_como_seguro": danger_as_safe,
        "relatorio_por_classe": report,
        "matriz_confusao": {"linhas_real_colunas_previsto": list(CLASSES), "valores": matrix.tolist()},
        "parametros_modelo": int(model.count_params()),
        "limiar_perigo": threshold,
        "limiar_escolha_validacao": [{"limiar": a, "revocacao_perigo": b, "f1_medio": c, "acuracia": d}
                                     for a, b, c, d in threshold_table],
        "sem_limiar_argmax": {
            "acuracia": round(float(accuracy_score(test_y, argmax_pred)), 4),
            "revocacao_perigo": round(argmax_report["perigo"]["recall"], 4),
            "perigo_classificado_como_seguro": int(confusion_matrix(test_y, argmax_pred, labels=range(3))[PERIGO, 0]),
        },
        "por_origem": by_origin,
    }
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS_DIR / "avaliacao.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    save_confusion_matrix(matrix, ARTIFACTS_DIR / "matriz_confusao.png")
    save_error_examples(test_x, test_y, predicted, confidence, ARTIFACTS_DIR / "exemplos_erros.jpg")

    summary = (
        f"AVALIAÇÃO NO CONJUNTO DE TESTE ({len(test_y)} imagens nunca vistas no treino)\n"
        f"Regra de decisão: 'perigo' quando P(perigo) >= {threshold:.2f} (escolhido na validação: maior F1 médio com revocação de perigo >= 0,75).\n"
        f"Sem limiar (classe mais provável): acurácia {results['sem_limiar_argmax']['acuracia']:.3f}, "
        f"revocação de perigo {results['sem_limiar_argmax']['revocacao_perigo']:.3f}\n\n"
        f"{report_text}\n"
        f"Matriz de confusão (linhas = real, colunas = previsto; ordem {', '.join(CLASSES)}):\n{matrix}\n\n"
        f"Acurácia da CNN:                          {accuracy:.3f}\n"
        f"Acurácia do modelo de base (reg. logística): {baseline:.3f}\n"
        f"Revocação de 'perigo':                    {report['perigo']['recall']:.3f}\n"
        f"Casos de 'perigo' previstos como 'seguro': {danger_as_safe}\n"
    )
    for origin, data in by_origin.items():
        if data:
            summary += (f"\nSó {origin.replace('_', ' ')} ({data['amostras']} imagens): acurácia {data['acuracia']:.3f}, "
                        f"revocação de perigo {data['revocacao_perigo']:.3f}\n"
                        f"  matriz: {data['matriz_confusao']}\n")
    (ARTIFACTS_DIR / "relatorio_avaliacao.txt").write_text(summary, encoding="utf-8")
    print(summary)
    print(f"Arquivos em {ARTIFACTS_DIR}/: avaliacao.json, relatorio_avaliacao.txt, matriz_confusao.png, exemplos_erros.jpg")


if __name__ == "__main__":
    main()
