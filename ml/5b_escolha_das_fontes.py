"""ETAPA 5b - QUAIS FONTES PÚBLICAS AJUDAM? (seleção de dados)"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

OPTIONS = ("nenhuma", "reais", "montagens", "todas")
PY = sys.executable


def run(script: str, env: dict, *extra: str) -> None:
    subprocess.run([PY, "-u", f"ml/{script}", *extra], env=env, check=True,
                   stdout=subprocess.DEVNULL if "4_" in script else None)


def evaluate_on_validation(base: Path, artifacts: Path, env: dict) -> dict:
    code = f"""
import json, numpy as np
from sklearn.metrics import balanced_accuracy_score, recall_score
from comum import load_split, tf, POSITIVE, MODEL_PATH
from pathlib import Path
ds = load_split("validacao", 64)
xs, ys = [], []
for x, y in ds:
    xs.append(x.numpy()); ys.append(y.numpy().argmax(1))
x, y = np.concatenate(xs), np.concatenate(ys)
official = np.array([Path(p).name.startswith("oficial_") for p in ds.file_paths])
m = official if official.any() else np.ones_like(y, dtype=bool)
prob = tf.keras.models.load_model(MODEL_PATH).predict(x[m], verbose=0)[:, POSITIVE]
best = max(((balanced_accuracy_score(y[m], (prob >= t).astype(int) if POSITIVE == 1 else (prob < t).astype(int)), t)
            for t in np.arange(0.1, 0.91, 0.05)))
pred = (prob >= 0.5).astype(int) if POSITIVE == 1 else (prob < 0.5).astype(int)
print(json.dumps({{"amostras_validacao_oficial": int(m.sum()),
                   "acuracia_balanceada_0_5": round(float(balanced_accuracy_score(y[m], pred)), 4),
                   "acuracia_balanceada_melhor_limiar": round(float(best[0]), 4),
                   "revocacao_obstaculo_0_5": round(float(recall_score(y[m], pred, pos_label=POSITIVE)), 4)}}))
"""
    out = subprocess.run([PY, "-c", code], env=env, check=True, capture_output=True, text=True, cwd="ml").stdout
    return json.loads(out.strip().splitlines()[-1])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--epocas", type=int, default=40)
    parser.add_argument("--opcoes", nargs="+", default=list(OPTIONS), choices=OPTIONS)
    args = parser.parse_args()

    results = {}
    for option in args.opcoes:
        base = Path(f"data/bases/{option}")
        artifacts = Path(f"ml/artefatos/experimentos/{option}")
        env = dict(os.environ, ROTA_BASE=str(base.resolve()), ROTA_ARTEFATOS=str(artifacts.resolve()),
                   ROTA_PUBLICAS=option)
        print(f"\n=== Fontes públicas: {option} ===", flush=True)
        run("4_montar_base_rotulada.py", env, "--publicas", option, "--saida", str(base))
        run("5_treinamento.py", env, "--epocas", str(args.epocas))
        results[option] = evaluate_on_validation(base, artifacts, env)
        print(f"Validação (fotos oficiais): {results[option]}", flush=True)

    chosen = max(results, key=lambda o: (results[o]["acuracia_balanceada_melhor_limiar"],
                                         results[o]["acuracia_balanceada_0_5"]))
    print(f"\nEscolhida: {chosen}")

    if Path("data/base").exists():
        shutil.rmtree("data/base")
    shutil.copytree(f"data/bases/{chosen}", "data/base")
    for name in ("modelo.keras", "historico_treino.csv", "curvas_treino.png"):
        shutil.copy2(f"ml/artefatos/experimentos/{chosen}/{name}", f"ml/artefatos/{name}")
    Path("ml/artefatos/escolha_fontes.json").write_text(
        json.dumps({"criterio": "maior acurácia balanceada nas fotos oficiais da VALIDAÇÃO", "escolhida": chosen,
                    "resultados": results}, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Base e modelo escolhidos copiados para data/base e ml/artefatos/. Rode a etapa 6 para avaliar no teste.")


if __name__ == "__main__":
    main()
