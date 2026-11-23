#!/usr/bin/env bash
# Executa o projeto inteiro, na ordem pedida pelo professor:
# coleta -> rotulagem -> divisão treino/teste -> base rotulada -> treino -> avaliação -> exportação -> relatório.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=".venv/bin/python"
[ -x "$PY" ] || { echo "Crie o ambiente antes (veja o README): python3.12 -m venv .venv"; exit 1; }

echo "== 1. Coleta de dados =="                 && $PY ml/1_coleta_dados.py
if [ -f data/rotulos_automaticos.csv ] && [ "${REROTULAR:-0}" != "1" ]; then
  echo "== 2. Rotulagem automática: já feita (REROTULAR=1 para refazer) =="
else
  echo "== 2. Rotulagem automática =="
  until $PY ml/2_rotulagem_automatica.py; do [ $? -eq 3 ] || exit 1; done
fi
if find data/maquete/sem_rotulo -maxdepth 1 -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" -o -iname "*.heic" -o -iname "*.webp" \) 2>/dev/null | grep -q .; then
  echo "== 2b. Rotulagem automática das fotos da maquete =="
  rm -f data/rotulos_maquete.parcial.csv
  until $PY ml/2_rotulagem_automatica.py --fotos data/maquete/sem_rotulo --saida data/rotulos_maquete.csv; do
    [ $? -eq 3 ] || exit 1
  done
fi
echo "== 3. Divisão treino/validação/teste =="  && $PY ml/3_divisao_treino_teste.py
echo "== 4. Montagem da base rotulada =="       && $PY ml/4_montar_base_rotulada.py
echo "== 5. Treinamento =="                     && $PY ml/5_treinamento.py
echo "== 6. Avaliação =="                       && $PY ml/6_avaliacao.py
echo "== 7. Exportação para o app =="           && $PY ml/7_exportar_para_app.py
echo "== 8. Relatório em PDF =="              && $PY ml/8_relatorio_pdf.py
echo "Pronto! Rode o app com: npm run dev"
