# Rota Segura

Detecção assistiva de obstáculos no navegador do celular, com uma CNN treinada do zero.

A câmera olha o caminho à frente e o modelo classifica cada quadro em **sem obstáculo** ou **com obstáculo**. Quando há obstáculo, o celular emite dois bipes e vibra, repetindo o alerta enquanto o obstáculo continuar à frente.

**Demo:** https://rota-segura-sepia.vercel.app

> Protótipo acadêmico (2026.2 · NSIN-0104 · Inteligência Artificial Aplicada). Não substitui bengala, cão-guia nem orientação profissional.

## Resultados

Avaliação na base oficial (fotos da maquete tiradas pelo grupo), com divisão por rajada para evitar vazamento entre treino e teste:

| Conjunto | Acurácia | Obstáculos detectados | Alarmes falsos |
| --- | --- | --- | --- |
| Teste (80 fotos) | 100% | 100% | 0% |
| Validação cruzada, 5 partes (média ± desvio) | 86,6% ± 6,8% | 88,2% ± 11,9% | 15,7% ± 19,3% |
| Modelo de referência (regressão logística) | 85,0% (acurácia balanceada) | — | — |

A validação cruzada é a estimativa mais confiável, porque o teste único tem poucas rajadas. Detalhes em [ml/artefatos/relatorio_avaliacao.txt](ml/artefatos/relatorio_avaliacao.txt).

## Como funciona

1. **Dados:** a base oficial tem cerca de 500 fotos da maquete, tiradas do ponto de vista do pedestre, mais quadros extraídos de vídeos da maquete.
2. **Rotulagem:** um detector pré-treinado (OWLv2) encontra os objetos de cada foto. Uma regra de tamanho e posição gera o rótulo: a foto é *com obstáculo* quando algum objeto no corredor central (25% a 75% da largura) ocupa 15% ou mais da altura. O OWLv2 só rotula os dados e não vai para o app.
3. **Treino:** uma CNN pequena (cerca de 200 mil parâmetros, entrada 160×160) é treinada do zero com TensorFlow/Keras.
4. **App:** o modelo é convertido para TensorFlow.js e roda inteiro no celular, sem servidor.

## Executar o app

```bash
npm install
npm run dev
```

No celular, na mesma rede Wi-Fi, abra o endereço `https://<ip>:5173` mostrado no terminal e aceite o aviso de certificado. O navegador só libera a câmera em HTTPS.

O som só começa depois do toque em **Abrir câmera**. No iPhone, a chave de silencioso também corta o som, e o Safari não vibra.

## Treinar o modelo

Use Python 3.11 ou 3.12.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r ml/requirements.txt -r ml/requirements-rotulagem.txt
bash ml/executar_tudo.sh
```

| Etapa | Script |
| --- | --- |
| Coleta de fotos públicas | `ml/1_coleta_dados.py` |
| Quadros dos vídeos da maquete | `ml/1b_quadros_dos_videos.py` |
| Base oficial (Google Drive) | `ml/1c_fotos_oficiais_drive.py` |
| Rotulagem automática | `ml/2_rotulagem_automatica.py` |
| Divisão treino/validação/teste | `ml/3_divisao_treino_teste.py` |
| Montagem da base | `ml/4_montar_base_rotulada.py` |
| Treinamento | `ml/5_treinamento.py` |
| Escolha das fontes de dados | `ml/5b_escolha_das_fontes.py` |
| Avaliação | `ml/6_avaliacao.py` |
| Validação cruzada | `ml/6b_validacao_cruzada.py` |
| Exportação para TensorFlow.js | `ml/7_exportar_para_app.py` |

As fotos de bases públicas não ficam no repositório, porque as licenças não permitem redistribuir. A etapa 1 baixa tudo de novo.

## Estrutura

```
src/              app React + TypeScript (câmera, inferência, alertas)
public/model/     modelo em TensorFlow.js
ml/               pipeline de dados, treino e avaliação
ml/artefatos/     modelo Keras, métricas e gráficos
data/             rótulos, divisão e fotos da maquete
```

## Stack

React · TypeScript · Vite · TensorFlow.js · TensorFlow/Keras · OWLv2 (Hugging Face Transformers)

## Fontes de dados

- Places365 (MIT), uso acadêmico
- Open Images V7 (Google), CC BY 2.0
- Junhan0518/Traffic_Cone (Hugging Face), Apache-2.0
- OWLv2 (`google/owlv2-base-patch16-ensemble`), usado só como rotulador
