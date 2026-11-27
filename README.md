# Rota Segura

Detecção assistiva de obstáculos no navegador do celular, com uma CNN treinada do zero.

A câmera olha o caminho à frente e o modelo classifica cada quadro em **sem obstáculo** ou **com obstáculo**. Quando há obstáculo, o celular emite dois bipes e vibra, repetindo o alerta enquanto o obstáculo continuar à frente. Tudo roda no próprio aparelho, sem servidor.

**Demo:** https://rota-segura-sepia.vercel.app

**Equipe:** David Henrique · Victoria Maria · Luiza Teixeira · José Heitor

> Protótipo acadêmico (2026.2 · NSIN-0104 · Inteligência Artificial Aplicada). Não substitui bengala, cão-guia nem orientação profissional.

## Resultados

| Avaliação | Acurácia | Obstáculos detectados | Alarmes falsos |
| --- | --- | --- | --- |
| Validação cruzada, 5 partes (média ± desvio) | **86,6% ± 6,8%** | **88,2% ± 11,9%** | 15,7% ± 19,3% |
| Teste do modelo publicado (80 fotos) | 100% | 100% | 0% |
| Regressão logística (referência) | 85,0% (acurácia balanceada) | — | — |

A validação cruzada é a estimativa mais confiável: com poucas rajadas de fotos, o resultado de uma única divisão depende muito de quais fotos caem no teste.

## Versão 2: modelo do corredor (sem cor, com autocontraste)

O primeiro modelo aprendia a "cara da cena da maquete": muita grama, banco e latas virava obstáculo, e uma imagem lisa virava caminho livre. Quando o fundo mudava, ele errava (nos testes, 38% de alarmes falsos com laterais bagunçadas e acurácia de 42% a 50% com foto mais escura ou mais clara).

A versão 2 (`ml/treino_corredor.py`) muda o que o modelo enxerga e como ele treina:

| Mudança | Para que serve |
| --- | --- |
| Recorte só do corredor central (colunas 25% a 75%) | grama, banco e objetos das laterais não entram na decisão (a mesma região usada na rotulagem) |
| Tons de cinza | o modelo não depende da cor do objeto nem do fundo |
| Autocontraste por imagem (média 0, desvio 1) | foto mais clara ou mais escura dá o mesmo resultado |
| Rotação aleatória de ±10° e deslocamento lateral | o celular na mão nunca fica perfeitamente reto |
| Imagens lisas rotuladas como "sem obstáculo" | cor sólida = nada no caminho |
| Limiar 0,35 | mais sensível a obstáculos, por segurança |
| Média dos 2 últimos quadros no app | um quadro tremido ou borrado não dispara o alarme sozinho |

O app faz o mesmo pré-processamento (`src/App.tsx`) antes de chamar o modelo.

| Teste (127 fotos da maquete nunca vistas no treino) | Modelo antigo (receita original, média de 3 treinos) | Versão 2 |
| --- | --- | --- |
| Acurácia | 83,5% | 93,7% |
| Obstáculos detectados | 77% | 95,9% |
| Alarmes falsos com laterais bagunçadas | 37% | 9,4% (igual sem as laterais) |
| Acurácia com foto mais escura / mais clara | 57% / 48% | 91% / 78% |
| Imagem de cor sólida | 79% livre | 100% livre |

O modelo continua treinado só com fotos da maquete: ele funciona para a maquete e para objetos no corredor à frente, não para qualquer cenário. Para melhorar mais, o caminho é fotografar a maquete com mais fundos, ângulos e iluminações.

## Arquitetura do sistema

```
câmera traseira ─▶ recorte quadrado central ─▶ 160×160×3 ─▶ CNN (TensorFlow.js) ─▶ P(obstáculo) ≥ 0,20 ─▶ bipes + vibração
```

| Componente | Tecnologia |
| --- | --- |
| App | React, TypeScript, Vite, PWA |
| Inferência | TensorFlow.js no navegador (2 quadros por segundo) |
| Treino | TensorFlow / Keras 2 (`tf_keras`), Python 3.12 |
| Rotulagem | OWLv2 (Hugging Face Transformers), usado só como rotulador |
| Hospedagem | Vercel |

## Pipeline de dados e treino

### 1. Dados

| Fonte | Quantidade | Uso |
| --- | --- | --- |
| Fotos oficiais da maquete (Google Drive, em rajada) | 498 | treino, validação e teste |
| Quadros de vídeos da maquete (3 quadros/s) | 392 | reforço do treino |
| Fotos públicas (Places365, Open Images, Traffic_Cone) | 3.279 | testadas na seleção de fontes |
| Montagens (cone ou caixa colado em foto real) | geradas | testadas na seleção de fontes |

As fotos oficiais foram tiradas do ponto de vista do pedestre, olhando ao longo da faixa tátil amarela, com o caminho livre e com obstáculos (cones, latas, potes, bonecos, barreiras, carrinhos) a várias distâncias.

### 2. Rotulagem automática (professor-aluno)

Um detector pré-treinado, o **OWLv2** (`google/owlv2-base-patch16-ensemble`), recebe consultas de texto (`"a toy traffic cone"`, `"a road barrier"`, `"a can"`, `"a jar"`, `"a toy figure"`...) e devolve caixas com confiança ≥ 0,28. A detecção é feita no mesmo recorte quadrado que o app analisa. Uma regra transforma as caixas em rótulo:

| Rótulo | Regra |
| --- | --- |
| `com_obstaculo` | algum objeto no corredor central (25% a 75% da largura) com altura ≥ 15% da imagem |
| `sem_obstaculo` | nada no corredor, ou só objetos com altura < 8% |
| descartada | só objetos entre 8% e 15% no corredor (longe, ambíguo) |

Resultado na base oficial: 264 com obstáculo, 161 sem obstáculo e 73 descartadas.

O detector confundia a faixa tátil com uma barreira, então caixas grandes com mais de 35% de pixels amarelo-vivo são ignoradas. **O OWLv2 não vai para o app:** ele só cria os rótulos, e a CNN do projeto aprende com eles (destilação).

### 3. Divisão sem vazamento

Fotos vizinhas de uma rajada são quase idênticas. Se uma ficasse no treino e a vizinha no teste, o teste mediria memorização. Por isso:

- cada rajada é cortada em **trechos de 10 fotos seguidas**, e cada trecho vai inteiro para uma única parte;
- os quadros de vídeo são divididos por trechos de 5 segundos;
- entre várias divisões sorteadas (semente 42), fica a que deixa validação e teste com a proporção de classes mais próxima da base completa.

| Parte | Fotos oficiais | Com obstáculo | Sem obstáculo | Uso |
| --- | --- | --- | --- | --- |
| Treino | 290 | 179 | 111 | ajustar os pesos |
| Validação | 55 | 35 | 20 | época, limiar e fontes de dados |
| Teste | 80 | 50 | 30 | resultado final, usado uma vez |

### 4. Seleção das fontes de dados

`5b_escolha_das_fontes.py` treina a CNN com e sem os dados públicos e compara a acurácia balanceada nas fotos oficiais da validação. Só fotos da maquete: 82,5%. Com fotos públicas: 81,1%. Os dados públicos ficaram de fora do modelo final.

### 5. Modelo

CNN treinada do zero, com pesos aleatórios (198.706 parâmetros, cerca de 800 KB):

| Camada | Saída | Detalhe |
| --- | --- | --- |
| Entrada | 160×160×3 | valores entre 0 e 1 |
| Blocos 1–4 | 80×80×16 → 10×10×96 | Conv 3×3 → BatchNorm → ReLU → MaxPool 2×2 |
| Conv final | 10×10×128 | Conv 3×3 → BatchNorm → ReLU |
| Pooling | 128 | Global Average Pooling + Dropout 0,3 |
| Densa | 64 | ReLU |
| Saída | 2 | softmax |

Uma CNN foi escolhida porque aprende bordas e formas sozinha, reconhece o objeto em qualquer posição com poucos parâmetros e preserva a informação espacial (centro × lateral da faixa). O pooling global substitui camadas densas grandes e reduz o sobreajuste.

### 6. Treinamento

| Hiperparâmetro | Valor |
| --- | --- |
| Otimizador | Adam, taxa 0,001, reduzida pela metade após 4 épocas sem melhora |
| Perda | entropia cruzada com peso por classe |
| Lote | 32 |
| Épocas | até 40, parada antecipada com paciência 10 (melhor época restaurada) |
| Aumento de dados | espelhamento horizontal, brilho, contraste e saturação |
| BatchNorm | momento 0,9 (com 0,99 a rede não convergia com poucas fotos) |

Zoom e deslocamento vertical não são usados, porque mudariam o tamanho e a posição do obstáculo, que definem a classe.

### 7. Avaliação

- **Validação cruzada em 5 partes** (`6b_validacao_cruzada.py`): partes formadas por grupos de rajada, com proporção de classes mantida. A CNN é treinada 5 vezes do zero.

| Parte | Fotos de teste | Acurácia treino | Acurácia teste | Obstáculos detectados |
| --- | --- | --- | --- | --- |
| 1 | 83 | 96,9% | 84,3% | 75,5% |
| 2 | 88 | 98,9% | 84,1% | 100% |
| 3 | 84 | 98,3% | 100% | 100% |
| 4 | 83 | 92,2% | 83,1% | 72,5% |
| 5 | 87 | 91,0% | 81,6% | 92,7% |

- **Limiar de decisão:** o app avisa obstáculo quando P(obstáculo) ≥ 0,20. O limiar foi escolhido na validação: entre os que detectam pelo menos 85% dos obstáculos, o de maior acurácia balanceada. Deixar passar um obstáculo é o erro mais grave, então alguns alarmes falsos são aceitos.
- **Sobreajuste:** a acurácia de treino (95,5%) fica acima da de teste (86,6%), o que é esperado numa base pequena.

Métricas completas em [ml/artefatos/relatorio_avaliacao.txt](ml/artefatos/relatorio_avaliacao.txt), curvas em [ml/artefatos/curvas_treino.png](ml/artefatos/curvas_treino.png) e matriz de confusão em [ml/artefatos/matriz_confusao.png](ml/artefatos/matriz_confusao.png).

### 8. App

- O modelo Keras é convertido para TensorFlow.js por `7_exportar_para_app.py` e fica em `public/model/`.
- O app abre a câmera traseira (o navegador exige HTTPS), recorta o quadrado central de cada quadro, reduz para 160×160 e classifica 2 vezes por segundo.
- Com obstáculo, toca dois bipes de 1100 Hz (Web Audio API) e vibra (`navigator.vibrate`), repetindo a cada 1,5 s.
- Celulares bloqueiam áudio sem interação, então o som é liberado no toque em **Abrir câmera**. No iPhone, a chave de silencioso corta o som e o Safari não vibra.

## Executar o app

```bash
npm install
npm run dev
```

No celular, na mesma rede Wi-Fi, abra o endereço `https://<ip>:5173` mostrado no terminal e aceite o aviso de certificado.

## Reproduzir o treino

Use Python 3.11 ou 3.12 (o TensorFlow 2.19 não funciona no 3.13).

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

## Limitações

- Base pequena, de uma única maquete e iluminação.
- Rótulos gerados por detector automático, com algum ruído.
- Objetos muito longe, no fim da faixa, às vezes não são detectados.
- Uma imagem isolada não mede distância nem velocidade de aproximação.

## Fontes de dados

- Places365 (MIT), uso acadêmico
- Open Images V7 (Google), CC BY 2.0
- Junhan0518/Traffic_Cone (Hugging Face), Apache-2.0
- OWLv2 (`google/owlv2-base-patch16-ensemble`), usado só como rotulador
