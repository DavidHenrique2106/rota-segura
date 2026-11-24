# Rota Segura — detecção assistiva de obstáculos com IA

Projeto da disciplina **2026.2 · NSIN-0104 · Inteligência Artificial Aplicada**.

A câmera do celular olha o caminho à frente. Uma rede neural treinada **do zero** pelo grupo classifica a cena em **seguro**, **atenção** ou **perigo**, e o celular avisa com **som e vibração**.

- **App publicado:** https://rota-segura-sepia.vercel.app (abra no celular e toque em *Abrir câmera*)
- **Relatório completo para o professor:** [Relatorio_Rota_Segura.pdf](Relatorio_Rota_Segura.pdf)
- **Resultados da última avaliação:** [ml/artefatos/relatorio_avaliacao.txt](ml/artefatos/relatorio_avaliacao.txt)

> **Segurança:** é uma demonstração acadêmica. Não substitui bengala, cão-guia nem orientação profissional.

---

## As etapas (uma por script, em `ml/`)

| Etapa | Script | O que faz |
| --- | --- | --- |
| 1. Coleta | `1_coleta_dados.py` | Baixa fotos reais de bases públicas (Places365, Open Images, Hugging Face) e recorta cones e caixas |
| 1b. Vídeos | `1b_quadros_dos_videos.py` | Extrai 3 quadros/s dos vídeos da maquete (`data/maquete/videos/`), ignora repetidos e borrados |
| 2. Rotulagem | `2_rotulagem_automatica.py` | Um detector pré-treinado (OWLv2) marca os obstáculos de cada foto real; uma regra de tamanho e posição gera o rótulo |
| 3. Divisão | `3_divisao_treino_teste.py` | Separa as fotos de origem em 70% treino, 15% validação e 15% teste, sem vazamento |
| 4. Base | `4_montar_base_rotulada.py` | Junta as fotos reais rotuladas com montagens de cones e caixas, com as classes equilibradas |
| 5. Treino | `5_treinamento.py` | Treina a CNN **do zero** (TensorFlow/Keras) |
| 6. Avaliação | `6_avaliacao.py` | Acurácia, precisão, revocação, matriz de confusão, comparação com regressão logística e resultado por origem da imagem |
| 7. Exportação | `7_exportar_para_app.py` | Converte o modelo para TensorFlow.js, que roda no celular |
| 8. Relatório | `8_relatorio_pdf.py` | Gera o PDF com tudo isso |

**Sobre o modelo pronto (OWLv2):** ele só **rotula os dados** na etapa 2, como um "professor". Não vai para o app. Quem classifica no celular é a nossa CNN, treinada do zero com esses rótulos. Essa técnica se chama professor-aluno, ou destilação.

### Regra de rótulo

O obstáculo precisa estar no corredor central da imagem, entre 25% e 75% da largura.

| Classe | Regra |
| --- | --- |
| `perigo` | obstáculo grande (48% ou mais da altura da imagem) com a base embaixo, ou objeto largo encostado na borda de baixo |
| `atencao` | obstáculo médio (22–36% da altura) |
| `seguro` | caminho livre, ou só obstáculos pequenos (menos de 13%) |

Fotos com obstáculos entre essas faixas são ambíguas e ficam fora da base.

---

## Executar

### Só o app (o modelo já vem treinado em `public/model/`)

```bash
npm install
npm run dev
```

No celular, na mesma rede Wi-Fi, abra o endereço `https://192.168...:5173` que aparece no terminal e aceite o aviso de certificado. Ou use direto o link da Vercel.

### Pipeline completo

Use Python 3.11 ou 3.12. O TensorFlow 2.19 não funciona no Python 3.13 ou mais novo.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r ml/requirements.txt -r ml/requirements-rotulagem.txt
bash ml/executar_tudo.sh
```

- A coleta baixa uns 900 MB.
- A rotulagem leva cerca de 45 minutos num Mac M4. Ela roda em blocos de 600 fotos para não acumular memória da GPU e retoma de onde parou se for interrompida.
- O treino leva cerca de 25 minutos.

As fotos baixadas da internet **não estão no repositório**, porque as licenças não permitem redistribuir; o script 1 baixa tudo de novo. Os rótulos gerados estão em `data/rotulos_automaticos.csv` e a divisão em `data/divisao.json`.

---

## Fotos e vídeos da maquete

**Vídeos (recomendado):** grave andando devagar pela calçada da maquete, do ponto de vista do boneco, em direção a cada obstáculo. Coloque os vídeos em `data/maquete/videos/` e rode `bash ml/executar_tudo.sh`. A etapa 1b extrai os quadros e a etapa 2 rotula cada um. A divisão treino/validação/teste é feita por **trechos de 5 segundos**, para que quadros quase iguais não fiquem em partes diferentes. Para a maquete, o rotulador procura só os objetos dela (cone, barreira, lata, pote, boneco, carrinho) e ignora a faixa amarela tátil, que ele confundia com uma barreira.

**Fotos:**

1. Tire as fotos **do ponto de vista do boneco**: celular em pé na calçada, câmera olhando para frente. Fotografe cada obstáculo bem perto, a uma distância média e longe, e também o caminho livre. Umas 150 fotos no total.
2. Coloque-as em `data/maquete/sem_rotulo/`. **Não precisa rotular**: a etapa 2 rotula sozinha. JPG, PNG e HEIC funcionam.
3. Rode `bash ml/executar_tudo.sh`. A avaliação mostra à parte o acerto nas fotos da maquete.
4. Republique o app: `npm run build`, depois `cd dist` e `vercel deploy --prod`.

As fotos em `data/maquete/visao_geral/` mostram a maquete de cima e servem só de registro.

---

## Estrutura

```
ml/                       pipeline (etapas 1–8), comum.py, requirements
ml/artefatos/             modelo treinado, métricas, gráficos, amostras da rotulagem
data/divisao.json         divisão treino/validação/teste das fotos de origem
data/rotulos_*.csv        rótulos gerados pela etapa 2
data/maquete/             fotos da maquete
public/model/             modelo exportado para o app (TensorFlow.js)
src/App.tsx               app: câmera, classificação, som e vibração
Relatorio_Rota_Segura.pdf relatório para o professor
```

## Fontes de dados e modelos

- Places365 (MIT), uso acadêmico
- Open Images V7 (Google), imagens CC BY 2.0
- Junhan0518/Traffic_Cone (Hugging Face), Apache-2.0
- OWLv2 (`google/owlv2-base-patch16-ensemble`), usado só como rotulador
