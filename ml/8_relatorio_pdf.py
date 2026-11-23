"""ETAPA 8 - RELATÓRIO EM PDF PARA O PROFESSOR

Gera Relatorio_Rota_Segura.pdf (na raiz do projeto) com a descrição do
projeto, das etapas e dos resultados. Os números e figuras são lidos dos
arquivos produzidos pelas etapas 1 a 7, então o relatório sempre reflete o
último treino.

Uso:
    python ml/8_relatorio_pdf.py
"""

from __future__ import annotations

import csv
import json
import random
from collections import Counter
from datetime import date
from pathlib import Path

from PIL import Image
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (Image as RLImage, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer,
                                Table, TableStyle)

ROOT = Path(".")
ART = ROOT / "ml/artefatos"
OUT = ROOT / "Relatorio_Rota_Segura.pdf"
APP_URL = "https://rota-segura-sepia.vercel.app"
CLASSES = ("seguro", "atencao", "perigo")
PT = {"seguro": "seguro", "atencao": "atenção", "perigo": "perigo"}

FOREST = colors.HexColor("#123d35")
LIME = colors.HexColor("#d8ed61")
LINE = colors.HexColor("#c9cfc4")
SOFT = colors.HexColor("#eef1e8")

styles = getSampleStyleSheet()
S = {
    "title": ParagraphStyle("t", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=26, leading=31,
                            textColor=FOREST, alignment=0, spaceAfter=6),
    "subtitle": ParagraphStyle("st", parent=styles["Normal"], fontSize=13, leading=18, textColor=colors.HexColor("#4b5a52")),
    "h1": ParagraphStyle("h1", parent=styles["Heading1"], fontName="Helvetica-Bold", fontSize=16, leading=20,
                         textColor=FOREST, spaceBefore=14, spaceAfter=6),
    "h2": ParagraphStyle("h2", parent=styles["Heading2"], fontName="Helvetica-Bold", fontSize=12.5, leading=16,
                         textColor=FOREST, spaceBefore=10, spaceAfter=4),
    "body": ParagraphStyle("b", parent=styles["Normal"], fontSize=10.3, leading=14.5, spaceAfter=6),
    "small": ParagraphStyle("s", parent=styles["Normal"], fontSize=8.6, leading=11.5, textColor=colors.HexColor("#4b5a52")),
    "caption": ParagraphStyle("c", parent=styles["Normal"], fontSize=8.6, leading=11.5, alignment=TA_CENTER,
                              textColor=colors.HexColor("#4b5a52"), spaceAfter=8),
    "cell": ParagraphStyle("cell", parent=styles["Normal"], fontSize=9, leading=11.5),
    "cellb": ParagraphStyle("cellb", parent=styles["Normal"], fontName="Helvetica-Bold", fontSize=9, leading=11.5),
    "code": ParagraphStyle("code", parent=styles["Code"], fontSize=8.6, leading=11.5, backColor=SOFT,
                           borderPadding=6, leftIndent=4, rightIndent=4, spaceBefore=4, spaceAfter=8),
}


def p(text: str, style: str = "body") -> Paragraph:
    return Paragraph(text, S[style])


def pct(value: float | None) -> str:
    return "—" if value is None else f"{value * 100:.1f}%".replace(".", ",")


def num(value: int) -> str:
    return f"{value:,}".replace(",", ".")


def table(rows: list[list], widths: list[float], header: bool = True, zebra: bool = True) -> Table:
    data = [[c if isinstance(c, Paragraph) else Paragraph(str(c), S["cellb" if header and i == 0 else "cell"])
             for c in row] for i, row in enumerate(rows)]
    t = Table(data, colWidths=[w * cm for w in widths], repeatRows=1 if header else 0)
    style = [("GRID", (0, 0), (-1, -1), 0.4, LINE), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
             ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]
    if header:
        style += [("BACKGROUND", (0, 0), (-1, 0), FOREST), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white)]
        data[0] = [Paragraph(c.text if isinstance(c, Paragraph) else str(c),
                             ParagraphStyle("hdr", parent=S["cellb"], textColor=colors.white)) for c in data[0]]
        t = Table(data, colWidths=[w * cm for w in widths], repeatRows=1)
    if zebra:
        for r in range(1 if header else 0, len(data)):
            if r % 2 == 0:
                style.append(("BACKGROUND", (0, r), (-1, r), SOFT))
    t.setStyle(TableStyle(style))
    return t


def figure(path: Path, width_cm: float, caption: str) -> list:
    if not path.exists():
        return [p(f"<i>(figura {path.name} não encontrada — rode as etapas anteriores)</i>", "small")]
    with Image.open(path) as im:
        w, h = im.size
    img = RLImage(str(path), width=width_cm * cm, height=width_cm * cm * h / w)
    return [KeepTogether([img, p(caption, "caption")])]


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def base_sample_sheet(output: Path) -> None:
    """Folha com exemplos da base de treino: linhas = classes, colunas = real/montagem."""
    base = ROOT / "data/base/treino"
    if not base.exists():
        return
    cell, cols = 150, 8
    sheet = Image.new("RGB", (cols * cell, 3 * cell), "white")
    rng = random.Random(7)
    for r, label in enumerate(CLASSES):
        real = sorted((base / label).glob("real_*.jpg"))
        comp = sorted((base / label).glob("montagem_*.jpg"))
        chosen = rng.sample(real, min(4, len(real))) + rng.sample(comp, min(cols - min(4, len(real)), len(comp)))
        for c, f in enumerate(chosen[:cols]):
            sheet.paste(Image.open(f).convert("RGB").resize((cell - 4, cell - 4)), (c * cell + 2, r * cell + 2))
    sheet.save(output, quality=88)


def on_page(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#6b776e"))
    canvas.drawString(2 * cm, 1.2 * cm, "Rota Segura · IA Aplicada · 2026.2 · NSIN-0104")
    canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"página {doc.page}")
    canvas.restoreState()


def main() -> None:
    evaluation = json.loads((ART / "avaliacao.json").read_text(encoding="utf-8"))
    division = json.loads((ROOT / "data/divisao.json").read_text(encoding="utf-8"))
    labels = read_csv(ROOT / "data/rotulos_automaticos.csv")
    labels_maq = read_csv(ROOT / "data/rotulos_maquete.csv")
    meta = read_csv(ROOT / "data/base/metadados.csv")
    history = read_csv(ART / "historico_treino.csv")
    report = evaluation["relatorio_por_classe"]
    matrix = evaluation["matriz_confusao"]["valores"]
    origin = evaluation.get("por_origem", {})
    real_eval, maq_eval, comp_eval = origin.get("fotos_reais") or {}, origin.get("maquete") or {}, origin.get("montagens") or {}

    fontes = ROOT / "data/fontes"
    n_fundos = len(list((fontes / "fundos").glob("*.jpg")))
    n_cones = len(list((fontes / "recortes/cone").glob("*.png")))
    n_caixas = len(list((fontes / "recortes/caixa").glob("*.png")))
    reais = sorted(p_.name for p_ in (fontes / "fotos_reais").glob("*.jpg"))
    n_places = sum(n.startswith("places_") for n in reais)
    n_oi = sum(n.startswith("oi_") for n in reais)
    n_hf = sum(n.startswith("hf_cone") for n in reais)
    label_counts = Counter(r["rotulo"] for r in labels)
    base_counts = Counter((r["particao"], r["classe"], r["tipo"]) for r in meta)
    best = min(range(len(history)), key=lambda i: float(history[i]["val_loss"])) if history else None
    params = evaluation.get("parametros_modelo", 0)

    base_sample_sheet(ART / "exemplos_base.jpg")

    doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm,
                            topMargin=1.8 * cm, bottomMargin=2 * cm,
                            title="Rota Segura — Relatório do Projeto", author="Equipe Rota Segura")
    st: list = []

    # ------------------------------------------------------------------ capa
    st += [Spacer(1, 3 * cm),
           p("INTELIGÊNCIA ARTIFICIAL APLICADA · 2026.2 · NSIN-0104", "small"), Spacer(1, 8),
           p("Rota Segura", "title"),
           p("Detecção assistiva de obstáculos por câmera de celular para pessoas com deficiência visual", "subtitle"),
           Spacer(1, 1.2 * cm)]
    summary_rows = [
        ["Pergunta", "Resposta"],
        ["O que o sistema faz", "A câmera do celular olha o caminho à frente; uma rede neural classifica a cena em "
                                "<b>seguro</b>, <b>atenção</b> ou <b>perigo</b> e o celular avisa com som e vibração."],
        ["Modelo", f"Rede neural convolucional (CNN) treinada <b>do zero</b> em TensorFlow/Keras "
                   f"({num(params)} parâmetros). Nenhum modelo pronto roda no app."],
        ["Dados", f"{num(len(reais))} fotos reais + recortes reais de {n_cones} cones e {n_caixas} caixas "
                  f"sobre {num(n_fundos)} fotos de caminhos; {num(len(meta))} imagens na base final."],
        ["Resultado no teste", f"Acurácia <b>{pct(evaluation['acuracia_teste'])}</b>; "
                               f"{pct(evaluation['revocacao_perigo'])} das situações de perigo detectadas"
                               + (f"; {pct(real_eval.get('acuracia'))} só nas fotos reais" if real_eval else "") + "."],
        ["App", f"Publicado em <link href='{APP_URL}' color='blue'>{APP_URL}</link> (abre no celular, usa a câmera)."],
    ]
    st += [table(summary_rows, [4, 13]), Spacer(1, 1 * cm),
           p(f"Relatório gerado em {date.today().strftime('%d/%m/%Y')} a partir dos arquivos do último treino.", "small"),
           PageBreak()]

    # --------------------------------------------------------------- 1. ideia
    st += [p("1. Ideia e objetivo", "h1"),
           p("A proposta inicial era um sistema com Arduino e sensor ultrassônico HC-SR04 que classificasse a "
             "proximidade de obstáculos. Evoluímos a ideia para usar a <b>câmera do celular</b>, que todo usuário já "
             "tem: o celular aponta para o caminho, a IA analisa a imagem e avisa por som e vibração. A maquete "
             "(uma rua com calçada, faixa tátil amarela, cones, barreira e outros objetos) serve para demonstrar."),
           p("Em vez de uma regra fixa, treinamos um modelo de <b>aprendizado de máquina</b> para classificar a cena em "
             "três níveis de risco:"),
           table([["Classe", "Significado"],
                  ["seguro", "Caminho livre, ou obstáculo pequeno e distante."],
                  ["atenção", "Há um obstáculo no caminho a alguns passos."],
                  ["perigo", "Obstáculo grande, logo à frente, no centro do caminho."]], [3, 14]),
           Spacer(1, 6),
           p("Seguindo a orientação da disciplina, o projeto passa pelas etapas de <b>coleta de dados, divisão em "
             "treino e teste, treinamento e avaliação</b>. Cada etapa é um script separado:"),
           table([["Etapa", "Script", "O que faz"],
                  ["1. Coleta", "ml/1_coleta_dados.py", "Baixa fotos reais de bases públicas e recorta cones e caixas."],
                  ["2. Rotulagem", "ml/2_rotulagem_automatica.py", "Rotula as fotos reais com um detector de objetos."],
                  ["3. Divisão", "ml/3_divisao_treino_teste.py", "Separa 70% treino, 15% validação, 15% teste."],
                  ["4. Base", "ml/4_montar_base_rotulada.py", "Monta a base final equilibrada (fotos reais + montagens)."],
                  ["5. Treino", "ml/5_treinamento.py", "Treina a CNN do zero."],
                  ["6. Avaliação", "ml/6_avaliacao.py", "Métricas, matriz de confusão e comparação com modelo simples."],
                  ["7. Exportação", "ml/7_exportar_para_app.py", "Converte o modelo para TensorFlow.js (celular)."],
                  ["8. Relatório", "ml/8_relatorio_pdf.py", "Gera este relatório a partir dos resultados."]],
                 [2.6, 5.6, 8.8])]

    # -------------------------------------------------------------- 2. coleta
    st += [p("2. Coleta de dados (etapa 1)", "h1"),
           p("Usamos somente bases públicas, baixadas automaticamente pelo script. Todas as fotos são reais."),
           table([["Fonte", "O que usamos", "Quantidade"],
                  ["Places365 (MIT)", "Cenas por onde se caminha (corredor, viela, calçadão...) usadas como fundo",
                   f"{num(n_fundos)} fotos"],
                  ["Open Images V7 (Google)", "Caixas recortadas com as máscaras de segmentação feitas por humanos",
                   f"{n_caixas} caixas"],
                  ["Traffic_Cone (Hugging Face)", "Fotos de cone em fundo branco, recortadas automaticamente",
                   f"{n_cones} cones"],
                  ["Places365 (outras cenas)", "Fotos reais com móveis, pessoas e objetos (rua, sala, estação...)",
                   f"{num(n_places)} fotos"],
                  ["Open Images V7", "Fotos reais com cadeiras, bancos, bicicletas, lixeiras, pessoas, cones...",
                   f"{num(n_oi)} fotos"],
                  ["Traffic_Cone", "Fotos de cones em cenas reais", f"{n_hf} fotos"]],
                 [4.4, 9.6, 3]),
           Spacer(1, 4),
           p("Os recortes de cones e caixas passaram por <b>revisão manual</b>: removemos os que tinham marca d'água "
             "ou máscara incompleta. Todas as fotos reais foram cortadas no <b>quadrado central</b>, que é exatamente "
             "a parte da câmera que o app analisa.")]

    # ----------------------------------------------------------- 3. rotulagem
    st += [p("3. Rotulagem automática (etapa 2)", "h1"),
           p("Rotular milhares de fotos à mão seria inviável. Usamos um modelo pré-treinado de detecção de objetos, o "
             "<b>OWLv2</b> (Google), apenas como <b>rotulador</b>: ele encontra obstáculos descritos em texto "
             "(\"traffic cone\", \"cardboard box\", \"person\", \"chair\", \"bench\", \"bicycle\", \"trash can\", "
             "\"pole\"...) e devolve a caixa de cada um. Depois aplicamos uma regra geométrica:"),
           table([["Classe", "Regra (obstáculo no corredor central, 25% a 75% da largura)"],
                  ["perigo", "altura do obstáculo de 48% da imagem ou mais, com a base na parte de baixo, ou objeto largo "
                             "encostado na borda inferior"],
                  ["atenção", "altura entre 22% e 36% da imagem, base entre 60% e 92%"],
                  ["seguro", "nenhum obstáculo no corredor, ou só obstáculos pequenos (menos de 13% da altura)"],
                  ["descartada", "obstáculo entre as faixas (ambíguo): não entra na base"]], [3, 14]),
           Spacer(1, 6),
           p("<b>Importante:</b> o OWLv2 não vai para o app e não faz a classificação final. Ele só cria os rótulos; "
             "quem aprende e roda no celular é a nossa CNN, treinada do zero. É a técnica de \"professor-aluno\" "
             "(destilação de conhecimento): um modelo grande rotula os dados e um modelo pequeno aprende com eles."),
           p(f"Resultado da rotulagem: {num(label_counts['seguro'])} seguro, {num(label_counts['atencao'])} atenção, "
             f"{num(label_counts['perigo'])} perigo e {num(label_counts['descartada'])} descartadas (ambíguas)."
             + (f" Fotos da maquete rotuladas: {len(labels_maq)}." if labels_maq else ""))]
    for label in CLASSES:
        st += figure(ART / f"rotulagem_{label}.jpg", 16.5,
                     f"Amostras rotuladas como <b>{PT[label]}</b>. Vermelho = perigo, laranja = atenção, "
                     "verde = pequeno, branco = fora do corredor central (linhas verde-limão).")

    # ------------------------------------------------------------- 4. divisão
    def split_row(name: str, key: str) -> list:
        parts = division.get(key, {})
        return [name] + [num(len(parts.get(p_, []))) for p_ in ("treino", "validacao", "teste")]

    st += [p("4. Divisão em treino, validação e teste (etapa 3)", "h1"),
           p("Dividimos <b>as fotos de origem</b> em 70% treino, 15% validação e 15% teste <b>antes</b> de montar a "
             "base. Assim nenhum cone, caixa, fundo ou foto real do teste aparece no treino. Se dividíssemos depois, "
             "a mesma caixa poderia estar nas duas partes e o teste pareceria melhor do que é (vazamento de dados). "
             "As fotos reais foram divididas de forma estratificada, mantendo a proporção de cada rótulo."),
           table([["Fonte", "Treino", "Validação", "Teste"],
                  split_row("Fundos (caminhos)", "fundos"), split_row("Cones (recortes)", "cone"),
                  split_row("Caixas (recortes)", "caixa"), split_row("Fotos reais rotuladas", "fotos_reais")]
                 + ([split_row("Fotos da maquete", "maquete")] if any(division.get("maquete", {}).values()) else []),
                 [6.5, 3.5, 3.5, 3.5]),
           p("A <b>validação</b> é usada durante o treino para escolher a melhor época; o <b>teste</b> só é usado uma "
             "vez, na avaliação final.", "small")]

    # ---------------------------------------------------------------- 5. base
    def bc(part: str, label: str, kind: str) -> int:
        return base_counts[(part, label, kind)]

    rows = [["Partição", "Classe", "Fotos reais", "Montagens", "Maquete", "Total"]]
    for part in ("treino", "validacao", "teste"):
        for label in CLASSES:
            r_, c_, m_ = bc(part, label, "foto_real_rotulada"), bc(part, label, "nenhum") + bc(part, label, "cone") + \
                bc(part, label, "caixa"), bc(part, label, "foto_maquete")
            rows.append([part if label == "seguro" else "", PT[label], num(r_), num(c_), num(m_), num(r_ + c_ + m_)])
    st += [p("5. Base rotulada (etapa 4)", "h1"),
           p("A base final junta as <b>fotos reais rotuladas</b> com <b>montagens</b>: um cone ou caixa real colado "
             "numa foto real de caminho, em tamanho e posição controlados (objeto grande e embaixo = perto). As "
             "montagens garantem muitos exemplos de cone e caixa em todas as distâncias e completam cada classe até o "
             "mesmo total, deixando a base <b>equilibrada</b>."),
           table(rows, [2.6, 2.4, 3, 3, 2.6, 2.4])]
    st += figure(ART / "exemplos_base.jpg", 16.5,
                 "Exemplos da base de treino. Linhas: seguro, atenção, perigo. As primeiras colunas são fotos reais; "
                 "as demais, montagens.")

    # --------------------------------------------------------------- 6. treino
    st += [p("6. Treinamento (etapa 5)", "h1"),
           p("A rede começa com <b>pesos aleatórios</b>; nenhum modelo pré-treinado é usado. Arquitetura:"),
           table([["Camada", "Detalhe"],
                  ["Entrada", "imagem 160 × 160 × 3 (quadrado central da câmera), valores de 0 a 1"],
                  ["4 blocos convolucionais", "Conv 3×3, BatchNorm, ReLU e MaxPool, com 16, 32, 64 e 96 filtros"],
                  ["Conv final", "Conv 3×3 com 128 filtros, Global Average Pooling e Dropout de 30%"],
                  ["Classificador", "Dense 64 (ReLU) e Dense 3 (softmax: seguro, atenção, perigo)"]], [4.5, 12.5]),
           Spacer(1, 6),
           table([["Hiperparâmetro", "Valor"],
                  ["Otimizador", "Adam, taxa 0,001 (reduzida pela metade quando a validação estagna)"],
                  ["Função de perda", "Entropia cruzada categórica"],
                  ["Lote / épocas", "32 imagens / até 40 épocas com parada antecipada (paciência 8)"],
                  ["Aumento de dados", "espelhamento horizontal, brilho, contraste e saturação (sem zoom nem "
                                       "deslocamento, que mudariam o rótulo)"],
                  ["Melhor época", (f"{best + 1} de {len(history)} (acurácia de validação "
                                    f"{pct(float(history[best]['val_accuracy']))})") if best is not None else "—"]],
                 [4.5, 12.5])]
    st += figure(ART / "curvas_treino.png", 15.5, "Perda e acurácia por época, no treino e na validação.")

    # ----------------------------------------------------------- 7. avaliação
    per_class = [["Classe", "Precisão", "Revocação", "F1", "Amostras"]] + [
        [PT[c], f"{report[c]['precision']:.3f}", f"{report[c]['recall']:.3f}", f"{report[c]['f1-score']:.3f}",
         num(int(report[c]["support"]))] for c in CLASSES]
    origin_rows = [["Subconjunto do teste", "Amostras", "Acurácia", "Revocação de perigo"]]
    for name, data in (("Fotos reais", real_eval), ("Fotos da maquete", maq_eval), ("Montagens", comp_eval)):
        if data:
            origin_rows.append([name, num(data["amostras"]), pct(data["acuracia"]), pct(data["revocacao_perigo"])])
    st += [PageBreak(), p("7. Avaliação (etapa 6)", "h1"),
           p(f"Avaliamos no conjunto de <b>teste</b> ({num(evaluation['amostras_teste'])} imagens nunca vistas no "
             "treino). A métrica mais importante para a segurança é a <b>revocação de perigo</b>: de todas as "
             "situações realmente perigosas, quantas o modelo detectou."),
           table([["Métrica", "Resultado"],
                  ["Acurácia da CNN", f"<b>{pct(evaluation['acuracia_teste'])}</b>"],
                  ["Acurácia de um modelo simples (regressão logística sobre os pixels)",
                   pct(evaluation["acuracia_modelo_base_regressao_logistica"])],
                  ["Revocação de perigo", f"<b>{pct(evaluation['revocacao_perigo'])}</b>"],
                  ["Casos de perigo classificados como seguro (erro mais grave)",
                   num(evaluation["perigo_classificado_como_seguro"])]], [12, 5]),
           Spacer(1, 8), p("Desempenho por classe", "h2"), table(per_class, [3.4, 3.4, 3.4, 3.4, 3.4]),
           Spacer(1, 8), p("Desempenho por origem da imagem", "h2"),
           p("As montagens são mais fáceis porque seguem a regra exatamente; as fotos reais mostram o desempenho em "
             "cenas do mundo real."),
           table(origin_rows, [7, 3, 3, 4])]
    threshold = evaluation.get("limiar_perigo")
    plain = evaluation.get("sem_limiar_argmax") or {}
    if threshold is not None:
        rows_t = [["Limiar de perigo", "Perigos detectados", "F1 médio", "Acurácia"]]
        for r in evaluation.get("limiar_escolha_validacao", []):
            mark = " (escolhido)" if abs(r["limiar"] - threshold) < 1e-6 else ""
            rows_t.append([f"{r['limiar']:.2f}{mark}".replace(".", ","), pct(r["revocacao_perigo"]),
                           f"{r['f1_medio']:.3f}", pct(r["acuracia"])])
        st += [Spacer(1, 8), p("Ajuste do limiar de decisão", "h2"),
               p("Escolher sempre a classe mais provável deixava passar muitos perigos "
                 f"(revocação de perigo de {pct(plain.get('revocacao_perigo'))} no teste). Como deixar passar um perigo "
                 "é o erro mais grave, o app avisa <b>perigo</b> quando a probabilidade de perigo passa de um limiar. "
                 "O limiar foi escolhido no conjunto de <b>validação</b> (nunca no teste): entre os limiares que "
                 "detectam pelo menos 75% dos perigos, o de maior F1 médio. Resultado: "
                 + f"<b>limiar {threshold:.2f}</b>".replace(".", ",") +
                 f". No teste, a revocação de perigo subiu de {pct(plain.get('revocacao_perigo'))} para "
                 f"{pct(evaluation['revocacao_perigo'])}, com acurácia de {pct(plain.get('acuracia'))} para "
                 f"{pct(evaluation['acuracia_teste'])}. Todas as métricas acima já usam esse limiar."),
               table(rows_t, [4.5, 4.2, 4, 4.3]),
               p("Limiares avaliados no conjunto de validação.", "caption")]
    comparison_path = ART / "comparacao_modelos.json"
    if comparison_path.exists():
        comp = json.loads(comparison_path.read_text(encoding="utf-8"))
        rows_c = [["Modelo", "Acurácia", "Fotos reais", "Perigo (fotos reais)", "Montagens"]]
        for m in comp["modelos"]:
            rows_c.append([m["nome"], pct(m["acuracia"]), pct(m["acuracia_fotos_reais"]),
                           pct(m["revocacao_perigo_fotos_reais"]), pct(m["acuracia_montagens"])])
        st += [Spacer(1, 6), p("Comparação entre as duas versões do modelo", "h2"),
               p("A primeira versão foi treinada só com montagens (cone ou caixa colados em fotos de caminhos) e tinha "
                 "87% de acerto num teste só de montagens. Ao testar as duas versões no <b>mesmo</b> teste, que "
                 "inclui fotos reais, fica claro que a primeira não generalizava para o mundo real. A segunda versão, "
                 "com fotos reais rotuladas, acerta bem mais nas fotos reais; a perda na revocação de perigo foi "
                 "compensada pelo ajuste de limiar acima. " + comp["descricao"]),
               table(rows_c, [6.2, 2.4, 2.6, 3.2, 2.6])]
    st += figure(ART / "matriz_confusao.png", 9.5,
                 "Matriz de confusão no teste: linhas = classe real, colunas = classe prevista.")
    st += figure(ART / "exemplos_erros.jpg", 16.5, "Exemplos de erros do modelo no teste (real × previsto).")
    st += [p("A comparação com a regressão logística mostra que a CNN aprende padrões espaciais (tamanho e posição do "
             "obstáculo) que um modelo linear sobre pixels não consegue captar.")]

    # ---------------------------------------------------------------- 8. app
    st += [p("8. Aplicativo no celular (etapa 7)", "h1"),
           p("O modelo treinado é convertido para <b>TensorFlow.js</b> e roda dentro do navegador do celular, sem "
             "servidor. O app (React + TypeScript) abre a câmera traseira, recorta o quadrado central de cada quadro, "
             "reduz para 160 × 160 e classifica 2 vezes por segundo, usando o mesmo limiar de perigo da avaliação. Em <b>atenção</b> toca um bipe curto e vibra uma "
             "vez; em <b>perigo</b>, um bipe agudo e vibração dupla, repetidos enquanto o perigo continuar."),
           p(f"Endereço: <link href='{APP_URL}' color='blue'>{APP_URL}</link>. A vibração funciona no Android; o "
             "iPhone não permite vibração por sites. O app também tem um modo <b>Coletar dados</b> para fotografar a "
             "maquete.")]

    # ------------------------------------------------------------- 9. maquete
    st += [p("9. Maquete", "h1"),
           p("Testamos o rotulador (OWLv2) em fotos da maquete: ele reconhece os cones listrados, a barreira, a lata, "
             "o pote, o boneco e a planta mesmo em miniatura. Por isso as fotos da maquete podem entrar no pipeline "
             "sem rotulagem manual: basta colocá-las em <font name='Courier'>data/maquete/sem_rotulo/</font> e rodar "
             "o pipeline de novo.")]
    st += figure(ART / "maquete_deteccao.jpg", 16.5, "Detecções do OWLv2 nas fotos da maquete.")
    st += [p("As fotos úteis para treino são as tiradas do ponto de vista do boneco (celular em pé na calçada, "
             "olhando para frente), em várias distâncias de cada obstáculo e também com o caminho livre."
             + (f" Nesta versão, {len(labels_maq)} fotos da maquete foram usadas." if labels_maq else
                " Nesta versão do relatório, as fotos da maquete desse tipo ainda não tinham sido incluídas."))]

    # --------------------------------------------------------- 10. limitações
    st += [p("10. Limitações e próximos passos", "h1"),
           p("• Os rótulos das fotos reais vêm de um detector automático e têm algum ruído (objetos não detectados ou "
             "detectados por engano). Descartar os casos ambíguos reduz, mas não elimina esse ruído.<br/>"
             "• A distância é estimada pela aparência (tamanho e posição na imagem), não medida. Uma foto isolada "
             "também não mede velocidade de aproximação.<br/>"
             "• Há poucos cones diferentes (recortes); cones muito diferentes podem ser confundidos.<br/>"
             "• O detector achou poucas fotos reais de <b>atenção</b> (obstáculo a meia distância); essa classe vem "
             "quase toda de montagens, e nas fotos reais o modelo quase não a usa (prefere seguro ou perigo). "
             "Fotos da maquete a meia distância ajudam a corrigir isso.<br/>"
             "• Com o limiar de perigo mais baixo, o modelo dá mais alarmes falsos em cenas seguras: é uma escolha "
             "consciente de priorizar a segurança.<br/>"
             "• Próximo passo: incluir mais fotos da maquete do ponto de vista do usuário e medir o desempenho "
             "separadamente nelas.<br/>"
             "• É um protótipo acadêmico: não substitui bengala, cão-guia ou orientação profissional.")]

    # --------------------------------------------------------------- 11. uso
    st += [p("11. Como executar", "h1"),
           p("Abrir o app já treinado:"),
           p("npm install<br/>npm run dev", "code"),
           p("Refazer todo o pipeline (Python 3.12):"),
           p("python3.12 -m venv .venv &amp;&amp; source .venv/bin/activate<br/>"
             "pip install -r ml/requirements.txt -r ml/requirements-rotulagem.txt<br/>"
             "bash ml/executar_tudo.sh", "code"),
           p("12. Referências", "h1"),
           p("• Zhou et al. Places: A 10 million Image Database for Scene Recognition (Places365), MIT.<br/>"
             "• Kuznetsova et al. The Open Images Dataset V4/V7, Google.<br/>"
             "• Junhan0518/Traffic_Cone, Hugging Face Datasets (Apache-2.0).<br/>"
             "• Minderer et al. Scaling Open-Vocabulary Object Detection (OWLv2), Google.<br/>"
             "• TensorFlow / Keras e TensorFlow.js; scikit-learn (divisão e métricas).", "small")]

    doc.build(st, onFirstPage=on_page, onLaterPages=on_page)
    print(f"Relatório salvo em {OUT}")


if __name__ == "__main__":
    main()
