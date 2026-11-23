"""Gera a figura metodológica em SVG e PNG, com o mesmo conteúdo e layout."""

from pathlib import Path
from html import escape
from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parent
WIDTH, HEIGHT = 1600, 1650
FONT = '/System/Library/Fonts/Supplemental/Arial.ttf'
BOLD = '/System/Library/Fonts/Supplemental/Arial Bold.ttf'
image = Image.new('RGB', (WIDTH, HEIGHT), 'white')
draw = ImageDraw.Draw(image)
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">', '<rect width="100%" height="100%" fill="white"/>']


def text(lines, y, size=33, bold=False, color='#17324d'):
    font = ImageFont.truetype(BOLD if bold else FONT, size)
    for i, line in enumerate(lines):
        cy = y + i * (size + 13)
        bbox = draw.textbbox((0, 0), line, font=font)
        assert bbox[2] - bbox[0] < WIDTH - 100, line
        draw.text((WIDTH / 2, cy), line, font=font, fill=color, anchor='mt')
        svg.append(f'<text x="800" y="{cy + size}" text-anchor="middle" font-family="Arial,sans-serif" font-size="{size}" font-weight="{"bold" if bold else "normal"}" fill="{color}">{escape(line)}</text>')


def box(y, h, lines, fill='#f1f5f9', size=33):
    draw.rounded_rectangle((100, y, 1500, y + h), radius=18, fill=fill, outline='#8ca2b6', width=3)
    svg.append(f'<rect x="100" y="{y}" width="1400" height="{h}" rx="18" fill="{fill}" stroke="#8ca2b6" stroke-width="3"/>')
    block_height = len(lines) * (size + 13) - 13
    text(lines, y + (h - block_height) / 2, size=size)


def arrow(y1, y2):
    draw.line((800, y1, 800, y2 - 12), fill='#17324d', width=5)
    draw.polygon(((788, y2 - 18), (812, y2 - 18), (800, y2)), fill='#17324d')
    svg.append(f'<path d="M800 {y1} V{y2 - 12}" stroke="#17324d" stroke-width="5"/><path d="M788 {y2 - 18} L812 {y2 - 18} L800 {y2} Z" fill="#17324d"/>')


text(['Fluxo da comparação experimental'], 25, size=43, bold=True)
boxes = [
    (105, 155, ['Dumps mensais da Open Library', 'Edições: 31/08/2026  |  Obras: 30/09/2026']),
    (305, 150, ['Normalização da CDD e recorte de 43 códigos', '4.300 edições: 100 por código, limitadas pela ordem do dump']),
    (500, 190, ['Enriquecimento com assuntos e divisão agrupada por obra', 'Estratificação por CDD; semente 42; proporção 60/20/20', 'Filtro de assuntos não vazios: 4.034 registros']),
    (735, 260, ['Treino: 2.423  |  Validação: 803', 'TF-IDF de palavras + regressão logística (C = 10)', 'XLM-R com agregação por autoatenção; encoder preservado', 'Ensemble: XLM-R-AA (0,45) + kNN de caracteres (0,55; k = 9)']),
    (1040, 150, ['Teste: 808 registros', 'Aplicação das configurações registradas; sem novo ajuste']),
    (1235, 190, ['Avaliação e inspeção dos erros', 'Acurácia, macro-F1, níveis CDD; McNemar, Holm e Stuart–Maxwell', 'Auditoria de IDs e títulos; análise de sensibilidade pós-hoc']),
]
for index, (y, height, lines) in enumerate(boxes):
    box(y, height, lines, fill='#eaf1f8' if index in (0, 3) else '#f5f7fa', size=31)
    if index < len(boxes) - 1:
        arrow(y + height + 5, boxes[index + 1][0] - 5)

draw.rounded_rectangle((100, 1470, 1500, 1600), radius=14, fill='#fff5e8', outline='#b7772e', width=2)
svg.append('<rect x="100" y="1470" width="1400" height="130" rx="14" fill="#fff5e8" stroke="#b7772e" stroke-width="2"/>')
text(['Ressalva: os escores neurais arquivados para a validação do ensemble', 'divergem das previsões de validação do método neural isolado.'], 1491, size=29, color='#80501b')
svg.append('</svg>')
image.save(OUT / 'fluxo-metodologia.png', dpi=(240, 240))
(OUT / 'fluxo-metodologia.svg').write_text('\n'.join(svg) + '\n', encoding='utf-8')
print('Figura gerada em SVG e PNG, 1600 × 1650 px.')
