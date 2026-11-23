"""Matriz de confusão por seção: valores absolutos da regressão logística."""
import csv
from pathlib import Path
from html import escape
from PIL import Image, ImageDraw, ImageFont

ROOT=Path(__file__).resolve().parent
with (ROOT/'resultados-auditaveis/confusao_secoes_RL.csv').open(newline='') as f:
    rows=list(csv.DictReader(f))
labels=['001','004','005','006','025']
matrix=[[int(r[c]) for c in labels] for r in rows]
assert sum(map(sum,matrix))==808
assert sum(matrix[i][i] for i in range(5))==684
image=Image.new('RGB',(1120,1040),'white')
draw=ImageDraw.Draw(image)
font='/System/Library/Fonts/Supplemental/Arial.ttf'
bold='/System/Library/Fonts/Supplemental/Arial Bold.ttf'
svg=['<svg xmlns="http://www.w3.org/2000/svg" width="1120" height="1040" viewBox="0 0 1120 1040">','<rect width="100%" height="100%" fill="white"/>']

def label(txt,x,y,size=30,b=False,color='#17324d',anchor='mt'):
    draw.text((x,y),txt,font=ImageFont.truetype(bold if b else font,size),fill=color,anchor=anchor)
    svg.append(f'<text x="{x}" y="{y+size}" text-anchor="middle" font-family="Arial,sans-serif" font-size="{size}" font-weight="{"bold" if b else "normal"}" fill="{color}">{escape(txt)}</text>')

label('Regressão logística: confusão por seção CDD',560,28,36,True)
label('Seção prevista',660,105,28)
startx,starty,cell=250,205,145
maximum=max(map(max,matrix))
for j,c in enumerate(labels):label(c,startx+(j+.5)*cell,155,31,True)
for i,r in enumerate(labels):
    label(r,170,starty+i*cell+cell/2-19,31,True)
    for j,v in enumerate(matrix[i]):
        intensity=(v/maximum)**0.6
        rgb=tuple(int(246*(1-intensity)+target*intensity) for target in (27,73,112))
        color='#' + ''.join(f'{channel:02x}' for channel in rgb)
        x,y=startx+j*cell,starty+i*cell
        draw.rectangle((x,y,x+cell,y+cell),fill=rgb,outline='white',width=3)
        svg.append(f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" fill="{color}" stroke="white" stroke-width="3"/>')
        label(str(v),x+cell/2,y+cell/2-24,39,True,'white' if intensity>.52 else '#17324d')
label('Linhas: seção de referência. Colunas: seção prevista.',560,952,27)
label('Valores absolutos; n = 808; diagonal = 684 (84,65%).',560,991,27)
svg.append('</svg>')
image.save(ROOT/'matriz-confusao-secoes-RL.png',dpi=(240,240))
(ROOT/'matriz-confusao-secoes-RL.svg').write_text('\n'.join(svg)+'\n',encoding='utf-8')
print(matrix)
