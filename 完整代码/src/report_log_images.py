"""Typeset actual log excerpts as figures; these are not desktop screenshots."""
from pathlib import Path
import textwrap
from PIL import Image, ImageDraw, ImageFont
from common import ROOT

def render(lines,title,source,name):
    def font_path(windows,linux):
        return next(str(p) for p in [Path(windows),Path(linux)] if p.exists())
    font=ImageFont.truetype(font_path('C:/Windows/Fonts/consola.ttf','/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf'),23)
    heading=ImageFont.truetype(font_path('C:/Windows/Fonts/arialbd.ttf','/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'),25)
    small=ImageFont.truetype(font_path('C:/Windows/Fonts/arial.ttf','/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'),17)
    wrapped=[]
    for line in lines:
        wrapped+=textwrap.wrap(line,width=76,replace_whitespace=False,drop_whitespace=False) or ['']
    width,height=1120,140+len(wrapped)*34
    canvas=Image.new('RGB',(width,height),'white'); draw=ImageDraw.Draw(canvas)
    draw.text((35,25),title,font=heading,fill='#243b50')
    draw.text((35,65),'Source: '+source+'  |  Actual file excerpt',font=small,fill='#546a7b')
    y=105
    for line in wrapped:
        draw.text((35,y),line,font=font,fill='#152e42'); y+=34
    canvas.save(ROOT/'report'/'assets'/name)

def main():
    lines=(ROOT/'logs'/'train_bert.log').read_text(encoding='utf-8').splitlines()
    selected=[line for line in lines if line.startswith('Device ') or 'step 288/288' in line or line.startswith('FINISHED BERT')]
    render(selected,'BERT training log excerpt','logs/train_bert.log','training_log_excerpt.png')
    lines=(ROOT/'logs'/'report_metric_check.log').read_text(encoding='utf-8').splitlines()
    render(lines,'Test metric verification output','logs/report_metric_check.log','metric_output_excerpt.png')
    print('Typeset two actual log excerpts; no desktop screenshots fabricated')

if __name__=='__main__': main()
