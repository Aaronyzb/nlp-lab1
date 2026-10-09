"""Build the illustrated report from saved results, source code and actual logs."""
from pathlib import Path
import ast
import html
import re
import textwrap
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
                               PageBreak, Image, KeepTogether, Preformatted)
from common import ROOT
from report_details import build_sections

class Report:
    def __init__(self):
        self.width=A4[0]-100
        self.story=[]; self.markdown=[]; self.figure_number=0; self.sections=[]
        choices={
            'CNBody':['C:/Windows/Fonts/simsun.ttc','/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc'],
            'CNHead':['C:/Windows/Fonts/simhei.ttf','/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc'],
            'Code':['C:/Windows/Fonts/consola.ttf','/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf']}
        for name,paths in choices.items():
            path=next((Path(p) for p in paths if Path(p).exists()),None)
            if path is None: raise FileNotFoundError('Missing font for '+name)
            pdfmetrics.registerFont(TTFont(name,str(path),subfontIndex=0))
        self.styles={
            'title':ParagraphStyle('Title',fontName='CNHead',fontSize=21,leading=29,spaceAfter=13),
            'h1':ParagraphStyle('Heading',fontName='CNHead',fontSize=15.5,leading=22,spaceAfter=11,keepWithNext=True),
            'h2':ParagraphStyle('Subheading',fontName='CNHead',fontSize=11.5,leading=17,spaceBefore=8,spaceAfter=6,keepWithNext=True),
            'body':ParagraphStyle('Body',fontName='CNBody',fontSize=10.5,leading=16.5,spaceAfter=8,wordWrap='CJK'),
            'small':ParagraphStyle('Small',fontName='CNBody',fontSize=9,leading=13.5,spaceAfter=7,wordWrap='CJK'),
            'cell':ParagraphStyle('Cell',fontName='CNBody',fontSize=9,leading=12.8,wordWrap='CJK'),
            'caption':ParagraphStyle('Caption',fontName='CNBody',fontSize=9,leading=13.2,spaceBefore=5,spaceAfter=9,alignment=1),
            'code':ParagraphStyle('Code',fontName='Code',fontSize=8.2,leading=11.8,spaceAfter=6,
                   backColor=colors.HexColor('#F3F6F9'),borderColor=colors.HexColor('#DCE4EB'),borderWidth=.4,borderPadding=8),
        }

    def p(self,text,style='body'):
        self.story.append(Paragraph(text,self.styles[style]))
        prefix={'title':'# ','h1':'## ','h2':'### '}.get(style,'')
        self.markdown.append(prefix+html.unescape(text)+'\n')

    def h(self,text): self.p(text,'h2')

    def section(self,title):
        if self.sections: self.story.append(PageBreak()); self.markdown.append('\n---\n')
        self.sections.append(title); self.p(title,'h1')

    def table(self,headers,rows,widths=None,padding=5.5):
        cells=[[Paragraph(html.escape(str(value)),self.styles['cell']) for value in row] for row in [headers,*rows]]
        item=Table(cells,colWidths=widths or [self.width/len(headers)]*len(headers),repeatRows=1,hAlign='LEFT')
        item.setStyle(TableStyle([
            ('BACKGROUND',(0,0),(-1,0),colors.HexColor('#E7EEF5')),
            ('GRID',(0,0),(-1,-1),.5,colors.HexColor('#D9D9D9')),
            ('VALIGN',(0,0),(-1,-1),'MIDDLE'),
            ('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),
            ('TOPPADDING',(0,0),(-1,-1),padding),('BOTTOMPADDING',(0,0),(-1,-1),padding),
            ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#F7F9FB')])]))
        self.story.extend([item,Spacer(1,9)])
        self.markdown+=['| '+' | '.join(map(str,headers))+' |','| '+' | '.join(['---']*len(headers))+' |']
        self.markdown+=['| '+' | '.join(map(str,row))+' |' for row in rows]+['']

    def figure(self,name,caption,width=None):
        from PIL import Image as PILImage
        path=ROOT/'report'/'assets'/name
        with PILImage.open(path) as im: iw,ih=im.size
        width=width or self.width
        self.figure_number+=1
        full=f'图 {self.figure_number} '+caption
        self.story.append(KeepTogether([Image(str(path),width=width,height=width*ih/iw),
                                       Paragraph(full,self.styles['caption'])]))
        self.markdown.append(f'![{full}](assets/{name})\n\n*{full}*\n')

    def code(self,text,source):
        rendered=[]
        for line in text.splitlines():
            rendered+=textwrap.wrap(line,width=94,replace_whitespace=False,drop_whitespace=False,
                                    subsequent_indent='       ') or ['']
        block=Preformatted('\n'.join(rendered),self.styles['code'])
        self.story.append(KeepTogether([block,Paragraph(html.escape(source)+'；长行仅折行显示。',self.styles['small'])]))
        code_text='\n'.join(re.sub(r'^\s*\d+  ', '', line) for line in text.splitlines())
        code_text=textwrap.dedent(code_text)
        self.markdown.append('```python\n'+code_text+'\n```\n\n来源：`'+source+'`\n')

    def excerpt(self,path,needle,count,before=0):
        lines=(ROOT/path).read_text(encoding='utf-8').splitlines()
        start=next(i for i,line in enumerate(lines) if needle in line)-before
        start=max(start,0); end=min(len(lines),start+count)
        self.code('\n'.join(f'{i+1:3d}  {lines[i]}' for i in range(start,end)),f'{path} 第 {start+1} 至 {end} 行')

    def function(self,path,name):
        source=(ROOT/path).read_text(encoding='utf-8'); lines=source.splitlines()
        node=next(n for n in ast.parse(source).body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==name)
        self.code('\n'.join(f'{i+1:3d}  {lines[i]}' for i in range(node.lineno-1,node.end_lineno)),
                  f'{path} 第 {node.lineno} 至 {node.end_lineno} 行')

    def log(self,text,source):
        rendered=[]
        for line in text.splitlines():
            rendered+=textwrap.wrap(line,width=94,replace_whitespace=False,drop_whitespace=False) or ['']
        self.story.append(Preformatted('\n'.join(rendered),self.styles['code']))
        self.p('来源 '+html.escape(source),'small')
        self.markdown.append('```text\n'+text+'\n```\n')

    def save(self):
        folder=ROOT/'report'; folder.mkdir(exist_ok=True)
        pdf=folder/'文本分类实验报告.pdf'
        def footer(canvas,doc):
            canvas.setFont('CNBody',8); canvas.setFillColor(colors.HexColor('#666666'))
            canvas.drawString(50,29,'自然语言处理作业一  文本分类实验')
            canvas.drawRightString(A4[0]-50,29,str(doc.page))
        doc=SimpleDocTemplate(str(pdf),pagesize=A4,leftMargin=50,rightMargin=50,topMargin=43,bottomMargin=49,
                              title='文本分类实验报告',author='')
        doc.build(self.story,onFirstPage=footer,onLaterPages=footer)
        (folder/'实验报告.md').write_text('\n'.join(self.markdown),encoding='utf-8')
        print(f'Generated {pdf}; {len(self.sections)} sections, {self.figure_number} figures',flush=True)

def main():
    required=['class_distribution.png','model_scores.png','validation_search.png','learning_curve.png',
              'class_f1.png','all_confusions.png','embedding_coverage.png','score_intervals.png','corpus_sizes.png']
    if any(not (ROOT/'report'/'assets'/name).exists() for name in required):
        from report_charts import main as charts
        charts()
    if not (ROOT/'report'/'assets'/'training_log_excerpt.png').exists():
        from report_log_images import main as logs
        logs()
    report=Report(); build_sections(report); report.save()

if __name__=='__main__': main()
