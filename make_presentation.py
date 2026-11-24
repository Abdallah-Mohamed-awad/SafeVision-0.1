"""Generate the updated evidence-aligned SafeVision AI presentation."""
from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
from pptx.enum.dml import MSO_THEME_COLOR

ROOT = Path(__file__).resolve().parent
DOCS = ROOT / 'docs'
OUT = DOCS / 'SafeVision_AI_Presentation.pptx'
prs = Presentation(); prs.slide_width = Inches(13.333); prs.slide_height = Inches(7.5)
BLANK = prs.slide_layouts[6]
NAVY = RGBColor(9, 19, 35); PANEL = RGBColor(17, 34, 54); PANEL2 = RGBColor(22, 45, 66)
TEAL = RGBColor(30, 190, 170); GREEN = RGBColor(104, 210, 155); WHITE = RGBColor(245, 248, 252)
MUTED = RGBColor(165, 182, 200); ORANGE = RGBColor(242, 175, 83); RED = RGBColor(231, 105, 105)
FONT='Aptos'


def bg(slide, color=NAVY):
    fill=slide.background.fill; fill.solid(); fill.fore_color.rgb=color


def textbox(slide, x,y,w,h,text,size=16,color=WHITE,bold=False,align=None,margin=0.06,font=FONT):
    box=slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf=box.text_frame; tf.clear(); tf.word_wrap=True
    tf.margin_left=Inches(margin); tf.margin_right=Inches(margin); tf.margin_top=Inches(margin); tf.margin_bottom=Inches(margin)
    p=tf.paragraphs[0]; p.text=text
    if align is not None: p.alignment=align
    for r in p.runs:
        r.font.name=font; r.font.size=Pt(size); r.font.bold=bold; r.font.color.rgb=color
    return box


def rect(slide,x,y,w,h,fill=PANEL,line=None,radius=True):
    shape=slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    shape.fill.solid(); shape.fill.fore_color.rgb=fill
    shape.line.color.rgb=line or fill
    return shape


def header(slide, section, title, subtitle=None, n=None):
    textbox(slide,.55,.25,12,.28,section.upper(),10,TEAL,True)
    textbox(slide,.55,.62,12.2,.58,title,26,WHITE,True)
    if subtitle: textbox(slide,.58,1.27,12,.35,subtitle,11,MUTED)
    # bottom footer
    textbox(slide,.58,7.13,10,.18,'SAFEVISION AI  •  NTI GRADUATION PROJECT',8,MUTED,True)
    if n is not None: textbox(slide,12.35,7.10,.4,.2,f'{n:02}',8,MUTED,True,PP_ALIGN.RIGHT)


def bullet_list(slide, x,y,w,items, size=15, color=WHITE, gap=.62, marker=TEAL):
    yy=y
    for item in items:
        dot=slide.shapes.add_shape(MSO_SHAPE.OVAL, Inches(x), Inches(yy+.13), Inches(.10), Inches(.10))
        dot.fill.solid(); dot.fill.fore_color.rgb=marker; dot.line.color.rgb=marker
        textbox(slide,x+.24,yy,w-.24,.48,item,size,color)
        yy+=gap


def metric_card(slide,x,y,w,h,value,label,sub=None,accent=TEAL):
    rect(slide,x,y,w,h,PANEL,PANEL2)
    textbox(slide,x+.18,y+.16,w-.35,.48,value,25,accent,True)
    textbox(slide,x+.18,y+.68,w-.35,.30,label,12,WHITE,True)
    if sub: textbox(slide,x+.18,y+1.02,w-.35,h-1.08,sub,9,MUTED)


def add_slide():
    s=prs.slides.add_slide(BLANK); bg(s); return s

# Slide 1
s=add_slide()
rect(s,.7,.65,11.95,6.1,PANEL)
textbox(s,1.2,1.15,10.8,.35,'NTI GRADUATION PROJECT  |  COMPUTER VISION',11,TEAL,True)
textbox(s,1.15,2.0,11,1.0,'SafeVision AI',38,WHITE,True)
textbox(s,1.2,3.05,10.5,.75,'AI-Powered Workplace Safety &\nPPE Compliance Monitoring',22,MUTED,False)
rect(s,1.2,4.45,4.5,.48,PANEL2)
textbox(s,1.35,4.52,4.2,.28,'PYTORCH  •  YOLO26  •  STREAMLIT',11,GREEN,True)
textbox(s,1.2,5.6,10,.36,'Technical methodology, controlled experiments and deployment design',12,MUTED)
textbox(s,1.2,6.65,10,.2,'November 2026',9,MUTED)

# Slide 2
s=add_slide(); header(s,'01 / Problem','Workplace PPE inspection is difficult to scale','A vision-based assistant can highlight visible evidence for human review.',2)
metric_card(s,.7,2.0,3.75,1.55,'MANUAL','Repeated inspection','Checks across busy and large sites.',TEAL)
metric_card(s,4.8,2.0,3.75,1.55,'VISUAL','Challenging scenes','Lighting, distance, scale and occlusion.',ORANGE)
metric_card(s,8.9,2.0,3.75,1.55,'AI ASSIST','Consistent review support','Flag visible PPE objects and missing-equipment regions.',GREEN)
rect(s,.7,4.15,11.95,1.5,PANEL2)
textbox(s,1.0,4.45,11.3,.8,'SafeVision AI is a decision-support tool. A qualified safety officer remains responsible for interpreting alerts and taking action.',17,WHITE,True,PP_ALIGN.CENTER)

# Slide 3
s=add_slide(); header(s,'02 / Scope','What the first release detects','Object detection of visible PPE and dataset-defined missing-PPE regions.',3)
left=[('PERSON','Person regions'),('PPE','Helmet, vest, gloves, boots, goggles'),('MISSING PPE','no_helmet, no_vest, no_gloves, no_boots, no_goggle'),('OTHER','Forklift')]
for i,(tag,desc) in enumerate(left):
    y=1.9+i*1.08; rect(s,.8,y,5.7,.83,PANEL)
    textbox(s,1.05,y+.12,1.6,.25,tag,10,TEAL,True)
    textbox(s,2.7,y+.1,3.5,.48,desc,13,WHITE)
rect(s,7.0,1.9,5.5,4.1,PANEL2)
textbox(s,7.35,2.25,4.8,.35,'SCOPE BOUNDARY',13,ORANGE,True)
bullet_list(s,7.35,2.85,4.7,[
    'A missing-PPE box is a visible region, not automatically a unique worker violation.',
    'Worker-to-PPE association and tracking are not claimed unless separately implemented and validated.',
    'No automated legal, disciplinary or safety certification decisions.'
],12,gap=.86,marker=ORANGE)

# Slide 4
s=add_slide(); header(s,'03 / Dataset','Construction-PPE dataset','Published dataset split: 800 training images and 200 validation images; no official test split.',4)
metric_card(s,.75,2.0,2.8,1.55,'1,000','Images','Published dataset',TEAL)
metric_card(s,3.85,2.0,2.8,1.55,'14,863','Bounding boxes','Reviewed annotations',GREEN)
metric_card(s,6.95,2.0,2.8,1.55,'12','Classes','PPE, missing PPE, person, forklift',ORANGE)
metric_card(s,10.05,2.0,2.5,1.55,'800 / 200','Train / Val','No official test split',RED)
rect(s,.75,4.15,11.8,1.35,PANEL)
textbox(s,1.05,4.42,11.1,.75,'The improved pipeline audits labels, class frequencies, bounding-box geometry, unreadable images and exact duplicates before training.',15,WHITE,True)

# Slide 5
s=add_slide(); header(s,'04 / Pipeline','From annotated data to a deployable detector','One reproducible path links data quality, model training, evaluation and inference.',5)
steps=[('01','DATA AUDIT','images + labels'),('02','SPLIT','train / val / holdout'),('03','YOLO26','transfer learning'),('04','EVALUATE','per-class metrics'),('05','APP','image / video')]
for i,(num,title,sub) in enumerate(steps):
    x=.6+i*2.55
    rect(s,x,2.6,2.1,1.65,PANEL2)
    textbox(s,x+.2,2.82,1.7,.25,num,11,TEAL,True)
    textbox(s,x+.2,3.22,1.7,.32,title,14,WHITE,True)
    textbox(s,x+.2,3.65,1.7,.34,sub,10,MUTED)
    if i<4:
        ln=s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x+2.12), Inches(3.4), Inches(x+2.47), Inches(3.4))
        ln.line.color.rgb=TEAL; ln.line.width=Pt(2)
rect(s,.8,5.15,11.7,.8,PANEL)
textbox(s,1.05,5.38,11.15,.34,'Final artifact: a selected checkpoint (best.pt) plus measured metrics, diagnostic plots and a Streamlit interface.',14,GREEN,True,PP_ALIGN.CENTER)

# Slide 6
s=add_slide(); header(s,'05 / Deep Learning','Transfer learning with YOLO26','YOLO learns visual representations and predicts class labels, confidence and bounding boxes.',6)
rect(s,.75,1.95,5.5,3.9,PANEL)
textbox(s,1.05,2.25,4.9,.35,'MODEL CANDIDATES',13,TEAL,True)
metric_card(s,1.05,2.85,2.1,1.55,'YOLO26n','Nano','Lightweight baseline',TEAL)
metric_card(s,3.55,2.85,2.1,1.55,'YOLO26s','Small','Larger accuracy candidate',GREEN)
textbox(s,1.1,4.9,4.8,.5,'Both variants use the same split and training settings in the new experiment.',11,MUTED)
rect(s,6.75,1.95,5.8,3.9,PANEL2)
textbox(s,7.1,2.3,5.0,.35,'FEATURE LEARNING',13,TEAL,True)
bullet_list(s,7.1,2.9,4.9,[
    'Convolutional layers learn edges, textures and shapes.',
    'Deeper feature representations support object localization and classification.',
    'Manual HOG/SIFT features are not required for this learned detector.'
],12,gap=.78)

# Slide 7 baseline results
s=add_slide(); header(s,'06 / Historical baseline','Previous validation results','These are actual recorded results from the earlier run, not results of the improved experiment.',7)
headers=['Metric','YOLO26n','YOLO26s']
rows=[('Precision','85.88%','87.76%'),('Recall','68.88%','76.79%'),('mAP50','74.90%','81.55%'),('mAP50-95','53.41%','60.01%')]
t= s.shapes.add_table(5,3, Inches(.8), Inches(2.0), Inches(6.3), Inches(3.7)).table
for j,h in enumerate(headers):
    c=t.cell(0,j); c.text=h; c.fill.solid(); c.fill.fore_color.rgb=PANEL2
for i,row in enumerate(rows,1):
    for j,val in enumerate(row):
        c=t.cell(i,j); c.text=val; c.fill.solid(); c.fill.fore_color.rgb=PANEL if i%2 else NAVY
for row in t.rows:
    for c in row.cells:
        c.margin_left=Inches(.12); c.margin_right=Inches(.12); c.margin_top=Inches(.08); c.margin_bottom=Inches(.08)
        for p in c.text_frame.paragraphs:
            for r in p.runs: r.font.name=FONT; r.font.size=Pt(13); r.font.color.rgb=WHITE
rect(s,7.55,2.0,4.9,3.7,PANEL)
textbox(s,7.9,2.35,4.2,.45,'IMPORTANT CAVEATS',13,ORANGE,True)
bullet_list(s,7.9,3.0,4.0,[
    'Validation metrics only; not an independent test.',
    'Old comparison used batch 8 for Nano and batch 4 for Small.',
    'Old notebook exported Nano because USE_SMALL=False.'
],12,gap=.8,marker=ORANGE)

# Slide 8 improved training
s=add_slide(); header(s,'07 / Improved experiment','Controlled training with regularization','The new training pipeline is prepared; these settings must still be executed on GPU.',8)
params=[('Max epochs','100'),('Early Stopping','patience = 20'),('Optimizer','AdamW'),('Weight decay','0.0005'),('Learning rate','0.001 → 0.00001'),('Scheduler','Cosine + warmup'),('Batch / size','4 / 640 for both'),('Checkpointing','best + last + each 10')]
for i,(a,b) in enumerate(params):
    col=i%2; row=i//2; x=.85+col*6.05; y=1.9+row*1.08
    rect(s,x,y,5.55,.82,PANEL)
    textbox(s,x+.2,y+.12,2.0,.25,a,11,TEAL,True)
    textbox(s,x+2.1,y+.12,3.1,.4,b,13,WHITE,True)

# Slide 9 regularization and diagnostics
s=add_slide(); header(s,'08 / Generalization','Regularization helps; diagnostics verify','Do not add techniques blindly or claim overfitting is solved by Early Stopping alone.',9)
columns=[('REGULARIZATION',['AdamW weight decay','Data augmentation','Pretrained initialization']),('TRAINING CONTROL',['Early Stopping','Cosine learning-rate schedule','Best / last checkpoints']),('DIAGNOSTICS',['Train vs validation losses','mAP over epochs','Per-class false negatives'])]
for i,(title,items) in enumerate(columns):
    x=.7+i*4.25; rect(s,x,2.0,3.9,3.8,PANEL if i!=1 else PANEL2)
    textbox(s,x+.25,2.3,3.4,.35,title,13,TEAL,True)
    bullet_list(s,x+.25,2.95,3.35,items,12,gap=.83)
textbox(s,.9,6.1,11.4,.4,'Overfitting: train loss falls while validation loss rises / mAP declines.  |  Underfitting: both train and validation remain poor.',12,ORANGE,True,PP_ALIGN.CENTER)

# Slide 10 eval and test split
s=add_slide(); header(s,'09 / Evaluation','Evaluate more than one headline metric','Safety-related missed detections require class-specific recall and a transparent test protocol.',10)
metric_card(s,.8,1.95,2.65,1.55,'Precision','Prediction correctness',None,TEAL)
metric_card(s,3.7,1.95,2.65,1.55,'Recall','Ground-truth coverage',None,GREEN)
metric_card(s,6.6,1.95,2.65,1.55,'mAP50-95','Localization quality',None,ORANGE)
metric_card(s,9.5,1.95,2.65,1.55,'Latency','Deployment speed',None,TEAL)
rect(s,.8,4.1,11.35,1.35,PANEL2)
textbox(s,1.1,4.38,10.75,.65,'Selection priority: macro Recall for no_helmet / no_vest / no_goggle / no_gloves / no_boots → mAP50-95 → size and measured latency.',15,WHITE,True,PP_ALIGN.CENTER)
textbox(s,.95,5.8,11.0,.4,'New holdout is taken from unused source-training images. It is same-dataset testing, not external site/camera testing.',11,ORANGE,True,PP_ALIGN.CENTER)

# Slide 11 app
s=add_slide(); header(s,'10 / Application & Deployment','A student-maintainable AI product','Streamlit interface uses one shared model loader and inference pipeline.',11)
blocks=[('IMAGE','Upload / capture'),('VIDEO','Sampled frames'),('INFERENCE','YOLO26 checkpoint'),('REPORT','Annotated media + CSV')]
for i,(a,b) in enumerate(blocks):
    x=.8+i*3.05
    rect(s,x,2.45,2.55,1.65,PANEL2)
    textbox(s,x+.2,2.78,2.1,.35,a,13,TEAL,True)
    textbox(s,x+.2,3.28,2.1,.5,b,12,WHITE)
rect(s,.8,4.8,11.75,1.0,PANEL)
textbox(s,1.1,5.1,11.1,.45,'Deployment assets: requirements, Dockerfile, model storage instructions, Streamlit entry point and smoke tests.',14,GREEN,True,PP_ALIGN.CENTER)

# Slide 12 limitations
s=add_slide(); header(s,'11 / Limitations','Responsible, honest claims are part of engineering','The system assists inspection; it does not replace safety procedures or qualified human review.',12)
bullet_list(s,1.0,2.0,11.2,[
    'Occlusion, low light, distance and unfamiliar scenes may cause false negatives.',
    'A missing-PPE box is not automatically associated with a specific worker.',
    'Video totals count detections across frames, not unique people or incidents.',
    'The same-dataset holdout does not establish generalization to other worksites or cameras.',
    'Workplace video requires permission, privacy safeguards and limited retention.'
],14,gap=.8,marker=ORANGE)

# Slide 13 conclusion
s=add_slide(); bg(s)
rect(s,.7,.7,11.95,6.0,PANEL)
textbox(s,1.15,1.15,10.8,.35,'CONCLUSION',11,TEAL,True)
textbox(s,1.1,1.85,10.9,.8,'SafeVision AI',34,WHITE,True)
textbox(s,1.15,2.75,10.5,.75,'A reproducible deep-learning pipeline for workplace PPE detection and human-reviewed safety alerts.',19,MUTED)
rect(s,1.15,4.0,10.8,1.1,PANEL2)
textbox(s,1.45,4.27,10.2,.55,'NEXT: run improved GPU training → inspect curves → select by safety-aware metrics → test once → deploy.',15,GREEN,True,PP_ALIGN.CENTER)
textbox(s,1.15,5.75,10,.35,'Thank you  |  Questions',17,WHITE,True)
textbox(s,1.15,6.45,10,.2,'SafeVision AI  •  NTI Graduation Project  •  2026',9,MUTED)

prs.core_properties.title='SafeVision AI - Improved Experimental Methodology'
prs.core_properties.subject='YOLO26 PPE detection project'
prs.core_properties.author='SafeVision AI Project Team'
prs.save(OUT)
print(OUT)
