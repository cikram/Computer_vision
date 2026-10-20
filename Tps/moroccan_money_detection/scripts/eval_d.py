"""Detector-level test on held-out scenes with held-out false-object styles (fold 0). Usage: eval_d.py weights [...]"""
import sys,json
from pathlib import Path
from ultralytics import YOLO
S=Path(__file__).resolve().parent.parent/'det/syn_d/test'; gt=json.load(open(S/'gt.json'))
def inside(b,g): cx,cy=(b[0]+b[2])/2,(b[1]+b[3])/2; return g[0]<=cx<=g[2] and g[1]<=cy<=g[3]
for w in sys.argv[1:]:
    m=YOLO(w); res={}
    for thr in (0.25,0.5):
        fp=img_fp=found=tot=0; neg_img=neg_fp=0
        for f,g in gt.items():
            r=m.predict(str(S/'images'/f),imgsz=416,conf=0.25,iou=0.5,verbose=False,device='cpu')[0]; hit=set(); n_fp=0
            for b,c,k in zip(r.boxes.xyxy.tolist(),r.boxes.conf.tolist(),r.boxes.cls.tolist()):
                kind='coin' if int(k)==0 else 'note'
                if c<thr or (kind=='coin' and c<0.5): continue
                ok=[i for i,o in enumerate(g) if o['kind']==kind and inside(b,o['box'])]
                if ok: hit.update(ok)
                else: n_fp+=1
            tot+=len(g); found+=len(hit); fp+=n_fp; img_fp+=n_fp>0
        res[thr]=dict(recall=round(found/tot,4),false_boxes=fp,images_with_false_box=img_fp,images=len(gt))
    print(w,json.dumps(res))
