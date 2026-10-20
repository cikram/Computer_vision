"""Detector (YOLO) + classifier (EfficientNet) -> total in DH. Evaluation helpers shared by all experiments."""
import sys, json, numpy as np, cv2, torch
from pathlib import Path
P=Path(__file__).resolve().parent.parent; O=P/'det'
sys.path.insert(0,str(P)); torch.set_num_threads(2)
import moneylib as ml
from PIL import Image
CL=ml.CLASSES; VAL=ml.VALUES; ISNOTE=ml.IS_NOTE
def iou(a,b):
    iw=min(a[2],b[2])-max(a[0],b[0]); ih=min(a[3],b[3])-max(a[1],b[1])
    if iw<=0 or ih<=0: return 0.0
    i=iw*ih; return i/((a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-i)
def contain_suppress(res,thr=0.7):
    """Drop a box that lies mostly inside a more confident box of the same kind (a fold or a corner of a note found twice)."""
    keep=[True]*len(res)
    order=sorted(range(len(res)),key=lambda i:-res[i]['det_conf'])
    for a,i in enumerate(order):
        for j in order[a+1:]:
            if not (keep[i] and keep[j]) or res[i]['kind']!=res[j]['kind']: continue
            bi,bj=res[i]['box'],res[j]['box']; iw=min(bi[2],bj[2])-max(bi[0],bj[0]); ih=min(bi[3],bj[3])-max(bi[1],bj[1])
            if iw<=0 or ih<=0: continue
            small=min((bi[2]-bi[0])*(bi[3]-bi[1]),(bj[2]-bj[0])*(bj[3]-bj[1]))
            if iw*ih/small>thr: keep[j]=False
    return [r for r,k in zip(res,keep) if k]
def crop(img,box,kind,pad):
    x0,y0,x1,y1=box; cx,cy=(x0+x1)/2,(y0+y1)/2; w,h=x1-x0,y1-y0
    if kind=='coin': s=max(w,h)*pad; w=h=s
    else: w,h=w*pad,h*pad
    H,Wd=img.shape[:2]; a,b,c,d=int(cx-w/2),int(cy-h/2),int(cx+w/2),int(cy+h/2)
    p=max(0,-a,-b,c-Wd,d-H)
    if p>0: img=cv2.copyMakeBorder(img,p,p,p,p,cv2.BORDER_REPLICATE); a,b,c,d=a+p,b+p,c+p,d+p
    return img[b:d,a:c]
class Clf:
    def __init__(self,path,rot_tta=False):
        self.m=ml.load_model(path); self.rot=rot_tta
    @torch.no_grad()
    def probs(self,crops,kinds):
        if not crops: return np.zeros((0,len(CL)))
        if self.rot:                                   # coins lie at any angle: average the 4 right-angle rotations
            base=self._probs(crops,kinds); acc=base.copy(); n=1
            for k in (1,2,3):
                rc=[np.ascontiguousarray(np.rot90(c,k)) if kd=='coin' else c for c,kd in zip(crops,kinds)]
                acc=acc+np.where(np.array(kinds)[:,None]=='coin',self._probs(rc,kinds),base); n+=1
            return acc/n
        return self._probs(crops,kinds)
    @torch.no_grad()
    def _probs(self,crops,kinds):
        out=[]
        for i in range(0,len(crops),16):
            xs=[Image.fromarray(cv2.cvtColor(c,cv2.COLOR_BGR2RGB)).resize((ml.IMG_SIZE,)*2,Image.BICUBIC) for c in crops[i:i+16]]
            t=torch.stack([torch.from_numpy(np.asarray(x).copy()).permute(2,0,1) for x in xs])
            out.append(self.m(ml.norm(t).contiguous(memory_format=torch.channels_last)).softmax(1))
        Pr=torch.cat(out).numpy().copy()
        for i,k in enumerate(kinds):
            Pr[i,ISNOTE if k=='coin' else ~ISNOTE]=0; Pr[i]/=Pr[i].sum()
        return Pr
class Pipe:
    def __init__(self,det_w,clf_w,det_conf=0.25,imgsz=640,pad_coin=1.15,pad_note=1.05,contain=None,coin_det_min=0.0,rot_tta=False):
        from ultralytics import YOLO
        self.det=YOLO(str(det_w)); self.clf=Clf(clf_w,rot_tta); self.conf=det_conf; self.imgsz=imgsz; self.pc=pad_coin; self.pn=pad_note; self.contain=contain; self.cdm=coin_det_min
    def __call__(self,img,thr_coin=0.6,thr_note=0.8):
        r=self.det.predict(img,imgsz=self.imgsz,conf=self.conf,iou=0.5,verbose=False,device='cpu')[0]
        res=[]
        for b,c,cls in zip(r.boxes.xyxy.numpy(),r.boxes.conf.numpy(),r.boxes.cls.numpy()):
            res.append(dict(box=[float(x) for x in b],det_conf=float(c),kind='coin' if int(cls)==0 else 'note'))
        res=[d for d in res if d['kind']=='note' or d['det_conf']>=self.cdm]
        if self.contain: res=contain_suppress(res,self.contain)
        crops=[crop(img,d['box'],d['kind'],self.pc if d['kind']=='coin' else self.pn) for d in res]; kinds=[d['kind'] for d in res]
        Pr=self.clf.probs(crops,kinds)
        for d,p in zip(res,Pr):
            i=int(p.argmax()); d.update(label=CL[i],value=float(VAL[i]),conf=float(p[i]))
            thr=thr_coin if d['kind']=='coin' else thr_note
            d['accepted']=d['conf']>=thr
        return res
def total(res): return round(sum(d['value'] for d in res if d['accepted']),2)
def score(preds_gts,iou_thr=0.5):
    """preds_gts: list of (res, gt) with gt=[dict(kind,label,box)]. Returns summary dict."""
    c=dict(photos=0,exact=0,gt=0,found=0,read=0,wrong=0,unsure=0,missed=0,fp=0); errs=[]
    for res,gt in preds_gts:
        c['photos']+=1; used=set()
        for g in sorted(gt,key=lambda g:(g['box'][2]-g['box'][0])*(g['box'][3]-g['box'][1])):
            c['gt']+=1; x0,y0,x1,y1=g['box']; cand=[j for j,d in enumerate(res) if j not in used and x0<=(d['box'][0]+d['box'][2])/2<=x1 and y0<=(d['box'][1]+d['box'][3])/2<=y1]
            if not cand: c['missed']+=1; continue
            best=max(cand,key=lambda j:(res[j]['accepted'],res[j]['conf'])); bi=1
            used.add(best); d=res[best]; c['found']+=1
            if not d['accepted']: c['unsure']+=1
            elif d['label']==g['label']: c['read']+=1
            else: c['wrong']+=1
        c['fp']+=sum(1 for j,d in enumerate(res) if j not in used and d['accepted'])
        t=sum(float(g['label']) for g in gt); p=total(res); errs.append(abs(t-p)); c['exact']+=abs(t-p)<1e-6
    n=max(c['gt'],1); c['recall_found']=c['found']/n; c['read_rate']=c['read']/n; c['exact_total']=c['exact']/max(c['photos'],1)
    c['mean_total_err_dh']=float(np.mean(errs)) if errs else 0.0; c['median_total_err_dh']=float(np.median(errs)) if errs else 0.0
    return {k:(round(v,4) if isinstance(v,float) else v) for k,v in c.items()}
