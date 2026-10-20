import sys,json,time,argparse,cv2,numpy as np,pandas as pd
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import pipe_eval as pe
from pathlib import Path
O,P=pe.O,pe.P
ap=argparse.ArgumentParser(); ap.add_argument('--det'); ap.add_argument('--clf',default=str(P/'models/holdout_model_v2.pt')); ap.add_argument('--baseline',action='store_true')
ap.add_argument('--n_syn',type=int,default=300); ap.add_argument('--out',required=True); ap.add_argument('--imgsz',type=int,default=416); ap.add_argument('--det_conf',type=float,default=0.25); ap.add_argument('--coin_det_min',type=float,default=0.0); ap.add_argument('--rot_tta',action='store_true'); ap.add_argument('--contain',type=float,default=0); ap.add_argument('--thr',default='0.6:0.6,0.6:0.8,0.8:0.8')
a=ap.parse_args()
real=json.load(open(O/'real.json')); gts=json.load(open(O/'syn/test/gt.json'))
idx=pd.read_csv(P/'data/merged/index.csv',dtype={'label':str})
sets={}
sets['syn_mixed']=[(str(O/'syn/test/images'/k),v) for k,v in list(gts.items())[:a.n_syn] if v]
sets['real_coin_photos']=[(r['file'],[dict(kind='coin',label=b[5],box=b[1:5]) for b in r['boxes']]) for r in real if r['kind']=='coin_photo' and r['fold']==0]
sets['real_note_photos']=[(str(P/'data/merged'/r.path),[dict(kind='note',label=r.label,box=[0,0,512,512])]) for r in idx[(idx.source=='phone_note')&(idx.fold==0)].itertuples()]
bg=pd.read_csv(O/'bg.csv'); neg=[str(O/f) for f in bg[bg.fold==0].file]+[str(f) for f in sorted((O/'syn/test/images').glob('*.jpg')) if not gts[f.name]]
sets['negatives']=[(f,[]) for f in neg]
if a.baseline:
    sys.path.insert(0,str(P)); import detect
    clf=detect.Classifier(str(P/'models'),'holdout_model.pt'); detect.SEGMENTER=detect.Segmenter()
    def runner(img,thr_coin,thr_note):
        import cv2 as cv
        res,_=detect.analyse(clf,img,thr_coin,None,False)
        out=[]
        for r in res:
            pts=detect._poly(r['geom']).astype(np.float32); x,y,w,h=cv.boundingRect(pts)
            out.append(dict(box=[x,y,x+w,y+h],kind=r['kind'] or 'note',label=('%g'%r['value']),value=r['value'],conf=r['conf'],accepted=r['accepted'] and r['conf']>=(thr_coin if (r['kind']=='coin') else thr_note)))
        return out
else:
    pipe=pe.Pipe(a.det,a.clf,det_conf=a.det_conf,imgsz=a.imgsz,contain=a.contain or None,coin_det_min=a.coin_det_min,rot_tta=a.rot_tta); runner=lambda img,tc,tn: pipe(img,tc,tn)
THR=[tuple(float(x) for x in t.split(':')) for t in a.thr.split(',')]
out={}
for name,items in sets.items():
    t0=time.time(); allres=[]
    for f,gt in items:
        img=cv2.imread(f)
        if a.baseline:
            r6=runner(img,0.6,0.8); allres.append(r6)   # baseline only supports one threshold run; coin 0.6 / note 0.8 applied inside
        else:
            # run detector+classifier once, apply thresholds afterwards
            res=runner(img,0.0,0.0); allres.append(res)
    out[name]={}
    for tc,tn in THR:
        pr=[]
        for res,(f,gt) in zip(allres,items):
            rr=[dict(d,accepted=d['conf']>=(tc if d['kind']=='coin' else tn)) for d in res] if not a.baseline else res
            pr.append((rr,gt))
        if name=='negatives':
            fp=[sum(1 for d in rr if d['accepted']) for rr,_ in pr]; out[name][f'coin{tc}_note{tn}']=dict(images=len(fp),images_with_false_detection=int(sum(1 for x in fp if x>0)),false_detections=int(sum(fp)))
        else: out[name][f'coin{tc}_note{tn}']=pe.score(pr)
        if a.baseline: break
    print(name,len(items),f'{time.time()-t0:.0f}s',json.dumps(out[name]),flush=True)
json.dump(out,open(a.out,'w'),indent=1)
import pickle; pickle.dump(None,open(a.out+'.ok','wb'))
