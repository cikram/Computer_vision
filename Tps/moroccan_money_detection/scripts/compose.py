"""Synthetic scenes: coins and notes pasted on real backgrounds. Objects and backgrounds only come from the allowed folds."""
import sys, json, math, random, cv2, numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent)); import distract
ROOT=Path(__file__).resolve().parent.parent; O=ROOT/'det'
CANVAS=640
cut=pd.read_csv(O/'cutouts.csv',dtype={'label':str}); bgs=pd.read_csv(O/'bg.csv')
_cache={}
def load(p):
    if p not in _cache: _cache[p]=cv2.imread(str(O/p),cv2.IMREAD_UNCHANGED)
    return _cache[p]
# relative coin size from phone coins (same camera distance inside a photo): median diameter per class / class 1
pc=cut[(cut.kind=='coin')&(~cut.web.fillna(False).astype(bool))]
med=pc.groupby('label').diam.median(); REL={l:float(med[l]/med['1']) for l in med.index}; REL.setdefault('0.1',0.85); REL.setdefault('2',1.05)
def procedural_bg(rng):
    base=np.array([rng.randint(20,200) for _ in range(3)],np.float32)
    img=np.ones((CANVAS,CANVAS,3),np.float32)*base
    n=cv2.resize(np.random.RandomState(rng.randint(0,10**6)).randn(8,8).astype(np.float32),(CANVAS,CANVAS),interpolation=cv2.INTER_CUBIC)*rng.uniform(5,25)
    fine=np.random.RandomState(rng.randint(0,10**6)).randn(CANVAS,CANVAS,1).astype(np.float32)*rng.uniform(1,6)
    return np.clip(img+n[...,None]+fine,0,255).astype(np.uint8)
def get_bg(folds,rng):
    if rng.random()<0.2: return procedural_bg(rng)
    r=bgs[bgs.fold.isin(folds)].sample(1,random_state=rng.randint(0,10**6)).iloc[0]
    im=load(r.file)[:,:,:3]; h,w=im.shape[:2]
    if min(h,w)<CANVAS:                      # small patch: mirrored tiling keeps the real texture at its real scale
        for ax in (0,1):
            while im.shape[ax]<CANVAS: im=np.concatenate([im,np.flip(im,ax)],axis=ax)
    elif rng.random()<0.5:
        s=rng.uniform(1.0,1.3); im=cv2.resize(im,None,fx=s,fy=s,interpolation=cv2.INTER_CUBIC)
    h,w=im.shape[:2]; x=rng.randint(0,max(w-CANVAS,0)); y=rng.randint(0,max(h-CANVAS,0)); im=im[y:y+CANVAS,x:x+CANVAS]
    if im.shape[:2]!=(CANVAS,CANVAS): im=cv2.resize(im,(CANVAS,CANVAS))
    if rng.random()<.5: im=im[:,::-1]
    return np.array(im,copy=True)          # never hand out a view of the cached background: paste() draws in place
def rotate(rgba,ang):
    h,w=rgba.shape[:2]; M=cv2.getRotationMatrix2D((w/2,h/2),ang,1.0)
    c,s=abs(M[0,0]),abs(M[0,1]); nw,nh=int(h*s+w*c),int(h*c+w*s); M[0,2]+=nw/2-w/2; M[1,2]+=nh/2-h/2
    return cv2.warpAffine(rgba,M,(nw,nh),flags=cv2.INTER_LINEAR,borderValue=(0,0,0,0))
def make_object(row,base_d,rng,is_note):
    rgba=load(row.file).copy(); h,w=rgba.shape[:2]
    if is_note:
        L=base_d*rng.uniform(4.8,6.5); s=L/max(h,w)
    else:
        s=base_d*REL.get(row.label,1.0)/max(h,w)
    rgba=cv2.resize(rgba,None,fx=s,fy=s,interpolation=cv2.INTER_AREA if s<1 else cv2.INTER_CUBIC)
    rgba=rotate(rgba,rng.uniform(0,360) if not is_note else rng.choice([rng.uniform(-25,25),rng.uniform(65,115),rng.uniform(155,205),rng.uniform(245,295)]))
    # colour / brightness jitter of the object only
    rgb=rgba[:,:,:3].astype(np.float32)*rng.uniform(0.75,1.2)+rng.uniform(-18,18); rgba[:,:,:3]=np.clip(rgb,0,255).astype(np.uint8)
    return rgba
def paste(canvas,idmap,rgba,x,y,oid,shadow=True):
    h,w=rgba.shape[:2]; H,Wd=canvas.shape[:2]
    x0,y0,x1,y1=max(x,0),max(y,0),min(x+w,Wd),min(y+h,H)
    if x1<=x0 or y1<=y0: return 0
    sub=rgba[y0-y:y1-y,x0-x:x1-x]; a=(sub[:,:,3:4]/255.0)
    if shadow:
        sh=cv2.GaussianBlur((sub[:,:,3]).astype(np.float32),(0,0),max(3,w*0.03))/255.0
        dx,dy=int(w*0.03),int(h*0.04)
        sx0,sy0=x0+dx,y0+dy; sx1,sy1=min(sx0+sh.shape[1],Wd),min(sy0+sh.shape[0],H)
        if sx1>sx0 and sy1>sy0:
            ss=sh[:sy1-sy0,:sx1-sx0,None]*0.45
            canvas[sy0:sy1,sx0:sx1]=(canvas[sy0:sy1,sx0:sx1]*(1-ss)).astype(np.uint8)
    canvas[y0:y1,x0:x1]=(canvas[y0:y1,x0:x1]*(1-a)+sub[:,:,:3]*a).astype(np.uint8)
    m=sub[:,:,3]>127; idmap[y0:y1,x0:x1][m]=oid
    return int(m.sum())
def distractors(canvas,rng):
    for _ in range(rng.randint(1,4)):
        col=tuple(int(c) for c in np.clip(np.array([rng.randint(30,220)]*3)+np.array([rng.randint(-30,30) for _ in range(3)]),0,255))
        cx,cy=rng.randint(40,CANVAS-40),rng.randint(40,CANVAS-40)
        if rng.random()<.5:
            cv2.ellipse(canvas,(cx,cy),(rng.randint(15,40),rng.randint(15,40)),rng.randint(0,180),0,360,col,-1,cv2.LINE_AA)
        else:
            w,h=rng.randint(30,160),rng.randint(20,110); box=cv2.boxPoints(((cx,cy),(w,h),rng.randint(0,180))).astype(np.int32); cv2.fillPoly(canvas,[box],col,cv2.LINE_AA)
def scene(folds,seed,allow_empty=True,styles=None):
    rng=random.Random(seed); canvas=get_bg(folds,rng); idmap=np.zeros((CANVAS,CANVAS),np.int32)
    if rng.random()<0.45: distractors(canvas,rng)
    pool=cut[cut.fold.isin(folds)]
    notes=pool[pool.kind=='note']; coins=pool[pool.kind=='coin']
    empty=allow_empty and rng.random()<0.06
    nn=0 if empty else rng.choices([0,1,2,3],[0.2,0.4,0.28,0.12])[0]; nc=0 if empty else rng.choices(range(0,9),[0.08,0.2,0.2,0.16,0.12,0.1,0.06,0.04,0.04])[0]
    if nn+nc==0 and not empty: nc=rng.randint(1,4)
    base=rng.uniform(36,70) if rng.random()<0.5 else rng.uniform(70,120); objs=[]; boxes=[]
    def free(x,y,w,h,lim):
        for (a,b,c,d) in boxes:
            iw=min(x+w,a+c)-max(x,a); ih=min(y+h,b+d)-max(y,b)
            if iw>0 and ih>0 and iw*ih/min(w*h,c*d)>lim: return False
        return True
    order=[('note',notes)]*nn+[('coin',coins)]*nc
    oid=0
    for kind,src in order:
        for _ in range(25):
            row=src.sample(1,random_state=rng.randint(0,10**6)).iloc[0]
            o=make_object(row,base,rng,kind=='note'); h,w=o.shape[:2]
            if max(h,w)>CANVAS*0.92: continue
            x=rng.randint(-int(w*.12),CANVAS-int(w*.88)); y=rng.randint(-int(h*.12),CANVAS-int(h*.88))
            lim=0.25 if kind=='note' else (0.9 if rng.random()<0.35 else 0.10)   # coins may lie on a note
            if kind=='coin' and not any(t[0]=='note' for t in objs) and lim==0.9: lim=0.1
            if free(x,y,w,h,lim):
                oid+=1; boxes.append((x,y,w,h)); objs.append((kind,row.label,oid,o,x,y)); break
    gt=[]
    for kind,label,oid,o,x,y in sorted(objs,key=lambda t:(t[0]=='coin',)):   # notes first, coins on top
        paste(canvas,idmap,o,x,y,oid,shadow=True)
    for kind,label,oid,o,x,y in objs:
        m=idmap==oid; full=(o[:,:,3]>127).sum()
        if full==0 or m.sum()<0.55*full or m.sum()<150: continue
        ys,xs=np.where(m); gt.append(dict(kind=kind,label=label,box=[int(xs.min()),int(ys.min()),int(xs.max())+1,int(ys.max())+1]))
    if styles:
        for _ in range(rng.choices([0,1,2,3],[.3,.4,.2,.1])[0] if not empty else rng.randint(1,3)): distract.overlay(canvas,idmap,rng,styles)
    # global degradation
    img=canvas.astype(np.float32)*rng.uniform(0.8,1.15)+rng.uniform(-12,12)
    if rng.random()<0.5: img=cv2.GaussianBlur(img,(0,0),rng.uniform(0.4,1.4))
    if rng.random()<0.5: img+=np.random.RandomState(seed).randn(*img.shape)*rng.uniform(1,6)
    img=np.clip(img,0,255).astype(np.uint8)
    ok,buf=cv2.imencode('.jpg',img,[cv2.IMWRITE_JPEG_QUALITY,rng.randint(45,95)]); img=cv2.imdecode(buf,1)
    return img,gt
def write_set(name,folds,n,seed0,out,allow_empty=True,styles=None):
    (out/name/'images').mkdir(parents=True,exist_ok=True); (out/name/'labels').mkdir(parents=True,exist_ok=True); gts={}
    for i in range(n):
        img,gt=scene(folds,seed0+i,allow_empty,styles); stem=f'syn_{name}_{i:05d}'
        cv2.imwrite(str(out/name/'images'/(stem+'.jpg')),img,[cv2.IMWRITE_JPEG_QUALITY,95])
        lines=[f"{0 if g['kind']=='coin' else 1} {(g['box'][0]+g['box'][2])/2/CANVAS:.6f} {(g['box'][1]+g['box'][3])/2/CANVAS:.6f} {(g['box'][2]-g['box'][0])/CANVAS:.6f} {(g['box'][3]-g['box'][1])/CANVAS:.6f}" for g in gt]
        (out/name/'labels'/(stem+'.txt')).write_text('\n'.join(lines)); gts[stem+'.jpg']=gt
    json.dump(gts,open(out/name/'gt.json','w'))
if __name__=='__main__':
    # compose.py N_train N_val N_test [NAME]   -> with NAME, scenes also get synthetic false objects (logos, icons, rings, shadows);
    # train/val use the TRAIN styles, test uses different TEST styles to measure generalisation
    out=O/(sys.argv[4] if len(sys.argv)>4 else 'syn'); n_train,n_val,n_test=int(sys.argv[1]),int(sys.argv[2]),int(sys.argv[3]); D=len(sys.argv)>4
    write_set('train',[1,2,3],n_train,0,out,styles=distract.TRAIN if D else None); write_set('val',[4],n_val,100000,out,styles=distract.TRAIN if D else None)
    write_set('test',[0],n_test,200000,out,styles=distract.TEST if D else None)
    print('done',REL)
