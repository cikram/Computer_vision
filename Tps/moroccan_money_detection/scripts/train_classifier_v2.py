"""Classifier v2: same recipe as notebook 02 (new), plus (a) tight note crops, (b) crops taken from synthetic training scenes,
(c) low-resolution augmentation so that small coins are not out of distribution. mode=holdout trains on folds 1-4, mode=all on everything."""
import sys,json,random,io,time,cv2,numpy as np,pandas as pd,torch
from pathlib import Path
from PIL import Image
P=Path(__file__).resolve().parent.parent; O=P/'det'
sys.path.insert(0,str(P)); sys.path.insert(0,str(P/'scripts'))
mode=sys.argv[1]; threads=int(sys.argv[2]); epochs=int(sys.argv[3]); out=sys.argv[4]
torch.set_num_threads(threads)
import moneylib as ml
_orig=ml.camera_aug
def camera_aug2(x):
    if random.random()<0.5:                                   # low resolution: a small coin / far note, then upsampled
        s=random.randint(40,200); c,h,w=x.shape
        im=Image.fromarray(x.permute(1,2,0).numpy()).resize((s,s),Image.BILINEAR).resize((w,h),random.choice([Image.BILINEAR,Image.BICUBIC]))
        x=torch.from_numpy(np.asarray(im).copy()).permute(2,0,1)
    return _orig(x)
ml.camera_aug=camera_aug2
df=ml.load_index(); keep=(df.fold!=0) if mode=='holdout' else (df.fold>=0)
tr=df[keep].reset_index(drop=True)
X=ml.load_images(tr,ml.PRE); y=list(tr.y.values); pol=[ml.policy_of(r) for r in tr.itertuples()]
def rsz(img_bgr):
    im=Image.fromarray(cv2.cvtColor(img_bgr,cv2.COLOR_BGR2RGB)).resize((ml.PRE,ml.PRE),Image.BICUBIC); return torch.from_numpy(np.asarray(im).copy()).permute(2,0,1).contiguous()
cut=pd.read_csv(O/'cutouts.csv',dtype={'label':str}); cut=cut[(cut.kind=='note')&(cut.web!=True)&(cut.fold!=0 if mode=='holdout' else cut.fold>=0)]
for r in cut.itertuples():                                     # tight note crops (original pixels inside the box, background included)
    im=cv2.imread(str(O/r.file),cv2.IMREAD_UNCHANGED)[:,:,:3]; X.append(rsz(im)); y.append(ml.CLASSES.index(r.label)); pol.append('note')
gts=json.load(open(O/'syn/train/gt.json')); rng=random.Random(0); objs=[(k,g) for k,v in gts.items() for g in v]; rng.shuffle(objs)
import pipe_eval as pe
nsyn=0; cache={}
for k,g in objs[:1800]:
    if k not in cache: cache={k:cv2.imread(str(O/'syn/train/images'/k))}
    c=pe.crop(cache[k],g['box'],g['kind'],1.15 if g['kind']=='coin' else 1.05)
    if min(c.shape[:2])<8: continue
    X.append(rsz(c)); y.append(ml.CLASSES.index(g['label'])); pol.append('coin' if g['kind']=='coin' else 'note'); nsyn+=1
y=np.array(y); print('train samples',len(X),'(index',len(tr),'note crops',len(cut),'synthetic crops',nsyn,')',np.bincount(y,minlength=11).tolist(),flush=True)
t0=time.time(); m,h=ml.train_model(X,y,pol,recipe='new',epochs=epochs,log=lambda s:print(s,flush=True))
ml.save_checkpoint(m,out,extra={'recipe':'new+v2','trained_on':mode}); print('saved',out,f'{time.time()-t0:.0f}s')
