"""Synthetic false objects (logos, icons, rings, shadows) drawn as overlays. TRAIN styles and TEST styles are different on purpose."""
import cv2, numpy as np
def _col(rng):
    if rng.random()<0.65:
        v=rng.randint(40,235); return tuple(int(np.clip(v+rng.randint(-12,12),0,255)) for _ in range(3))
    return tuple(rng.randint(30,230) for _ in range(3))
def _contrast(c): return (255,255,255) if sum(c)/3<130 else (35,35,35)
def _new(s): return np.zeros((s,s,3),np.uint8),np.zeros((s,s),np.uint8)
def _disc(s,rng,r=0.46,shade=None):
    bgr,a=_new(s); c=_col(rng); yy,xx=np.mgrid[:s,:s]; d=np.hypot(xx-s/2,yy-s/2)/(s*r)
    k=shade if shade is not None else rng.uniform(0,0.5); sh=1-k*np.clip(d,0,1)*rng.choice([-1,1])
    bgr[:]=np.clip(np.array(c,np.float32)[None,None]*sh[...,None],0,255).astype(np.uint8)
    cv2.circle(a,(s//2,s//2),int(s*r),255,-1,cv2.LINE_AA); return bgr,a,c
def ring(s,rng):
    bgr,a,c=_disc(s,rng); cv2.circle(a,(s//2,s//2),int(s*rng.uniform(.2,.34)),0,-1,cv2.LINE_AA); return bgr,a
def disc(s,rng):
    bgr,a,c=_disc(s,rng)
    if rng.random()<.5: cv2.circle(bgr,(s//2,s//2),int(s*.28),_col(rng),-1,cv2.LINE_AA)
    return bgr,a
def play(s,rng):
    bgr,a,c=_disc(s,rng); t=_contrast(c); p=np.array([[.38,.3],[.38,.7],[.72,.5]])*s
    cv2.fillPoly(bgr,[p.astype(np.int32)],t,cv2.LINE_AA); return bgr,a
def wifi(s,rng):
    bgr,a,c=_disc(s,rng); t=_contrast(c); cx,cy=s//2,int(s*.62)
    for i in range(rng.randint(2,3)):
        r=int(s*(.14+.12*i)); cv2.ellipse(bgr,(cx,cy),(r,r),0,-135,-45,t,max(2,s//18),cv2.LINE_AA)
    cv2.circle(bgr,(cx,cy),max(2,s//14),t,-1,cv2.LINE_AA); return bgr,a
def text(s,rng):
    bgr,a,c=_disc(s,rng) if rng.random()<.6 else (lambda b,m:(b,m,(120,)*3))(*_new(s))
    if a.max()==0:
        c=_col(rng); bgr[:]=c; cv2.rectangle(a,(2,int(s*.25)),(s-2,int(s*.75)),255,-1)
    w=rng.choice(['REC','HD','AI','X','M','LIVE','OK','4K','+','TV']); sc=s/70*rng.uniform(.6,1.0)
    (tw,th),_=cv2.getTextSize(w,cv2.FONT_HERSHEY_SIMPLEX,sc,2); cv2.putText(bgr,w,((s-tw)//2,(s+th)//2),cv2.FONT_HERSHEY_SIMPLEX,sc,_contrast(c),2,cv2.LINE_AA); return bgr,a
def shadow(s,rng):
    bgr,a=_new(s); bgr[:]=(15,15,15); m=np.zeros((s,s),np.float32); cv2.ellipse(m,(s//2,s//2),(int(s*rng.uniform(.2,.45)),int(s*rng.uniform(.15,.45))),rng.randint(0,180),0,360,1,-1)
    m=cv2.GaussianBlur(m,(0,0),s*.12); return bgr,(m*rng.uniform(120,230)).astype(np.uint8)
def button(s,rng):
    bgr,a=_new(s); c=_col(rng); bgr[:]=c; rad=int(s*.2); cv2.rectangle(a,(rad,rad),(s-rad,s-rad),255,-1); 
    for (x,y) in [(rad,rad),(s-rad,rad),(rad,s-rad),(s-rad,s-rad)]: cv2.circle(a,(x,y),rad,255,-1,cv2.LINE_AA)
    t=_contrast(c); cv2.line(bgr,(s//2,int(s*.3)),(s//2,int(s*.7)),t,max(2,s//14),cv2.LINE_AA); cv2.line(bgr,(int(s*.3),s//2),(int(s*.7),s//2),t,max(2,s//14),cv2.LINE_AA); return bgr,a
TRAIN=[ring,disc,play,wifi,text,shadow,button]
# ---- held-out styles: never seen during training
def target(s,rng):
    bgr,a=_new(s); cols=[_col(rng) for _ in range(3)]
    for i,r in enumerate([.46,.34,.22,.1]): cv2.circle(bgr,(s//2,s//2),int(s*r),cols[i%3],-1,cv2.LINE_AA)
    cv2.circle(a,(s//2,s//2),int(s*.46),255,-1,cv2.LINE_AA); return bgr,a
def camera(s,rng):
    bgr,a=_new(s); c=_col(rng); bgr[:]=c; cv2.rectangle(a,(2,int(s*.22)),(s-2,int(s*.8)),255,-1); t=_contrast(c)
    cv2.circle(bgr,(s//2,s//2),int(s*.2),t,max(2,s//12),cv2.LINE_AA); cv2.circle(bgr,(s//2,s//2),int(s*.08),t,-1,cv2.LINE_AA); return bgr,a
def badge(s,rng):
    bgr,a,c=_disc(s,rng); t=_contrast(c); p=np.array([[.3,.52],[.45,.67],[.72,.35]])*s
    cv2.polylines(bgr,[p.astype(np.int32)],False,t,max(2,s//11),cv2.LINE_AA); return bgr,a
def chevron_ring(s,rng):                      # looks like a broadcast / hotspot logo
    bgr,a,c=_disc(s,rng,shade=rng.uniform(.1,.3)); cv2.circle(a,(s//2,s//2),int(s*.31),0,-1,cv2.LINE_AA); t=_col(rng) if rng.random()<.3 else (200,)*3
    cv2.circle(bgr,(s//2,s//2),int(s*.18),(120,)*3,max(2,s//10),cv2.LINE_AA); cv2.circle(a,(s//2,s//2),int(s*.18),255,max(2,s//10),cv2.LINE_AA)
    p=np.array([[.3,.78],[.5,.5],[.7,.78]])*s; cv2.polylines(bgr,[p.astype(np.int32)],False,t,max(3,s//8),cv2.LINE_AA); cv2.polylines(a,[p.astype(np.int32)],False,255,max(3,s//8),cv2.LINE_AA); return bgr,a
def silver_disc(s,rng):                       # hard negative: shiny grey disc with a faint pattern
    bgr,a,c=_disc(s,rng,shade=rng.uniform(.2,.5)); n=np.random.RandomState(rng.randint(0,10**6)).randn(s,s,1)*rng.uniform(4,12)
    bgr=np.clip(bgr+n,0,255).astype(np.uint8); return bgr,a
TEST=[target,camera,badge,chevron_ring,silver_disc]
def overlay(canvas,idmap,rng,styles):
    s=rng.randint(28,150); fn=rng.choice(styles); bgr,a=fn(s,rng)
    if rng.random()<.5: bgr=cv2.GaussianBlur(bgr,(0,0),rng.uniform(.5,1.2)); a=cv2.GaussianBlur(a,(0,0),.6)
    H,W=canvas.shape[:2]; x=rng.randint(-s//6,W-s+s//6); y=rng.randint(-s//6,H-s+s//6)
    x0,y0,x1,y1=max(x,0),max(y,0),min(x+s,W),min(y+s,H)
    if x1<=x0 or y1<=y0: return
    if (idmap[y0:y1,x0:x1]>0).mean()>0.02: return          # never hide a labelled object
    m=(a[y0-y:y1-y,x0-x:x1-x,None]/255.0)*rng.uniform(.75,1.0); sub=bgr[y0-y:y1-y,x0-x:x1-x]
    canvas[y0:y1,x0:x1]=(canvas[y0:y1,x0:x1]*(1-m)+sub*m).astype(np.uint8)
