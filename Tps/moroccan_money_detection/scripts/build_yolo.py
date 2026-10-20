import sys
import json,cv2,shutil,numpy as np
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent; O=ROOT/'det'; SYN=sys.argv[1] if len(sys.argv)>1 else 'syn'; Y=O/'yolo'
real=json.load(open(O/'real.json'))
for s in ['train','val']:
    for k in ['images','labels']: (Y/s/k).mkdir(parents=True,exist_ok=True)
    for f in (O/SYN/s/'images').glob('*.jpg'):
        shutil.copy(f,Y/s/'images'/f.name); shutil.copy(O/SYN/s/'labels'/(f.stem+'.txt'),Y/s/'labels'/(f.stem+'.txt'))
n={'train':0,'val':0}
for i,r in enumerate(real):
    if r['fold']==0: continue
    s='val' if r['fold']==4 else 'train'
    img=cv2.imread(r['file']); H,Wd=img.shape[:2]; stem=f"real_{r['kind']}_{i}"
    cv2.imwrite(str(Y/s/'images'/(stem+'.jpg')),img,[cv2.IMWRITE_JPEG_QUALITY,95])
    lines=[]
    for b in r['boxes']:
        c,x0,y0,x1,y1=b[:5]; lines.append(f"{c} {(x0+x1)/2/Wd:.6f} {(y0+y1)/2/H:.6f} {(x1-x0)/Wd:.6f} {(y1-y0)/H:.6f}")
    (Y/s/'labels'/(stem+'.txt')).write_text('\n'.join(lines)); n[s]+=1
(Y/'data.yaml').write_text(f"path: {Y}\ntrain: train/images\nval: val/images\nnames:\n  0: coin\n  1: note\n")
print('real images added',n,{s:len(list((Y/s/'images').glob('*'))) for s in ['train','val']})
