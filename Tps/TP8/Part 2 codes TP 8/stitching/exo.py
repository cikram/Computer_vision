import cv2, glob

paths = sorted(glob.glob("parts/*.jpg"))
imgs = [cv2.imread(p) for p in paths]
if len(imgs) < 2 or any(img is None for img in imgs):
    raise RuntimeError("Il faut au moins deux images JPG valides dans parts/")

# Les morceaux sont tres petits : les agrandir aide OpenCV a trouver les points communs.
imgs = [cv2.resize(img, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
        for img in imgs]

stitcher = cv2.Stitcher_create(cv2.Stitcher_SCANS)   
stitcher.setPanoConfidenceThresh(0.3)
status, pano = stitcher.stitch(imgs)

if status == cv2.Stitcher_OK:
    cv2.imwrite("pano.jpg", pano)
else:
    
    print("failed:", status)