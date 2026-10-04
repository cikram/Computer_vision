import cv2
import numpy as np

# Definition de fonctions de dessin
def dess_ligne(image, pt_debut, pt_fin, couleur=(255,255,255), epaisseur=1, type_ligne=cv2.LINE_AA):
    cv2.line(image, pt_debut, pt_fin, couleur, epaisseur, type_ligne)
def dess_rectangle(image, haut_gauche, bas_droit, couleur=(255,255,255), epaisseur=1, type_ligne=cv2.LINE_AA):
    cv2.rectangle(image, haut_gauche, bas_droit, couleur, epaisseur, type_ligne)
def dess_cercle(image, centre, rayon, couleur=(255,255,255), epaisseur=1, type_ligne=cv2.LINE_AA):
    cv2.circle(image, centre, rayon, couleur, epaisseur, type_ligne)
def dess_ellipse(image, centre, axes, angle, debut_angle, fin_angle, couleur=(255,255,255), epaisseur=1, type_ligne=cv2.LINE_AA):
    cv2.ellipse(image, centre, axes, angle, debut_angle, fin_angle, couleur, epaisseur, type_ligne)
def dess_polylignes(image, points, est_fermer=True, couleur=(255,255,255), epaisseur=1, type_ligne=cv2.LINE_AA):
    cv2.polylines(image, points, est_fermer, couleur, epaisseur, type_ligne)
def dess_text(image, text, org, taille=1, couleur=(255,255,255), epaisseur=1,
              police=cv2.FONT_HERSHEY_COMPLEX, type_ligne=cv2.LINE_AA):
    cv2.putText(image, text, org, police, taille, couleur, epaisseur, type_ligne)

# Parametres
W, H = 643, 440
S = 4                        # facteur de sur-echantillonnage (1 = sans lissage supplementaire)
FOND   = (218, 218, 218)     # BGR
INDIGO = (121, 40, 49)
BLEU   = (173, 109, 2)
BLANC  = (250, 250, 250)

img = np.zeros((H * S, W * S, 3), np.uint8)
dess_rectangle(img, (0, 0), (W * S - 1, H * S - 1), FOND, -1, cv2.LINE_8)

def remplir(points, couleur):
    """Remplit un polygone avec des lignes horizontales (dess_ligne),
    puis lisse le contour avec dess_polylignes."""
    p = [(x * S, y * S) for x, y in points]
    ys = [q[1] for q in p]
    for y in range(int(min(ys)), int(max(ys)) + 1):
        yc, xs = y + 0.5, []
        for i in range(len(p)):
            (x1, y1), (x2, y2) = p[i], p[(i + 1) % len(p)]
            if y1 != y2 and min(y1, y2) <= yc < max(y1, y2):
                xs.append(x1 + (yc - y1) * (x2 - x1) / (y2 - y1))
        xs.sort()
        for a, b in zip(xs[0::2], xs[1::2]):
            dess_ligne(img, (int(round(a)), y), (int(round(b)), y), couleur, 1, cv2.LINE_8)
    contour = np.array([[round(x), round(y)] for x, y in p], np.int32).reshape(-1, 1, 2)
    dess_polylignes(img, [contour], True, couleur, 1, cv2.LINE_AA)

# 1. triangle blanc dans le A
remplir([(66.7, 175), (42.7, 218.7), (77.3, 196.3)], BLANC)

# 2. barre bleu clair entre le A et le M (sous le A)
remplir([(36.7, 223.3), (137.3, 158.3), (138, 176)], BLEU)

# 3. le M bleu clair
remplir([(137.3, 129), (138, 176), (182.3, 270), (231.7, 175), (289.3, 305),
         (311.7, 305), (231, 129), (230.6, 141), (188.3, 236)], BLEU)

# 4. le A indigo (au-dessus de la barre)
remplir([(68, 129), (146.7, 305), (125, 305), (66.7, 175), (0, 288.3), (67, 141)], INDIGO)

if S > 1:
    img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)

cv2.imwrite("logo_ensam.png", img)
cv2.imshow("logo", img)
cv2.waitKey(0)
cv2.destroyAllWindows()