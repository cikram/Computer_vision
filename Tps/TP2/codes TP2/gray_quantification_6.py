import cv2
import numpy as np


# ============================================================
# 1. LECTURE ET CONVERSION EN NIVEAUX DE GRIS
# ============================================================

img = cv2.imread('ressources/image1.jpg')

# Conversion BGR → grayscale
# Chaque pixel est compris entre 0 et 255
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

print("Dimensions :", gray.shape)

cv2.imshow("Image originale", gray)


# ============================================================
# 2. CONVERSION : 0-255 → 6 NIVEAUX
# ============================================================

# Au départ, on a 256 valeurs possibles :
# 0, 1, 2, ..., 255
#
# On veut seulement 6 niveaux :
# 0, 1, 2, 3, 4, 5
#
# On divise donc les 256 valeurs en 6 classes.
#
# Largeur d'une classe :
#
# 256 / 6 ≈ 42.67
#
# Formule :
#
# k = floor(p / (256 / 6))
#
# Exemple :
# p = 180
# k = floor(180 / 42.67)
# k = 4
#
# Donc 180 devient le niveau 4.

gray6 = (gray // (256 / 6)).astype(np.uint8)

print("Valeur originale  =", gray[100, 100])
print("Valeur niveau 6   =", gray6[100, 100])

# Vérifier que les seules valeurs présentes sont 0,1,2,3,4,5
print("Niveaux présents :", np.unique(gray6))


# ============================================================
# 3. AFFICHAGE DES 6 NIVEAUX
# ============================================================

# gray6 contient les valeurs :
# 0, 1, 2, 3, 4, 5
#
# Mais cv2.imshow() interprète une image uint8 entre 0 et 255.
# Donc 0,1,2,3,4,5 apparaissent presque tous noirs.
#
# Pour afficher correctement les 6 niveaux, on les transforme
# temporairement en :
#
# 0, 51, 102, 153, 204, 255
#
# Cela ne change pas gray6.

gray6_display = (gray6 * 51).astype(np.uint8)

cv2.imshow("6 niveaux", gray6_display)


# ============================================================
# 4. CONVERSION INVERSE : 6 NIVEAUX → 0-255
# ============================================================

# gray6 contient :
#
# 0, 1, 2, 3, 4, 5
#
# On veut les répartir sur [0,255] :
#
# 0 →   0
# 1 →  51
# 2 → 102
# 3 → 153
# 4 → 204
# 5 → 255
#
# Il y a 5 intervalles entre les 6 niveaux :
#
# 0 → 1 → 2 → 3 → 4 → 5
#
# Donc l'écart entre deux niveaux est :
#
# (255 - 0) / (5 - 0)
# = 255 / 5
# = 51
#
# Formule :
#
# p' = k * (255 / 5)

gray255 = (gray6 * (255 / 5)).astype(np.uint8)

print("Valeur après conversion inverse =", gray255[100, 100])

cv2.imshow("Retour 0-255", gray255)


# ============================================================
# 5. ATTENTION À 256/6 POUR LA CONVERSION INVERSE
# ============================================================

# Si on faisait :
#
# gray6 * (256 / 6)
#
# on obtiendrait :
#
# 0 → 0
# 1 → 42.67
# 2 → 85.33
# 3 → 128
# 4 → 170.67
# 5 → 213.33
#
# Le dernier niveau n'atteindrait donc jamais 255.
#
# 256/6 sert à DIVISER la plage originale en 6 classes.
#
# 255/5 sert à ÉTALER les 6 niveaux sur [0,255].


cv2.waitKey(0)
cv2.destroyAllWindows()