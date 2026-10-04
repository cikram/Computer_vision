import cv2
import numpy as np
import matplotlib.pyplot as plt


# ==================================================
# 1. Lire l'image originale
# ==================================================

original = cv2.imread("image.png")

if original is None:
    print("Erreur : image.png introuvable")
    exit()

# Conversion en niveaux de gris
original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)


# ==================================================
# 2. Ajouter un bruit sel-poivre
# ==================================================

def bruit_sel_poivre(image, proportion=0.05):

    noisy = image.copy()

    hauteur, largeur = image.shape

    nombre_pixels = int(proportion * hauteur * largeur)

    # Sel : pixels blancs
    for _ in range(nombre_pixels // 2):
        x = np.random.randint(0, hauteur)
        y = np.random.randint(0, largeur)
        noisy[x, y] = 255

    # Poivre : pixels noirs
    for _ in range(nombre_pixels // 2):
        x = np.random.randint(0, hauteur)
        y = np.random.randint(0, largeur)
        noisy[x, y] = 0

    return noisy


noisy = bruit_sel_poivre(original, 0.05)


# ==================================================
# 3. Appliquer les 4 filtres
# ==================================================

mean = cv2.blur(noisy, (5, 5))

gaussian = cv2.GaussianBlur(noisy, (5, 5), 0)

median = cv2.medianBlur(noisy, 5)

bilateral = cv2.bilateralFilter(noisy, 9, 75, 75)


# ==================================================
# 4. Fonction MSE
# ==================================================

def mse(img1, img2):

    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)

    return np.mean((img1 - img2) ** 2)


# ==================================================
# 5. MSE entre ORIGINAL et NOISY
# ==================================================

mse_original_noisy = mse(original, noisy)

print("\n========== MSE avec l'image originale ==========")

print("Original vs Noisy     :", mse_original_noisy)

print("Original vs Moyenneur:", mse(original, mean))

print("Original vs Gaussien :", mse(original, gaussian))

print("Original vs Médian   :", mse(original, median))

print("Original vs Bilatéral:", mse(original, bilateral))


# ==================================================
# 6. MSE entre NOISY et les images filtrées
# ==================================================

print("\n========== MSE avec l'image bruitée ==========")

print("Noisy vs Moyenneur:", mse(noisy, mean))

print("Noisy vs Gaussien :", mse(noisy, gaussian))

print("Noisy vs Médian   :", mse(noisy, median))

print("Noisy vs Bilatéral:", mse(noisy, bilateral))


# ==================================================
# 7. Affichage
# ==================================================

images = [
    original,
    noisy,
    mean,
    gaussian,
    median,
    bilateral
]

titles = [
    "Originale",
    "Sel-Poivre",
    "Moyenneur",
    "Gaussien",
    "Médian",
    "Bilatéral"
]

plt.figure(figsize=(12, 8))

for i in range(6):

    plt.subplot(2, 3, i + 1)

    plt.imshow(images[i], cmap="gray")

    plt.title(titles[i])

    plt.axis("off")

plt.tight_layout()
plt.show()