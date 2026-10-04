import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("image_bruitee_2.jpg", cv2.IMREAD_GRAYSCALE)

sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

magnitude = cv2.magnitude(
    sobel_x.astype(np.float32),
    sobel_y.astype(np.float32)
)

sobel_x = cv2.convertScaleAbs(sobel_x)
sobel_y = cv2.convertScaleAbs(sobel_y)

plt.figure(figsize=(16,4))

plt.subplot(1,4,1)
plt.imshow(image, cmap="gray")
plt.title("Originale")

plt.subplot(1,4,2)
plt.imshow(sobel_x, cmap="gray")
plt.title("Sobel X")

plt.subplot(1,4,3)
plt.imshow(sobel_y, cmap="gray")
plt.title("Sobel Y")

plt.subplot(1,4,4)
plt.imshow(magnitude, cmap="gray")
plt.title("Magnitude")


plt.show()