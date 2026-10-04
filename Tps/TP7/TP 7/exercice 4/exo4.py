import cv2
import numpy as np

original = cv2.imread("image_bruitee_4.jpg")

gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

mean = cv2.blur(gray, (5,5))
gaussian = cv2.GaussianBlur(gray, (5,5), 0)
median = cv2.medianBlur(gray, 5)
bilateral = cv2.bilateralFilter(gray, 9, 75, 75)

def mse(img1, img2):
    return np.mean((img1.astype(float) - img2.astype(float)) ** 2)

print("MSE moyenneur :", mse(gray, mean))
print("MSE gaussien :", mse(gray, gaussian))
print("MSE médian :", mse(gray, median))
print("MSE bilatéral :", mse(gray, bilateral))