import cv2
import numpy as np
import matplotlib.pyplot as plt


image = cv2.imread("image.png")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


noisy = image.copy()

prob = 0.5

random = np.random.random(noisy.shape[:2])

noisy[random < prob/2] = 0
noisy[random > 1 - prob/2] = 255

median = cv2.medianBlur(noisy, 5)
gaussian = cv2.GaussianBlur(noisy, (5, 5), 0)
mean = cv2.blur(noisy, (5, 5))


plt.figure(figsize =(12,8))

images = [
    image , 
    noisy , 
    gaussian , 
    mean ,
    median
]


titles = [
    "Originale.jpg" ,   
    "Noisy.jpg" ,
    "Gaussian.jpg" ,   
    "Mean.jpg" ,
    "Median.jpg"
]

for i in range(5):
    plt.subplot(2, 3, i+1)
    plt.imshow(images[i])
    plt.title(titles[i])
    plt.axis("off")
    image = cv2.cvtColor(images[i], cv2.COLOR_RGB2BGR)
    cv2.imwrite(titles[i],image)

plt.show()




