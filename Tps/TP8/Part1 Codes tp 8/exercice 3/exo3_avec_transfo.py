import cv2
import matplotlib.pyplot as plt



def detect_corners(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    corners = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=500,
        qualityLevel=0.05,
        minDistance=6
    )

    result = image.copy()

    if corners is not None:

        corners = corners.astype(int)

        for corner in corners:

            x, y = corner.ravel()

            cv2.circle(
                result,
                (x, y),
                4,
                (0, 0, 255),
                -1
            )

    return result


image = cv2.imread("chess.jpg")


original = image.copy()

original_corners = detect_corners(original)


resized = cv2.resize(
    image,
    None,
    fx=0.5,
    fy=0.5
)

resized_corners = detect_corners(resized)


h, w = image.shape[:2]

center = (w // 2, h // 2)

M = cv2.getRotationMatrix2D(
    center,
    45,     
    1       
)

rotated = cv2.warpAffine(
    image,
    M,
    (w, h)
)

rotated_corners = detect_corners(rotated)

flipped_horizontal = cv2.flip(
    image,
    1
)

flipped_horizontal_corners = detect_corners(
    flipped_horizontal
)

flipped_vertical = cv2.flip(
    image,
    0
)

flipped_vertical_corners = detect_corners(
    flipped_vertical
)

final_image = cv2.resize(
    image,
    None,
    fx=0.5,
    fy=0.5
)

h2, w2 = final_image.shape[:2]

center2 = (w2 // 2, h2 // 2)

M2 = cv2.getRotationMatrix2D(
    center2,
    45,
    1
)

final_image = cv2.warpAffine(
    final_image,
    M2,
    (w2, h2)
)

final_image = cv2.flip(
    final_image,
    1
)

final_image = cv2.flip(
    final_image,
    0
)

final_corners = detect_corners(final_image)


plt.figure(figsize=(15, 10))


plt.subplot(2, 3, 1)

plt.imshow(
    cv2.cvtColor(original_corners, cv2.COLOR_BGR2RGB)
)

plt.title("Originale")
plt.axis("off")


plt.subplot(2, 3, 2)

plt.imshow(
    cv2.cvtColor(resized_corners, cv2.COLOR_BGR2RGB)
)

plt.title("Resize 50%")
plt.axis("off")


plt.subplot(2, 3, 3)

plt.imshow(
    cv2.cvtColor(rotated_corners, cv2.COLOR_BGR2RGB)
)

plt.title("Rotation 45°")
plt.axis("off")


plt.subplot(2, 3, 4)

plt.imshow(
    cv2.cvtColor(
        flipped_horizontal_corners,
        cv2.COLOR_BGR2RGB
    )
)

plt.title("Flip horizontal")
plt.axis("off")


plt.subplot(2, 3, 5)

plt.imshow(
    cv2.cvtColor(
        flipped_vertical_corners,
        cv2.COLOR_BGR2RGB
    )
)

plt.title("Flip vertical")
plt.axis("off")


plt.subplot(2, 3, 6)

plt.imshow(
    cv2.cvtColor(
        final_corners,
        cv2.COLOR_BGR2RGB
    )
)

plt.title("Toutes les transformations")
plt.axis("off")


plt.tight_layout()
plt.show()