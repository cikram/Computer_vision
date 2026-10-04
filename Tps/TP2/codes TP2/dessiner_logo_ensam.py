import cv2
import numpy as np

W, H = 643, 440
S = 6  # supersampling for smooth edges

BG     = (218, 218, 218)  # BGR
INDIGO = (121, 40, 49)
BLUE   = (173, 109, 2)
WHITE  = (250, 250, 250)

img = np.full((H * S, W * S, 3), BG, np.uint8)

def poly(pts, color):
    p = np.round(np.array(pts, np.float32) * S).astype(np.int32)
    cv2.fillPoly(img, [p], color, cv2.LINE_AA)

# 1. white triangle inside the A
poly([(66.7, 175), (42.7, 218.7), (77.3, 196.3)], WHITE)

# 2. light-blue bar linking the A to the M (drawn under the A)
poly([(36.7, 223.3), (137.3, 158.3), (138, 176)], BLUE)

# 3. light-blue M (two peaks, one valley)
poly([
    (137.3, 130),   # left apex
    (138.0, 176),
    (182.3, 270),   # valley bottom
    (231.7, 175),   # notch under right peak
    (289.3, 305),
    (311.7, 305),
    (231.0, 130),   # right apex
    (188.3, 236),   # valley inner vertex
], BLUE)

# 4. indigo A (on top of the bar)
poly([
    (68.3, 130),    # apex
    (146.7, 305),
    (125.0, 305),
    (66.7, 175),    # inner apex
    (0, 288.3),     # tip of the left leg
], INDIGO)

img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
cv2.imshow("ensam_logo.png", img)
cv2.waitKey(0)