import cv2

capture = cv2.VideoCapture("ressources/video.mp4")

succes, img = capture.read()

print("Lecture réussie :", succes)

while succes:

    cv2.imshow("Video", img)

    if cv2.waitKey(50) & 0xFF == 27:
        break

    succes, img = capture.read()

capture.release()
cv2.destroyAllWindows()