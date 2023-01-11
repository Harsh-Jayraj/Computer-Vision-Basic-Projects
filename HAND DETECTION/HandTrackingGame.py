import cv2
import mediapipe as mp
import time
import Hand_Detection_Module as HDM


cap = cv2.VideoCapture(0)
cT, pT = 0, 0
detector = HDM.hand_detector()
while True:
    success, img = cap.read()
    img = detector.find_hands(img)
    lmlist = detector.find_position(img)
    if len(lmlist) != 0:
         print(lmlist[8])

    cT = time.time()
    fps = 1 / (cT - pT)
    pT = cT

    cv2.putText(img, "FPS:" + str(int(fps)), (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 240), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)