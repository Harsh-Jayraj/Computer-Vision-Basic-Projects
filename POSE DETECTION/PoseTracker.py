import mediapipe as mp
import cv2
import time
import Pose_Detection_Module as pm


cap = cv2.VideoCapture("PoseVideos/pose-8.mp4")
cT, pT = 0, 0
detector = pm.pose_detector()
while True:
    success, img = cap.read()
    img = detector.find_pose(img)
    lmlist = detector.find_pose_position(img, draw = False)
    if len(lmlist) != 0:
        cv2.circle(img, (lmlist[15][1], lmlist[15][2]), 10, (0, 340, 20), cv2.FILLED)
        cv2.circle(img, (lmlist[16][1], lmlist[16][2]), 10, (0, 340, 20), cv2.FILLED)


    cT = time.time()
    fps = 1 / (cT - pT)
    pT = cT

    cv2.putText(img, "FPS:" + str(int(fps)), (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 240), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)