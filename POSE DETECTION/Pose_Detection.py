import mediapipe as mp
import cv2
import time

cap=cv2.VideoCapture("PoseVideos/pose-1.mp4")

mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

ctime, ptime = 0, 0
while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)
    if results.pose_landmarks:
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
        for id, ldmk in enumerate(results.pose_landmarks.landmark):
            h, w, c = img.shape
            cx, cy = int(ldmk.x * w), int(ldmk.y * h)
            cv2.circle(img, (cx, cy), 7, (100, 255, 0), cv2.FILLED)

    #FPS VIEWER
    ctime= time.time()
    fps = 1/(ctime-ptime)
    ptime = ctime

    cv2.imshow("Video", img)
    cv2.waitKey(1)