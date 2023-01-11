import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)
mpFace =mp.solutions.face_detection
face = mpFace.FaceDetection(.75)
mpdraw = mp.solutions.drawing_utils

cT, pT = 0, 0

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results = face.process(imgRGB)
    #print(results.detections)
    if results.detections:
        for id, detects in enumerate(results.detections):
            #mpdraw.draw_detection(img,detects)
            #print(id, detects.location_data.relative_bounding_box)
            FboxC = detects.location_data.relative_bounding_box
            h , w , c = img.shape

            Fbox = int(FboxC.xmin*w), int(FboxC.ymin*h), int(FboxC.width*w),int(FboxC.height*h)

            cv2.rectangle(img,Fbox,(225,220,0),3)
            cv2.putText(img, "ACC:" + str(int(detects.score[0]*100)), (Fbox[0], Fbox[1]-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 220, 0), 3)


    cT = time.time()
    fps = 1 / (cT - pT)
    pT = cT

    cv2.putText(img, "FPS:" + str(int(fps)), (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 240), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)