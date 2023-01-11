import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)
mpHands =mp.solutions.hands
hands = mpHands.Hands()
drawHand = mp.solutions.drawing_utils

cT , pT =0,0

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    #print(results.multi_hand_landmarks)
    if results.multi_hand_landmarks:
        for handldmk in results.multi_hand_landmarks:
            for id, ldmk in enumerate(handldmk.landmark):
                h, w, c = img.shape
                cx, cy = int(ldmk.x*w), int(ldmk.y*h)

                if id==4:
                    cv2.circle(img,(cx,cy),15,(255,255,0), cv2.FILLED)

            drawHand.draw_landmarks(img, handldmk, mpHands.HAND_CONNECTIONS)

    cT=time.time()
    fps = 1/(cT-pT)
    pT=cT

    cv2.putText(img, "FPS:"+str(int(fps)), (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 240), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)