import cv2
import mediapipe as mp
import time
import numpy as np
import math
import Hand_Detection_Module as HDM
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

#####################################
wwt, wht = 1280, 720
#####################################

cap = cv2.VideoCapture(0)
cap.set(3, wwt)
cap.set(4, wht)
cT, pT = 0, 0
detector = HDM.hand_detector()

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
# volume.GetMute()
#volume.GetMasterVolumeLevel()
VolRange = volume.GetVolumeRange()
# volume.SetMasterVolumeLevel(-20,None)
minVol = VolRange[0]
maxVol = VolRange[1]
print(minVol, maxVol)

vol, volBar, volper, sysvol = 0, 450, 0, 0

while True:
    success, img = cap.read()
    img = detector.find_hands(img)
    lmlist = detector.find_position(img, draw=False)
    if len(lmlist) != 0:
        # print(lmlist[4], lmlist[8])
        id1, x1, y1 = lmlist[4]
        id2, x2, y2 = lmlist[8]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        cv2.circle(img, (x1, y1), 10, (200, 200, 0), cv2.FILLED)
        cv2.circle(img, (x2, y2), 10, (200, 200, 0), cv2.FILLED)

        cv2.line(img, (x1, y1), (x2, y2), (200, 200, 0), 2)

        length = math.hypot(x2 - x1, y2 - y1)
        # print(length)
        if length < 50:
            cv2.circle(img, (cx, cy), 10, (0, 0, 200), cv2.FILLED)
        else:
            cv2.circle(img, (cx, cy), 10, (200, 200, 0), 2)

        # finger range 50-300
        # vol range -65.25 - 0.0
        vol = np.interp(length, [50, 450], [minVol, maxVol])
        sysvol = volume.GetMasterVolumeLevel()
        volBar = np.interp(length, [50,450], [450, 150])
        volper = np.interp(sysvol, [-65.25, 0.0], [0, 100])



        print(length, sysvol)
        volume.SetMasterVolumeLevel(vol, None)

    cv2.rectangle(img, (50, 150), (55, 450), (100, 255, 0), 1)
    cv2.rectangle(img, (50, int(volBar)), (55, 450), (100, 255, 0), cv2.FILLED)
    cv2.putText(img, f"{int(volper)}%", (60, 300), cv2.FONT_HERSHEY_PLAIN, 2, (100, 255, 0), 2)

    cT = time.time()
    fps = 1 / (cT - pT)
    pT = cT

    cv2.putText(img, "FPS:" + str(int(fps)), (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 240), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
