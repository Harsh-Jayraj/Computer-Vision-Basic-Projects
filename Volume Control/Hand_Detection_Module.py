import cv2
import mediapipe as mp
import time


class hand_detector():
    def __init__(self, mode=False, numhands=2, detcon=0.5, trackcon=0.5):
        self.mode = mode
        self.numhands = numhands
        self.detcon = detcon
        self.trackcon = trackcon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.numhands, 1, self.detcon, self.trackcon)
        self.drawHand = mp.solutions.drawing_utils

    def find_hands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)
        if self.results.multi_hand_landmarks:
            for handldmk in self.results.multi_hand_landmarks:
                if draw:
                    self.drawHand.draw_landmarks(img, handldmk, self.mpHands.HAND_CONNECTIONS)
        return img

    def find_position(self, img, handno=0, draw=True):
        lmlist = []
        if self.results.multi_hand_landmarks:
            myhand = self.results.multi_hand_landmarks[handno]
            for id, ldmk in enumerate(myhand.landmark):
                h, w, c = img.shape
                cx, cy = int(ldmk.x * w), int(ldmk.y * h)
                lmlist.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 10, (255, 255, 0), cv2.FILLED)
        return lmlist


def main():
    cap = cv2.VideoCapture(0)
    cT, pT = 0, 0
    detector = hand_detector()
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


if __name__ == "__main__":
    main()
