import cv2
import mediapipe as mp
import time

class pose_detector():
    def __init__(self,mode=False,complexity =1,sm_ldmks = True, en_segm=False, sm_segm = True , detcon=0.5, trackcon=0.5):
        self.mode = mode
        self.complexity = complexity
        self.sm_ldmks = sm_ldmks
        self.en_segm = en_segm
        self.sm_segm = sm_segm
        self.detcon = detcon
        self.trackcon = trackcon

        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(mode, complexity, sm_ldmks, en_segm, sm_segm, detcon, trackcon)
        self.mpDraw = mp.solutions.drawing_utils

    def find_pose(self,img,draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        return img

    def find_pose_position(self,img , draw=True):
        lmlst=[]
        if self.results.pose_landmarks:
            for id, ldmk in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(ldmk.x * w), int(ldmk.y * h)
                lmlst.append([id,cx,cy])
                if draw:
                    cv2.circle(img, (cx, cy), 7, (100, 255, 0), cv2.FILLED)
        return lmlst


def main():
    cap=cv2.VideoCapture(0)
    cT, pT = 0, 0
    detector = pose_detector()
    while True:
        success, img = cap.read()
        img = detector.find_pose(img)
        lmlist = detector.find_pose_position(img)
        if len(lmlist)!=0:
            print(lmlist[20])


        cT = time.time()
        fps = 1 / (cT - pT)
        pT = cT

        cv2.putText(img, "FPS:" + str(int(fps)), (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 240), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()