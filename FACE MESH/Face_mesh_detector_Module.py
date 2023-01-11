import cv2
import mediapipe as mp
import time

class Face_Mesh_detector():
    def __init__(self, mode = False, max_num_faces=1, refine_landmarks=False, detcon=0.5, trackcon=0.5, color=(255, 255, 255), thickness=1, circle_radius=1):
        self.mode = mode
        self.max_num_faces = max_num_faces
        self.refine_landmarks = refine_landmarks
        self.detcon = detcon
        self.trackcon = trackcon
        self.color = color
        self.thickness =thickness
        self.circle_radius = circle_radius

        self.mpFace = mp.solutions.face_mesh
        self.FaceMesh = self.mpFace.FaceMesh(self.mode, self.max_num_faces, self.refine_landmarks, self.detcon, self.trackcon)
        self.mpdraw = mp.solutions.drawing_utils
        self.drawspec = self.mpdraw.DrawingSpec(self.color, self.thickness, self.circle_radius)

    def create_mesh(self,img,Draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.FaceMesh.process(imgRGB)
        # print(results.multi_face_landmarks)
        if self.results.multi_face_landmarks:
            for meshldmk in self.results.multi_face_landmarks:
                if Draw:
                    self.mpdraw.draw_landmarks(img, meshldmk, self.mpFace.FACEMESH_CONTOURS, self.drawspec, self.drawspec)
        return img

    def find_landmarks(self,img,num,Draw=True):
        lmlst = []
        if self.results.multi_face_landmarks:
            for meshldmk in self.results.multi_face_landmarks:
                for id, ldmk in enumerate(meshldmk.landmark):
                    h, w, c = img.shape
                    cx, cy = int(ldmk.x * w), int(ldmk.y * h)
                    lmlst.append([id, cx, cy])
                    if Draw:
                        cv2.putText(img, str(id), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 0), 1)
                    if id==num and Draw==False:
                        cv2.putText(img, str(num), (cx, cy), cv2.FONT_HERSHEY_PLAIN, 0.8, (0, 200, 0), 2)
        return img, lmlst


def main():
    cap = cv2.VideoCapture(0)
    cT, pT = 0, 0
    detector = Face_Mesh_detector()
    while True:
        success, img = cap.read()
        img = detector.create_mesh(img)
        img, lmlst = detector.find_landmarks(img, 200, False)
        print(lmlst)


        cT = time.time()
        fps = 1 / (cT - pT)
        pT = cT

        cv2.putText(img, "FPS:" + str(int(fps)), (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 240), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()