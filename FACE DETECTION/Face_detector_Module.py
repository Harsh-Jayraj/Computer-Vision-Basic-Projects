import cv2
import mediapipe as mp
import time


class Face_detector():
    def __init__(self, confidence=0.5, model=0):
        self.confidence = confidence
        self.model = model

        self.mpFace = mp.solutions.face_detection
        self.face = self.mpFace.FaceDetection(self.confidence, self.model)
        self.mpdraw = mp.solutions.drawing_utils

    def find_face(self, img, Draw=True):
        bboxs = []
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.face.process(imgRGB)
        # print(results.detections)
        if self.results.detections:
            for id, detects in enumerate(self.results.detections):
                # mpdraw.draw_detection(img,detects)
                # print(id, detects.location_data.relative_bounding_box)
                FboxC = detects.location_data.relative_bounding_box
                h, w, c = img.shape

                Fbox = int(FboxC.xmin * w), int(FboxC.ymin * h), int(FboxC.width * w), int(FboxC.height * h)
                bboxs.append([id, Fbox, detects.score])
                if Draw:
                    cv2.rectangle(img, Fbox, (225, 220, 0), 3)
                    cv2.putText(img, "ACC:" + str(int(detects.score[0] * 100)), (Fbox[0], Fbox[1] - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 220, 0), 3)
        return img, bboxs

    def facybox(self, img, bbox, l=20, t=4, rt=1):
            x, y, w, h = bbox[0][1]
            x1, y1 = x + w, y + h
            print(x, y, w, h)
            cv2.rectangle(img, bbox[0][1], (225, 220, 0), rt)

            cv2.line(img, (x, y), (x + l, y), (225, 220, 0), t)
            cv2.line(img, (x, y), (x, y + l), (225, 220, 0), t)

            cv2.line(img, (x1, y), (x1 - l, y), (225, 220, 0), t)
            cv2.line(img, (x1, y), (x1, y + l), (225, 220, 0), t)

            cv2.line(img, (x, y1), (x + l, y1), (225, 220, 0), t)
            cv2.line(img, (x, y1), (x, y1 - l), (225, 220, 0), t)

            cv2.line(img, (x1, y1), (x1 - l, y1), (225, 220, 0), t)
            cv2.line(img, (x1, y1), (x1, y1 - l), (225, 220, 0), t)
            return img


def main():
    cap = cv2.VideoCapture(0)
    cT, pT = 0, 0
    detector = Face_detector()
    while True:
        success, img = cap.read()
        img, bboxs = detector.find_face(img, False)
        img = detector.facybox(img, bboxs)
        print(bboxs)

        cT = time.time()
        fps = 1 / (cT - pT)
        pT = cT

        cv2.putText(img, "FPS:" + str(int(fps)), (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 240), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
