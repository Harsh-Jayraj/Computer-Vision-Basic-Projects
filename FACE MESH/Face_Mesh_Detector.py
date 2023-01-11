import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)
mpFace = mp.solutions.face_mesh
FaceMesh = mpFace.FaceMesh(max_num_faces=2, refine_landmarks=True)
mpdraw = mp.solutions.drawing_utils
drawspec = mpdraw.DrawingSpec(color=(255,0,220), thickness=1, circle_radius=1)

cT, pT = 0, 0

while True:
    success, img = cap.read()

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = FaceMesh.process(imgRGB)
    # print(results.multi_face_landmarks)
    if results.multi_face_landmarks:
        for meshldmk in results.multi_face_landmarks:
            mpdraw.draw_landmarks(img, meshldmk, mpFace.FACEMESH_CONTOURS, drawspec, drawspec)

            for id, ldmk in enumerate(meshldmk.landmark):
                h, w, c = img.shape
                cx, cy = int(ldmk.x * w), int(ldmk.y * h)
                print(id,cx,cy)
            # if id == 4:
            #    cv2.circle(img, (cx, cy), 15, (255, 255, 0), cv2.FILLED)

    cT = time.time()
    fps = 1 / (cT - pT)
    pT = cT

    cv2.putText(img, "FPS:" + str(int(fps)), (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 240), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
