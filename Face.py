# importer les paquets nécessaires
from imutils import face_utils
import argparse
import imutils
import dlib
import cv2
import numpy

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("FaceReco\shape_predictor_68_face_landmarks.dat")

video = cv2.VideoCapture(0)
a = 0

while True:
    a = a+1

    check, frame = video.read()


    image = frame
    image = imutils.resize(image, width=600)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    rects = detector(gray, 1)

    for (i, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        (x, y, w, h) = face_utils.rect_to_bb(rect)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 1)

        for (x, y) in shape:
            cv2.circle(image, (x, y), 1, (0, 0, 255), -1)

    cv2.imshow("Face reco", image)

    key = cv2.waitKey(1)

    if(key == ord('q')):
        break


video.release()
cv2.destroyAllWindows()