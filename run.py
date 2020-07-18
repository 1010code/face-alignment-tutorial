import cv2

import dlib

import numpy

PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"

predictor = dlib.shape_predictor(PREDICTOR_PATH)

# cascade_path="haarcascade_frontalface_default.xml"

# cascade = cv2.CascadeClassifier(cascade_path)

def get_landmarks(im):

    # rects = cascade.detectMultiScale(im, 1.3, 8)

    for (x,y,w,h) in rects:

        rect=dlib.rectangle(x,y,x+w,y+h)

        dimface = numpy.matrix([[p.x, p.y] for p in predictor(im, rect).parts()])

        im = annotate_landmarks(im, dimface)

    return im

def annotate_landmarks(im, landmarks):

    im = im.copy()

    for idx, point in enumerate(landmarks):

        pos = (point[0, 0], point[0, 1])

        cv2.putText(im, str(idx), pos,

                    fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,

                    fontScale=0.4,

                    color=(0, 0, 255))

        cv2.circle(im, pos, 3, color=(0, 255, 255))

    return im

im=cv2.imread('test1.jpg')

cv2.imwrite('output.jpg', annotate_landmarks(im))