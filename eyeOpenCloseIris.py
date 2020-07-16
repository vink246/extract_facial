import cv2
from mtcnn.mtcnn import MTCNN
import numpy as np
import time
import imutils


def getEyeCoords(co_ord, face_width, face_height):
    x = co_ord[0]
    y = co_ord[1]
    wrec = int(face_width / 4)
    hrec = int(face_height / 8)
    ex = int(x - (wrec / 2))
    ey = int(y - (hrec / 2))
    return (ex, ey), (ex + wrec, ey + hrec)


def getEyeOpenCloseIris(eye_frame, eyedecp1=45, eyedecp2=25):
    eye_open = False

    # check to see if input image exists
    if eye_frame is not None:
        # make a copy of input image/frame and rescale
        clone_eye = eye_frame.copy()
        clone_eye = imutils.resize(clone_eye, width=250, inter=cv2.INTER_CUBIC)

        # convert to grayscale, increase contrast and blur
        gray_eye_frame = cv2.cvtColor(clone_eye, cv2.COLOR_BGR2GRAY)
        gray_eye_frame = cv2.medianBlur(gray_eye_frame, 5)
        gray_eye_frame = cv2.convertScaleAbs(gray_eye_frame, alpha=3, beta=-200)
        gray_eye_frame = cv2.GaussianBlur(gray_eye_frame, (5, 5), cv2.BORDER_DEFAULT)

        # detect circles to find iris
        circlesle = cv2.HoughCircles(gray_eye_frame, cv2.HOUGH_GRADIENT, 1, 100, param1=eyedecp1, param2=eyedecp2,
                                     minRadius=20, maxRadius=80)

        # check to see if circle is detected
        if circlesle is not None:
            eye_open = True
            for x, y, r in circlesle[0]:
                cv2.circle(clone_eye, (x, y), r, (0, 0, 255), 1)

        return eye_open, clone_eye

    else:
        return eye_open, None


detector = MTCNN()

roiM = None
roiRE = None
roiLE = None
grayM = None
grayRE = None
grayLE = None
openLE = False
openRE = False

image_width = 500

cap = cv2.VideoCapture(0)

while 1:
    ret, img = cap.read()
    if ret:
        img = imutils.resize(img, width=image_width)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clone = img.copy()

        results = detector.detect_faces(img)

        for result in results:
            rect = result['box']
            keyPoints = result['keypoints']

            x, y, w, h = rect[0], rect[1], rect[2], rect[3]
            # cv2.circle(clone, (x, y), 4, (0, 0, 255), -1)
            # cv2.circle(clone, (x+w, y+h), 4, (0, 255, 0), -1)
            cv2.rectangle(clone, (x, y), (x + w, y + h), (255, 0, 0), 2)

            mRight, mLeft, nose = 0, 0, 0
            for n, v in keyPoints.items():
                cv2.circle(clone, v, 2, (0, 0, 255), -1)

                if n == 'left_eye':
                    eye1, eye2 = getEyeCoords(v, w, h)
                    cv2.rectangle(clone, eye1, eye2, (0, 255, 0), 2)
                    roiLE = img[eye1[1]:eye2[1], eye1[0]:eye2[0]]
                    roiLE = imutils.resize(roiLE, width=250, inter=cv2.INTER_CUBIC)
                    grayLE = cv2.cvtColor(roiLE, cv2.COLOR_BGR2GRAY)

                if n == 'right_eye':
                    eye1, eye2 = getEyeCoords(v, w, h)
                    cv2.rectangle(clone, eye1, eye2, (0, 255, 0), 2)
                    roiRE = img[eye1[1]:eye2[1], eye1[0]:eye2[0]]
                    roiRE = imutils.resize(roiRE, width=250, inter=cv2.INTER_CUBIC)
                    grayRE = cv2.cvtColor(roiRE, cv2.COLOR_BGR2GRAY)

        openLE, contLE = getEyeOpenCloseIris(roiLE)
        openRE, contRE = getEyeOpenCloseIris(roiRE)

        if openLE or openRE:
            cv2.putText(clone, 'Eyes Open', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        else:
            cv2.putText(clone, 'Eyes Closed', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        try:
            cv2.imshow('LEContours', contLE)
            cv2.imshow('REContours', contRE)
            # cv2.imshow('mouth', roiM)
            cv2.imshow('leftEye', roiLE)
            cv2.imshow('rightEye', roiRE)

        except cv2.error:
            time.sleep(0.1)

        cv2.imshow('original', img)
        cv2.imshow('out', clone)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
cap.release()
cv2.destroyAllWindows()
