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


def getMouthCoords(leftm, rightm, nose, face_width, face_height):
    wrec = rightm[0] - leftm[0] + int(face_width / 10)
    hrec = int(1.5 * (leftm[1] - nose[1] - (face_height / 10)))
    mx = leftm[0] - int(face_width / 20)
    # mx = nose[0] - int(wrec / 2) - int(face_width / 20)
    my = nose[1] + int(face_height / 8)
    return (mx, my), (mx + wrec, my + hrec)


def getMouthOpenClose(mouth_frame):
    mouth_open = False

    # check to see if input image exists
    if mouth_frame is not None:
        # make a copy of input image/frame and rescale
        clone_mouth = mouth_frame.copy()
        clone_mouth = imutils.resize(clone_mouth, width=250, inter=cv2.INTER_CUBIC)

        # convert to grayscale
        gray_mouth_frame = cv2.cvtColor(clone_mouth, cv2.COLOR_BGR2GRAY)

        # calculate median brightness of mouth_frame
        median = np.median(gray_mouth_frame)

        # convert mouth_frame to hsv
        hsv = cv2.cvtColor(clone_mouth, cv2.COLOR_BGR2HSV)

        # define adaptive 'value' threshold
        ubv = int((median * 0.38) + 5)
        lower = np.array([0, 0, 0])
        upper = np.array([255, 255, ubv])

        # create mask on mouth frame for inside mouth
        mask = cv2.inRange(hsv, lower, upper)

        # filter and blur mask
        mask = cv2.bilateralFilter(mask, 9, 75, 75)
        mask = cv2.GaussianBlur(mask, (5, 5), cv2.BORDER_DEFAULT)

        # find contours over mask
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # filter unwanted contours
        good = []
        c = 0
        for contour in contours:
            flag = True
            for point in contour:
                if point[0][1] == 0 or point[0][0] == 0 or point[0][0] < 20:
                    flag = False
            if flag and len(contour) > 40:
                good.append(contours[c])

            c += 1

        # draw contours
        cont = cv2.drawContours(clone_mouth, good, -1, (0, 0, 255), 2)

        # check to see if appropriate contour is present
        if len(good) > 0:
            mouth_open = True

        return mouth_open, clone_mouth

    else:
        return mouth_open, None


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

                if n == 'nose':
                    nose = v

                if n == 'mouth_left':
                    mLeft = v

                if n == 'mouth_right':
                    mRight = v
                    mouth1, mouth2 = getMouthCoords(mLeft, mRight, nose, w, h)
                    cv2.rectangle(clone, mouth1, mouth2, (0, 255, 0), 2)
                    try:
                        roiM = img[mouth1[1]:mouth2[1], mouth1[0]:mouth2[0]]
                        roiM = imutils.resize(roiM, width=250, inter=cv2.INTER_CUBIC)
                        grayM = cv2.cvtColor(roiM, cv2.COLOR_BGR2GRAY)

                    except cv2.error:
                        time.sleep(0.1)

        openM, testM = getMouthOpenClose(roiM)

        if openM:
            cv2.putText(clone, 'Mouth Open', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        else:
            cv2.putText(clone, 'Mouth Closed', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        try:
            cv2.imshow('mouthTest', testM)
            cv2.imshow('mouth', roiM)
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
