import cv2
import numpy as np
import imutils


def getEyeOpenClose(eye_frame):
    eye_open = False

    # check to see if input image exists
    if eye_frame is not None:
        # make a copy of input image/frame and rescale
        clone_eye = eye_frame.copy()
        clone_eye = imutils.resize(clone_eye, width=250, inter=cv2.INTER_CUBIC)

        # convert to grayscale and increase contrast
        gray_eye_frame = cv2.cvtColor(clone_eye, cv2.COLOR_BGR2GRAY)
        gray_eye_frame = cv2.convertScaleAbs(gray_eye_frame, alpha=2, beta=-30)

        # calculate median brightness of eye_frame
        median = np.median(gray_eye_frame)

        # convert eye_frame to hsv
        hsv = cv2.cvtColor(clone_eye, cv2.COLOR_BGR2HSV)

        # define adaptive 'value' threshold
        ubv = int((median * 0.48) + 48)
        lower = np.array([0, 0, ubv])
        upper = np.array([255, 45, 255])

        # create mask on eye_frame for sclera
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
                if point[0][1] == 0 or point[0][0] == 0 or point[0][0] > 240:
                    flag = False
            if flag and len(contour) > 25:
                good.append(contours[c])

            c += 1

        # draw contours
        cont = cv2.drawContours(clone_eye, good, -1, (0, 0, 255), 2)

        # check to see if appropriate contour is present
        if len(good) > 0:
            eye_open = True

        return eye_open, clone_eye

    else:
        return eye_open, None
