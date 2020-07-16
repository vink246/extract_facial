import cv2
import numpy as np
import imutils


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
