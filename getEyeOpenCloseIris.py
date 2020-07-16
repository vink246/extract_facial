import cv2
import imutils


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
