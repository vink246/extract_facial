import time
from typing import Any, Dict, Tuple

import cv2
import imutils
import numpy as np

from .constants.mtcnn_detector import detector
from .models.roi import ROI


def get_eye_coords(
    coordinates: Tuple[int, int],
    face_width: float,
    face_height: float,
):
    x, y = coordinates
    wrec = int(face_width / 4)
    hrec = int(face_height / 8)
    ex = int(x - (wrec / 2))
    ey = int(y - (hrec / 2))
    return (ex, ey), (ex + wrec, ey + hrec)


def get_mouth_coords(
    leftm: Tuple[int, int],
    rightm: Tuple[int, int],
    nose: Tuple[int, int],
    face_width: float,
    face_height: float,
) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    wrec = rightm[0] - leftm[0] + int(face_width / 10)
    hrec = int(1.5 * (leftm[1] - nose[1] - (face_height / 10)))
    mx = leftm[0] - int(face_width / 20)
    my = nose[1] + int(face_height / 8)
    return (mx, my), (mx + wrec, my + hrec)


def extract_roi(
    img: np.ndarray,
    resize: float,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:

    roi = ROI(
        faces=[],
        left_eyes=[],
        right_eyes=[],
        mouths=[],
    )

    coords = ROI(
        faces=[],
        left_eyes=[],
        right_eyes=[],
        mouths=[],
    )

    detections = detector.detect_faces(img)

    # check to see if detections are present:
    if len(detections) > 0:
        # loop through results
        for result in detections:
            rect = result['box']
            keypoints = result['keypoints']

            # get face coordinates
            x, y, w, h = rect[0], rect[1], rect[2], rect[3]
            coords.faces.append([(x, y), (x+w, y+h)])

            try:
                # get face ROI
                roi.faces.append(
                    imutils.resize(
                        img[y:y+h, x:x+w], width=resize, inter=cv2.INTER_CUBIC,
                    )
                )

            except cv2.error:
                time.sleep(0.1)

            # initialize mouth and nose coords as 0
            mright, mleft, nose = 0, 0, 0
            for n, v in keypoints.items():

                if n == 'left_eye':
                    # get left eye coordinates and ROI
                    eye1, eye2 = get_eye_coords(v, w, h)
                    coords.left_eyes.append([eye1, eye2])
                    try:
                        roile = img[eye1[1]:eye2[1], eye1[0]:eye2[0]]
                        roile = imutils.resize(roile, width=resize, inter=cv2.INTER_CUBIC)
                        roi.left_eyes.append(roile)

                    except cv2.error:
                        time.sleep(0.1)

                if n == 'right_eye':
                    # get right eye coordinates and ROI
                    eye1, eye2 = get_eye_coords(v, w, h)
                    coords.right_eyes.append([eye1, eye2])

                    try:
                        roire = img[eye1[1]:eye2[1], eye1[0]:eye2[0]]
                        roire = imutils.resize(roire, width=resize, inter=cv2.INTER_CUBIC)
                        roi.right_eyes.append(roire)

                    except cv2.error:
                        time.sleep(0.1)

                # save nose coords for later
                if n == 'nose':
                    nose = v

                if n == 'mouth_left':
                    mleft = v

                if n == 'mouth_right':
                    mright = v

                    # get mouth coordinates and ROI
                    mouth1, mouth2 = get_mouth_coords(mleft, mright, nose, w, h)
                    coords.mouths.append([mouth1, mouth2])

                    try:
                        roim = img[mouth1[1]:mouth2[1], mouth1[0]:mouth2[0]]
                        roim = imutils.resize(roim, width=resize, inter=cv2.INTER_CUBIC)
                        roi.mouths.append(roim)

                    except cv2.error:
                        time.sleep(0.1)

    return roi.dict(), coords.dict()
