# extract_facial
Extract facial features such as the eyes and mouth from MTCNN's predictions and find whether eyes or mouths are closed/open.
## Installation
To use, add the extract_facial folder to your Python path. The following dependencies are required: opencv, imutils, and numpy.
## Functions
extract_facial has four functions: extractRoi, getEyeOpenClose, getEyeOpenCloseIris, and getMouthOpenClose.
## extractRoi
exctractRoi takes in the input image, the results from the MTCNN detector, and a resize factor as its arguments. It outputs a dictionary containing the resized regions of interest (ROI) of the face, left eye, right eye, and mouth, as well as a dictionary containing coordinates to bound all those regions.
### Syntax
```python 
extractedRois, extractedCoords = extract_face.extractRoi(img, detections, resize_factor)```
### Usage
```python
import cv2
from mtcnn.mtcnn import MTCNN
import extract_facial

detector = MTCNN()

img = cv2.imread('person.jpg')
results = detector.detect_faces(img)
extractedRois, extractedCoords = extract_facial.extractRoi(img, results, 250)
```
