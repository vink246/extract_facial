# Functions

There are four main functions which `extract_facial` provides which allow you to
extract facial features from images

### extract_roi

extract_roi takes in the input image from `opencv-python`, and a resize factor as its arguments. 
It returns the resized regions of interest (ROI) of the face, left eye, right eye, and mouth, as well as 
coordinates to bound all those regions.

```Python
import cv2
from extract_facial import extract_roi
image = cv2.imread('person.jpg') # you can point to the filepath of any image
roi, roi_coordinates = extract_roi(image, 250)
```
