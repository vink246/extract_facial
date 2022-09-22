# extract_facial
Extract facial features such as the eyes and mouth from MTCNN's predictions and find whether eyes or mouths are closed/open.

## Installation

extract_facial can be installed using the pip package manager:
```
pip install extract_facial
```

## Basic Usage

```python
import cv2
import extract_facial

img = cv2.imread('person.jpg')
results = detector.detect_faces(img)
extractedRois, extractedCoords = extract_facial.extractRoi(img, 250)
```

## Contributing and Local Development

Please check the [CONTRIBUTING](/CONTRIBUTING.md) guidelines for information 
on how to contribute to extract_facial.
