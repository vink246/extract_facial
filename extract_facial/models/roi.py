from typing import Any, List
from pydantic import BaseModel

class ROI(BaseModel):
    faces: List[Any]
    left_eyes: List[Any]
    right_eyes: List[Any]
    mouths: List[Any]
