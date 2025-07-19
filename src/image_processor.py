import cv2
import numpy as np 
from typing import Tuple, Optional, List 
import os 

class ImageProcessor:
    def __init__(self):
        self.mCurrentImage = None 

    def loadImage(self, aImagePath: str) -> bool:
        self.mCurrentImage = cv2.imread(aImagePath)
        return self.mCurrentImage is not None 

    def convertToGrayscale(self) -> np.ndarray: 
        if self.mCurrentImage is None:
            return None 

        self.mCurrentImage = cv2.cvtColor(self.mCurrentImage, cv2.COLOR_BGR2GRAY)
        return self.mCurrentImage

    
