import cv2 
import numpy as np
from typing import Tuple, Optional, List, Dict 
import hashlib 

class FeatureExtractor:
    def __init__(self):
        self.mCurrentFeatures = None
        self.mFeatureType = None

        self.mORBDetector = cv2.ORB_create(nfeatures=500)
        self.mSIFTDetector = cv2.SIFT_create()

    def extractORBFeatures(self, aImage: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if aImage is None:
            return None, None 

        if len(aImage.shape) == 3:
            tGrayImage = cv2.cvtColor(aImage, cv2.COLOR_BGR2GRAY)
        else:
            tGrayImage = aImage 

        tKeypoints, tDescriptors = self.mORBDetector.detectAndCompute(tGrayImage, None)

        if tKeypoints:
            tKeypointArray = np.array([[kp.pt[0], kp.pt[1], kp.angle, kp.response]
                                       for kp in tKeypoints])
        else:
            tKeypointArray = None 

        self.mCurrentFeatures = tDescriptors 
        self.mFeatureType = "ORB"

        return tKeypointArray, tDescriptors

    def extractSIFTFeatures(self, aImage: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if aImage is None:
            return None, None 

        if len(aImage.shape) == 3:
            tGrayImage = cv2.cvtColor(aImage, cv2.COLOR_BGR2GRAY)
        else:
            tGrayImage = aImage 

        tKeypoints, tDescriptors = self.mSIFTDetector.detectAndCompute(tGrayImage, None)

        if tKeypoints:
            tKeypointArray = np.array([[kp.pt[0], kp.pt[1], kp.angle, kp.response, kp.size]
                                       for kp in tKeypoints])
        else:
            tKeypointArray =  None 

        self.mCurrentFeatures = tDescriptors
        self.mFeatureType = "SIFT"

        return tKeypointArray, tDescriptors

    def extractColorHistogram(self, aImage: np.ndarray, aBins: int = 256) -> Optional[np.ndarray]:
        if aImage is None:
            return None 

        if len(aImage.shape) == 3:
            tHistograms = []
            for i in range(aImage.shape[2]):
                tHist = cv2.calcHist([aImage], [i], None, [aBins], [0, 256])
                tHistograms.append(tHist.flatten())
            tFeatureVector = np.concatenate(tHistograms)
        else:
            #gray scale 
            tHist = cv2.calcHist([aImage], [0], None, [aBins], [0, 256])
            tFeatureVector = tHist.flatten()

        #dont forget to Normalize!
        tFeatureVector = tFeatureVector / (tFeatureVector.sum() + 1e-7)

        self.mCurrentFeatures = tFeatureVector
        self.mFeatureType = "COLOR_HISTOGRAM"

        return tFeatureVector

    def extractPerceptualHash(self, aImage:np.ndarray, aHashSize:int = 8) -> Optional[str]:
        if aImage is None:
            return None 

        if len(aImage.shape) == 3:
            tGrayImage = cv2.cvtColor(aImage, cv2.COLOR_BGR2GRAY)
        else:
            tGrayImage = aImage 
        #resize to hash size
        tResized = cv2.resize(tGrayImage, (aHashSize, aHashSize), interpolation=cv2.INTER_CUBIC)

        #apply discrete cosine transform
        tDCT = cv2.dct(np.float32(tResized))

        #extrac top left 8x8 for low frequencies 
        tDCTLowFreq = tDCT[:aHashSize//2, :aHashSize//2]

        tMedian = np.median(tDCTLowFreq)

        tBinaryHash = tDCTLowFreq > tMedian

        #convert to hex string
        tHashValue = 0
        for i, row, in enumerate(tBinaryHash.flatten()):
            if row:
                tHashValue |= (1 << i) #if this cell is true set the i'th bit to 1, :]

        tHexHash = format(tHashValue, 'x').zfill(aHashSize)

        self.mCurrentFeatures = tHexHash
        self.mFeatureType = "PERCEPTUAL_HASH"

        return tHexHash

    def getFeatureSummary(self) -> Dict[str, any]:
        if self.mCurrentFeatures is None: 
            return {"type": None, "count": 0, "shape": None}

        #common stuff
        tSummary =  {
            "type": self.mFeatureType,
            "shape": self.mCurrentFeatures.shape if hasattr(self.mCurrentFeatures, 'shape') else None,
            "data_type": type(self.mCurrentFeatures).__name__ 
        }
        #feature specific
        if self.mFeatureType in ["ORB", "SIFT"]:
            tSummary["descriptor_count"] = len(self.mCurrentFeatures) if self.mCurrentFeatures is not None else 0
            tSummary["descriptor_size"] = self.mCurrentFeatures.shape[1] if self.mCurrentFeatures is not None else 0
        elif self.mFeatureType == "COLOR_HISTOGRAM":
            tSummary["histogram_bins"] = len(self.mCurrentFeatures)
        elif self.mFeatureType == "PERCEPTUAL_HASH":
            tSummary["hash_length"] = len(self.mCurrentFeatures)
            
        return tSummary

    def saveFeatures(self, aFilePath: str) -> bool:
        if self.mCurrentFeatures is None:
            return False 
    
        try:
            if self.mFeatureType == "PERCEPTUAL_HASH":
                with open(aFilePath, 'w') as f:
                    f.write(f"{self.mFeatureType}\n{self.mCurrentFeatures}")
            else:
                np.savez(aFilePath, 
                        features = self.mCurrentFeatures,
                        feature_type = self.mFeatureType)
            return True 
        except Exception as tError:
            print(f"error saving features: {tError}")
            return False 

    def loadFeatures(self, aFilePath: str) -> bool:
        try:
            if aFilePath.endswith(".txt"):
                with open(aFilePath, 'r') as f:
                    tLines = f.readlines()
                    self.mFeatureType = tLines[0].strip()
                    self.mCurrentFeatures = tLines[1].strip()
            else:
                tData = np.load(aFilePath)
                self.mCurrentFeatures = tData['features']
                self.mFeatureType = str(tData['feature_type'])
            return True 
        except Exception as tError:
            print(f"Error loading features: {tError}")
            return False


