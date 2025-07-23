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

    def getColorFeatures(self, aImage: np.ndarray) -> List[float]:
        if aImage is None:
            return [0.0] * 8

        #our buildable return 
        tFeatures = []

        #lets first get the mean of R, G, B 
        if len(aImage.shape) == 3:
            tMeanB, tMeanG, tMeanR = cv2.mean(aImage)[:3] #ignore the alpha channel 
            tFeatures.extend([tMeanR, tMeanG, tMeanB])
        else:
            tMeanGray = cv2.mean(aImage)[0] # must just be a gray image at this point, so take the first channel 
            tFeatures.extend([tMeanGray, tMeanGray, tMeanGray])

        # Now get dominant color ala HSV style 
        if len(aImage.shape) == 3:
            tHsvImage = cv2.cvtColor(aImage, cv2.COLOR_BGR2HSV)
            #get histograms 
            #calcHist(images, chanels, mask, histSize, ranges)
            tHHist = cv2.calcHist([tHsvImage], [0], None, [32], [0, 180]) #Hue is in range 0-180 on the 0 channel 
            tSHist = cv2.calcHist([tHsvImage], [1], None, [32], [0, 256]) #saturation is in range 0-256 on 1 channel 
            tVHist = cv2.calcHist([tHsvImage], [2], None, [32], [0, 256]) #value is in range 0-256 on 2 channel

            #get the largest elements in these arrays 
            tDomHue = np.argmax(tHHist) * (180.0 / 32) #convert this back to the hue range
            tDomSat = np.argmax(tSHist) * (256.0 / 32) #convert back to the saturation range 
            tDomVal = np.argmax(tVHist) * (256.0 / 32) #convert back to the value range 

            tFeatures.extend([tDomHue, tDomSat, tDomVal])

        else:
            #grayscale 
            tFeatures.extend([0.0, 0.0, cv2.mean(aImage)[0]])

        #lastly get color histogram stats 
        tColorHist = self.extractColorHistogram(aImage, aBins=32)

        if tColorHist is not None:
            tHistEntropy = -np.sum(tColorHist * np.log(tColorHist + 1e-10))

            tHistStd = np.std(tColorHist)

            tFeatures.extend([tHistEntropy, tHistStd])
        else:
            tFeatures.extend([0.0, 0.0])

        return tFeatures

    def getTextureFeatures(self, aImage: np.ndarray) -> List[float]:
        if aImage is None:
            return [0.0] * 8

        tFeatures = []

        if len(aImage.shape) == 3:
            tGrayImage = cv2.cvtColor(aImage, cv2.COLOR_BGR2GRAY)
        else:
            tGrayImage = aImage 

        #ORB based texture features 
        try:
            tOrbKeypoints, tOrbDescriptors = self.extractORBFeatures(aImage)

            if tOrbKeypoints is not None and len(tOrbKeypoints) > 0:
                #key point density per 1000 pixels
                tImageArea = tGrayImage.shape[0] * tGrayImage.shape[1]
                tKeypointDensity = (len(tOrbKeypoints) / tImageArea) * 1000

                #mean keypoint strength of features
                tMeanResponse = np.mean(tOrbKeypoints[:, 3]) #4th column is response
                
                #response standard deviation
                tResponseStd = np.std(tOrbKeypoints[:, 3])

                tMeanAngleVariation = np.mean(np.abs(tOrbKeypoints[:, 2])) #3rd column is angle 

                tFeatures.extend([tKeypointDensity, tMeanResponse, tResponseStd, tMeanAngleVariation])
            else:
                tFeatures.extend([0.0, 0.0, 0.0, 0.0])

        except Exception as tError:
            print(f"Error extracting ORB features {tError}")
            tFeatures.extend([0.0, 0.0, 0.0, 0.0])

        #SIFT features 
        try: 
            tSiftKeypoints, tSiftDescriptors = self.extractSIFTFeatures(aImage)

            if tSiftKeypoints is not None and len(tSiftKeypoints) > 0:
                #density again by 1000 pixels 
                tSiftDensity = (len(tSiftKeypoints) / tImageArea) * 1000

                tSiftMeanResponse = np.mean(tSiftKeypoints[:, 3])

                tMeanKeypointSize = np.mean(tSiftKeypoints[:, 4]) # 5th column is size 

                if tSiftDescriptors is not None:
                    #mean of all descriptor valus
                    tDescriptorMean = np.mean(tSiftDescriptors)
                else:
                    tDescriptorMean = 0.0 

                tFeatures.extend([tSiftDensity, tSiftMeanResponse, tMeanKeypointSize, tDescriptorMean])
            else:
                #no sift keypoints found 
                tFeatures.extend([0.0, 0.0, 0.0, 0.0])

        except Exception as tError:
            print(f"error extracting SIFT features: {tError}")
            tFeatures.extend([0.0, 0.0, 0.0, 0.0])

        return tFeatures

    def getShapeFeatures(self, aImage: np.ndarray) -> List[float]:
        if aImage is None:
            return [0.0] * 6
        tFeatures = []

        if len(aImage.shape) == 3:
            tGrayImage = cv2.cvtColor(aImage, cv2.COLOR_BGR2GRAY)
        else:
            tGrayImage = aImage 

        #get edge density 
        try: 
            tEdges = cv2.Canny(tGrayImage, 50, 150) #built in canny edge detection provided by cv2 
            tTotalPixels = tGrayImage.shape[0] * tGrayImage.shape[1]
            tEdgePixels = np.sum(tEdges > 0)
            tEdgeDensity = tEdgePixels / tTotalPixels
            tFeatures.append(tEdgeDensity)
        except Exception as tError:
            print(f"Error calculating edge density: {tError}")
            tFeatures.append(0.0) 

        #contour based shape features 
        try:
            tContours, _ = cv2.findContours(tEdges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if tContours:
                # get the largest contour 
                tLargestContours = max(tContours, key=cv2.contourArea)

                #normalized contour area 
                tContourArea = cv2.contourArea(tLargestContours)
                tNormalizedArea = tContourArea / tTotalPixels

                #normalized perimeter
                tPerimeter = cv2.arcLength(tLargestContours, True)
                tNormalizedPerimeter = tPerimeter / (tGrayImage.shape[0] + tGrayImage.shape[1])

                #solidity 
                tHull = cv2.convexHull(tLargestContours)
                tHullArea = cv2.contourArea(tHull)
                tSolidity = tContourArea / tHullArea if tHullArea > 0 else 0.0 
                
                tFeatures.extend([tNormalizedArea, tNormalizedPerimeter, tSolidity])
            else:
                tFeatures.extend([0.0, 0.0, 0.0])

        except Exception as tError:
            print(f"Error Calculating features: {tError}")
            tFeatures.extend([0.0, 0.0, 0.0])

        #hu moments
        try:
            tMoments = cv2.moments(tGrayImage)

            tHuMoments = cv2.HuMoments(tMoments).flatten()

            tHu1 = -np.sign(tHuMoments[0]) * np.log10(np.abs(tHuMoments[0]) + 1e-10)
            tHu2 = -np.sign(tHuMoments[1]) * np.log10(np.abs(tHuMoments[1]) + 1e-10)

            tFeatures.extend([tHu1, tHu2])

        except Exception as tError:
            print(f"Error calculating hu moments: {tError}")
            tFeatures.extend([0.0, 0.0])

        return tFeatures

    def getStatisticalFeatures(self, aImage: np.ndarray) -> List[float]:
        if aImage is None:
            return [0.0] * 3

        tFeatures = [] 

        if len(aImage.shape) == 3:
            tGrayImage = cv2.cvtColor(aImage, cv2.COLOR_BGR2GRAY)
        else:
            tGrayImage = aImage 

        try:
            tMeanBrightness = np.mean(tGrayImage)

            tStdBrightness = np.std(tGrayImage)

            tHist = cv2.calcHist([tGrayImage], [0], None, [256], [0, 256])
            tHistNormalized = tHist / (tHist.sum() + 1e-10)
            tEntropy = -np.sum(tHistNormalized * np.log2(tHistNormalized + 1e-10))

            tFeatures.extend([tMeanBrightness, tStdBrightness, tEntropy])

        except Exception as tError:
            print(f"Error calculating stat features: {tError}")
            tFeatures.extend([0.0, 0.0, 0.0])

        return tFeatures

    def extractAllFeatures(self, aImagePath: str) -> List[float]:
        tImage = cv2.imread(aImagePath)
        if tImage is None:
            print(f"Error: could not load image {aImagePath}")
            return [0.0] * 25 

        tAllFeatures = []

        tColorFeatures = self.getColorFeatures(tImage)
        tTextureFeatures = self.getTextureFeatures(tImage)
        tShapeFeatures = self.getShapeFeatures(tImage)
        tStatFeatures = self.getStatisticalFeatures(tImage)
    
        tAllFeatures.extend(tColorFeatures)     #8 features
        tAllFeatures.extend(tTextureFeatures)   #8 features 
        tAllFeatures.extend(tShapeFeatures)     #6 features 
        tAllFeatures.extend(tStatFeatures)      #3 features 

        return tAllFeatures

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

    def extractFeaturesForCSV(self, aImagePath: str) -> str:
        import os

        tImageName = os.path.basename(aImagePath)
        tFeatures = self.extractAllFeatures(aImagePath)

        tFeatureStrings = [str(f) for f in tFeatures]
        tCsvLine = f"{tImageName},{','.join(tFeatureStrings)}"

        return tCsvLine
