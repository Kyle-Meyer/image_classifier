import sys 
import os 
sys.path.append('src')

from image_processor import ImageProcessor
import cv2 

def testImageProcessor():

    tProcessor = ImageProcessor()

    print("Testing load image ")
    tTestImagePath = "resources/image_0.jpg"
    tLoadSuccess = tProcessor.loadImage(tTestImagePath);
    print(f"Load Status: {tLoadSuccess}")

    if tLoadSuccess:
        print(f"Original Image Shape, {tProcessor.mCurrentImage.shape}")
        print(f"Original Image Dtype, {tProcessor.mCurrentImage.dtype}")

    print("Testing grayscale")
    if tLoadSuccess:
        tGrayScale = tProcessor.convertToGrayscale()
        if tGrayScale is not None:
            print(f"Grayscale image shape: {tGrayScale.shape}")
            print(f"Grayscale image Dtype: {tGrayScale.dtype}")

        cv2.imwrite("test_grayscale.jpg", tGrayScale)
        print("saved grayscale ")

    else:
        print("bad grayscale, failed")

    print("testing bad image")
    tInvalidPath = "resources/faggot.png"
    tInvalidLoad = tProcessor.loadImage(tInvalidPath)
    print(f"invalid image path reuslt: {tInvalidLoad}")

    print("testing conversion to grayscale when no image")
    tEmptyProcessor = ImageProcessor()
    tEmptyResult = tEmptyProcessor.convertToGrayscale()
    print(f"Empty Conversion Result: {tEmptyResult}")

    print("test multiple images")
    tTestImages = ["image_1.jpg", "image_2.jpg", "image_3.jpg"]

    for tImageName in tTestImages:
        tImagePath = f"resources/{tImageName}"
        if tProcessor.loadImage(tImagePath):
            print(f"loaded, {tImageName}: {tProcessor.mCurrentImage.shape}")
            tGray = tProcessor.convertToGrayscale()
            print(f"Converted {tImageName} -> to grayscale: {tGray.shape}")
        else:
            print(f"failed to load image: {tImageName}")

if __name__ == "__main__":
    testImageProcessor()
