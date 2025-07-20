import sys 
import os 
sys.path.append('src')

from image_processor import ImageProcessor
from feature_extractor import FeatureExtractor 
import cv2 
import numpy as np 
import time 

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
    tInvalidPath = "resources/not_here_lol.png"
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

def testFeatureExtractor():
    print("\n" + "=" * 40)
    print("testing feature extractor")
    print("=" * 40)

    tProcessor = ImageProcessor()
    tExtractor = FeatureExtractor()

    tTestImagePath = "resources/image_0.jpg"
    print(f"testing with {tTestImagePath}")

    if not tProcessor.loadImage(tTestImagePath):
        print("failed to load image")
        return False 

    print(f"Original image shape: {tProcessor.mCurrentImage.shape}")
    tGrayImage = tProcessor.convertToGrayscale()
    print(f"Grayscale image shape: {tGrayImage.shape}")
    
    # Test ORB Features
    print("\n--- Testing ORB Features ---")
    tStartTime = time.time()
    tKeypoints, tORBDescriptors = tExtractor.extractORBFeatures(tGrayImage)
    tORBTime = time.time() - tStartTime
    
    if tORBDescriptors is not None:
        print(f"ORB Success: {len(tORBDescriptors)} descriptors extracted")
        print(f"ORB Descriptor shape: {tORBDescriptors.shape}")
        print(f"ORB Time: {tORBTime:.4f} seconds")
        if tKeypoints is not None:
            print(f"ORB Keypoints shape: {tKeypoints.shape}")
    else:
        print("ORB Failed: No descriptors extracted")
    
    # Test SIFT Features
    print("\n--- Testing SIFT Features ---")
    tStartTime = time.time()
    tKeypointsSIFT, tSIFTDescriptors = tExtractor.extractSIFTFeatures(tGrayImage)
    tSIFTTime = time.time() - tStartTime
    
    if tSIFTDescriptors is not None:
        print(f"SIFT Success: {len(tSIFTDescriptors)} descriptors extracted")
        print(f"SIFT Descriptor shape: {tSIFTDescriptors.shape}")
        print(f"SIFT Time: {tSIFTTime:.4f} seconds")
        if tKeypointsSIFT is not None:
            print(f"SIFT Keypoints shape: {tKeypointsSIFT.shape}")
    else:
        print("SIFT Failed: No descriptors extracted")
    
    # Test Color Histogram
    print("\n--- Testing Color Histogram ---")
    tStartTime = time.time()
    tColorHist = tExtractor.extractColorHistogram(tProcessor.mCurrentImage)
    tHistTime = time.time() - tStartTime
    
    if tColorHist is not None:
        print(f"Color Histogram Success: {len(tColorHist)} bins")
        print(f"Histogram shape: {tColorHist.shape}")
        print(f"Histogram sum: {np.sum(tColorHist):.6f} (should be ~1.0)")
        print(f"Histogram Time: {tHistTime:.4f} seconds")
    else:
        print("Color Histogram Failed")
    
    # Test Perceptual Hash
    print("\n--- Testing Perceptual Hash ---")
    tStartTime = time.time()
    tPHash = tExtractor.extractPerceptualHash(tGrayImage)
    tHashTime = time.time() - tStartTime
    
    if tPHash is not None:
        print(f"Perceptual Hash Success: {tPHash}")
        print(f"Hash length: {len(tPHash)} characters")
        print(f"Hash Time: {tHashTime:.4f} seconds")
    else:
        print("Perceptual Hash Failed")
    
    # Test feature summary
    print("\n--- Testing Feature Summary ---")
    tSummary = tExtractor.getFeatureSummary()
    print(f"Feature Summary: {tSummary}")
    
    return True

def testMultipleImageFeatures():
    print("\n" + "="*60)
    print("TESTING MULTIPLE IMAGE FEATURES")
    print("="*60)
    
    tProcessor = ImageProcessor()
    tExtractor = FeatureExtractor()
    
    # Test with first 10 images
    tTestImages = [f"image_{i}.jpg" for i in range(10)]
    tResults = []
    
    for tImageName in tTestImages:
        tImagePath = f"resources/{tImageName}"
        print(f"\nProcessing: {tImageName}")
        
        if not tProcessor.loadImage(tImagePath):
            print(f"  Failed to load {tImageName}")
            continue
            
        tGrayImage = tProcessor.convertToGrayscale()
        
        # Extract ORB features
        tKeypoints, tDescriptors = tExtractor.extractORBFeatures(tGrayImage)
        tORBCount = len(tDescriptors) if tDescriptors is not None else 0
        
        # Extract hash
        tHash = tExtractor.extractPerceptualHash(tGrayImage)
        
        tResults.append({
            'name': tImageName,
            'orb_features': tORBCount,
            'hash': tHash,
            'shape': tProcessor.mCurrentImage.shape
        })
        
        print(f"  ORB Features: {tORBCount}")
        print(f"  Hash: {tHash}")
        print(f"  Shape: {tProcessor.mCurrentImage.shape}")
    
    # Summary statistics
    print(f"\n--- Summary for {len(tResults)} images ---")
    if tResults:
        tORBCounts = [r['orb_features'] for r in tResults]
        print(f"ORB Features - Min: {min(tORBCounts)}, Max: {max(tORBCounts)}, Avg: {np.mean(tORBCounts):.1f}")
        
        # Check for unique hashes
        tHashes = [r['hash'] for r in tResults if r['hash']]
        tUniqueHashes = set(tHashes)
        print(f"Perceptual Hashes - Total: {len(tHashes)}, Unique: {len(tUniqueHashes)}")
        
        if len(tUniqueHashes) < len(tHashes):
            print("WARNING: Found duplicate hashes! Possible similar images detected.")
            # Show which images have duplicate hashes
            tHashCount = {}
            for tResult in tResults:
                tHash = tResult['hash']
                if tHash:
                    if tHash in tHashCount:
                        tHashCount[tHash].append(tResult['name'])
                    else:
                        tHashCount[tHash] = [tResult['name']]
            
            for tHash, tImageNames in tHashCount.items():
                if len(tImageNames) > 1:
                    print(f"  Duplicate hash {tHash}: {tImageNames}")

def testFeaturePersistence():
    print("\n" + "="*60)
    print("TESTING FEATURE PERSISTENCE")
    print("="*60)
    
    tProcessor = ImageProcessor()
    tExtractor = FeatureExtractor()
    
    # Load and extract features
    tTestImagePath = "resources/image_5.jpg"
    if not tProcessor.loadImage(tTestImagePath):
        print("Failed to load test image for persistence test")
        return False
    
    tGrayImage = tProcessor.convertToGrayscale()
    tOriginalKeypoints, tOriginalDescriptors = tExtractor.extractORBFeatures(tGrayImage)
    
    if tOriginalDescriptors is None:
        print("No features extracted for persistence test")
        return False
    
    print(f"Original features shape: {tOriginalDescriptors.shape}")
    
    # Save features
    tSaveFile = "test_features_orb.npz"
    tSaveSuccess = tExtractor.saveFeatures(tSaveFile)
    print(f"Save operation: {'Success' if tSaveSuccess else 'Failed'}")
    
    if not tSaveSuccess:
        return False
    
    # Create new extractor and load features
    tNewExtractor = FeatureExtractor()
    tLoadSuccess = tNewExtractor.loadFeatures(tSaveFile)
    print(f"Load operation: {'Success' if tLoadSuccess else 'Failed'}")
    
    if tLoadSuccess:
        print(f"Loaded features shape: {tNewExtractor.mCurrentFeatures.shape}")
        print(f"Features match: {np.array_equal(tOriginalDescriptors, tNewExtractor.mCurrentFeatures)}")
        
        tLoadedSummary = tNewExtractor.getFeatureSummary()
        print(f"Loaded feature summary: {tLoadedSummary}")
    
    # Clean up
    try:
        os.remove(tSaveFile)
        print("Test file cleaned up")
    except:
        print("Failed to clean up test file")
    
    return tLoadSuccess

def testPerformanceBenchmark():
    print("\n" + "="*60)
    print("PERFORMANCE BENCHMARK")
    print("="*60)
    
    tProcessor = ImageProcessor()
    tExtractor = FeatureExtractor()
    
    # Test with first 15 images for performance
    tTestImages = [f"image_{i}.jpg" for i in range(15)]
    tPerformanceResults = {
        'orb_times': [],
        'sift_times': [],
        'hist_times': [],
        'hash_times': []
    }
    
    print(f"Benchmarking with {len(tTestImages)} images...")
    
    for tImageName in tTestImages:
        tImagePath = f"resources/{tImageName}"
        
        if not tProcessor.loadImage(tImagePath):
            continue
            
        tGrayImage = tProcessor.convertToGrayscale()
        
        # Benchmark ORB
        tStart = time.time()
        tExtractor.extractORBFeatures(tGrayImage)
        tPerformanceResults['orb_times'].append(time.time() - tStart)
        
        # Benchmark SIFT
        tStart = time.time()
        tExtractor.extractSIFTFeatures(tGrayImage)
        tPerformanceResults['sift_times'].append(time.time() - tStart)
        
        # Benchmark Histogram
        tStart = time.time()
        tExtractor.extractColorHistogram(tProcessor.mCurrentImage)
        tPerformanceResults['hist_times'].append(time.time() - tStart)
        
        # Benchmark Hash
        tStart = time.time()
        tExtractor.extractPerceptualHash(tGrayImage)
        tPerformanceResults['hash_times'].append(time.time() - tStart)
    
    # Print results
    print(f"\nPerformance Results (averaged over {len(tPerformanceResults['orb_times'])} images):")
    for tMethod, tTimes in tPerformanceResults.items():
        if tTimes:
            tAvgTime = np.mean(tTimes)
            tStdTime = np.std(tTimes)
            print(f"{tMethod.upper().replace('_', ' ')}: {tAvgTime:.4f}s ± {tStdTime:.4f}s")

def testErrorHandling():
    print("\n" + "="*60)
    print("TESTING ERROR HANDLING")
    print("="*60)
    
    tExtractor = FeatureExtractor()
    
    # Test with None image
    print("Testing with None image...")
    tResult = tExtractor.extractORBFeatures(None)
    print(f"ORB with None: {tResult}")
    
    tResult = tExtractor.extractColorHistogram(None)
    print(f"Histogram with None: {tResult}")
    
    tResult = tExtractor.extractPerceptualHash(None)
    print(f"Hash with None: {tResult}")
    
    # Test with empty extractor
    print("\nTesting with empty extractor...")
    tSummary = tExtractor.getFeatureSummary()
    print(f"Empty summary: {tSummary}")
    
    tSaveResult = tExtractor.saveFeatures("test_empty.npz")
    print(f"Save empty features: {tSaveResult}")
    
    # Test loading non-existent file
    print("\nTesting with non-existent file...")
    tLoadResult = tExtractor.loadFeatures("non_existent_file.npz")
    print(f"Load non-existent file: {tLoadResult}")


if __name__ == "__main__":
    print("FIND SIMILAR - COMPREHENSIVE TEST SUITE")
    print("=" * 80)
    
    try:
        # Test Image Processor (original functionality)
        testImageProcessor()
        
        # Test Feature Extractor (new functionality)
        testFeatureExtractor()
        testMultipleImageFeatures()
        testFeaturePersistence()
        testPerformanceBenchmark()
        testErrorHandling()
        
        print("\n" + "="*80)
        print("ALL TESTS COMPLETED! ✓")
        print("Check output above for any failures or warnings.")
        print("="*80)
        
    except Exception as e:
        print(f"Test suite failed with exception: {e}")
        import traceback
        traceback.print_exc()
