# Image Similarity Finder

A Python tool that finds similar images by extracting and comparing visual features like color, texture, and shape.

## What it does

This project analyzes images and creates a "fingerprint" for each one based on visual characteristics. You can then find which images are most similar to any given image.

## Features

- **Extract 25 different visual features** from images including:
  - Colors (RGB averages, dominant colors)
  - Textures (using ORB and SIFT detection)
  - Shapes (edges, contours, geometric properties)
  - General statistics (brightness, contrast)

- **Find similar images** by comparing these features
- **Process batches** of images efficiently
- **Save results** to CSV format for analysis

## Requirements

- Python 3.7+
- OpenCV: `pip install opencv-python`
- NumPy: `pip install numpy`

## Quick Start

### 1. Extract features from a single image

```bash
python extract_features.py resources/image_0.jpg
```

This prints out the image name and 25 numbers representing its visual features.

### 2. Extract features from all your images

```bash
# Create a feature database
for image in resources/*.jpg; do
    python extract_features.py "$image" >> all_features.csv
done
```

### 3. Find similar images

```bash
python en626BestMach.py all_features.csv 5
```

This finds the 12 most similar images to the image on line 5 of your CSV file.

## Project Structure

```
├── src/
│   ├── feature_extractor.py    # Main feature extraction code
│   └── image_processor.py      # Image loading and conversion
├── resources/                  # Your image collection  
├── extract_features.py         # Command-line tool
├── en626BestMach.py           # Similarity matching (provided)
├── testRun.py                 # Test everything works
└── all_features.csv           # Feature database
```

## How to use the code

### Extract features programmatically

```python
from src.feature_extractor import FeatureExtractor

extractor = FeatureExtractor()

# Get all 25 features as a list
features = extractor.extractAllFeatures("path/to/image.jpg")

# Or get specific types
color_features = extractor.getColorFeatures(image)
texture_features = extractor.getTextureFeatures(image) 
shape_features = extractor.getShapeFeatures(image)
```

### Process images

```python
from src.image_processor import ImageProcessor

processor = ImageProcessor()
processor.loadImage("path/to/image.jpg")
grayscale = processor.convertToGrayscale()
```

## The 25 Features Explained

| Type | Count | What it measures |
|------|-------|------------------|
| **Color** | 8 | Average colors, dominant hues, color variety |
| **Texture** | 8 | Surface patterns, keypoint density, edge responses |
| **Shape** | 6 | Object boundaries, geometric properties |
| **Stats** | 3 | Overall brightness and contrast |

## Testing

Run the test suite to make sure everything works:

```bash
python testRun.py
```

This will test all the features and show performance benchmarks.

## Examples

**Extract features:**
```bash
$ python extract_features.py resources/image_5.jpg
image_5.jpg,125.4,118.3,112.7,85.2,45.8,128.9,2.45,0.12...
```

**Find similar images:**
```bash
$ python en626BestMach.py all_features.csv 10
*** Finding best match for image image_10.jpg ***
image_10.jpg image_45.jpg image_23.jpg image_67.jpg...
```

## How it works

1. **Feature Extraction**: Each image gets converted into 25 numbers that describe its visual properties
2. **Normalization**: All features are scaled to be comparable (provided algorithm)
3. **Similarity**: Images with similar feature numbers are considered visually similar (provided algorithm)

The similarity matching algorithm (`en626BestMach.py`) was provided as part of the assignment and handles the mathematical comparison between feature vectors.

## Troubleshooting

**Can't import cv2?**
```bash
pip install opencv-python
```

**Image not loading?**
- Check the file path is correct
- Make sure it's a supported format (JPG, PNG, etc.)

**No similar matches found?**
- Make sure you have enough images in your database
- Try with different query images

## Resources Used

### ORB 
"Feature Matching Using ORB Algorithm in Python-OpenCV." GeeksforGeeks, June 4, 2024. https://www.geeksforgeeks.org/python/feature-matching-using-orb-algorithm-in-python-opencv/.

Siromer. "Detecting and Tracking Objects with ORB Algorithm Using OpenCV." The Deep Hub, April 22, 2025. https://medium.com/thedeephub/detecting-and-tracking-objects-with-orb-using-opencv-d228f4c9054e.

"Feature Detection (SIFT, SURF, ORB) – OpenCV 3.4 with Python 3 Tutorial 25." Pysource, September 20, 2024. https://pysource.com/2018/03/21/feature-detection-sift-surf-obr-opencv-3-4-with-python-3-tutorial-25/.

### SIFT 
"SIFT Interest Point Detector Using Python - OpenCV." GeeksforGeeks, December 9, 2020. https://www.geeksforgeeks.org/sift-interest-point-detector-using-python-opencv/.

Waheed, Ahmed. "SIFT Feature Extraction using OpenCV in Python." The Python Code. Updated May 2024. https://thepythoncode.com/article/sift-feature-extraction-using-opencv-in-python.

Durga Prasad. "OpenCV Feature Matching — SIFT Algorithm (Scale Invariant Feature Transform)." Analytics Vidhya, November 28, 2023. https://medium.com/analytics-vidhya/opencv-feature-matching-sift-algorithm-scale-invariant-feature-transform-16672eafb253.

Babin, Ihor. "An Overview of SIFT." Medium, January 5, 2022. https://medium.com/@ibabin/an-overview-of-sift-69a8b42cd5da.
