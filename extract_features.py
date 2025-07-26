import sys 
import os 
sys.path.append('src')

from feature_extractor import FeatureExtractor

def main():
    if len(sys.argv) != 2:
        print("usage: ./extract_features <image_filename>", file=sys.stderr)
        sys.exit(1)

    tImagePath = sys.argv[1]

    if not os.path.exists(tImagePath):
        print(f"Error: could not find image: {tImagePath}", file=sys.stderr)
        sys.exit(1)

    tExtractor = FeatureExtractor()

    try:
        tCsvLine = tExtractor.extractFeaturesForCSV(tImagePath)
        print(tCsvLine)

    except Exception as tError:
        print(f"Error Processing image: {tImagePath}, threw error {tError}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
