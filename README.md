# Keyboard Layout Matching

A Python-based computer vision project that matches corresponding keys between different keyboard layouts using feature detection and matching algorithms. Now supports matching between Mac and Windows keyboards, as well as different Mac keyboard variants (e.g., white and pink).

## Overview

This project implements an automated system to match keyboard pieces between different keyboard layouts using computer vision techniques. It uses SIFT (Scale-Invariant Feature Transform) for feature detection and matching, along with contour-based analysis for improved accuracy.

## Features

- Image preprocessing and binary conversion
- Intelligent keyboard piece extraction with edge cropping
- Contour-based feature analysis and filtering
- SIFT-based feature matching with adaptive scoring
- Smart match filtering with gap-based score thresholds
- Detailed visualization of matches with keyboard location mapping
- Support for multiple keyboard layout comparisons:
  - Mac to Windows keyboard mapping
  - Different Mac keyboard variants (e.g., pink to white)

## Requirements

```
- Python 3.7+
- OpenCV (cv2)
- NumPy
- Matplotlib
```

## Installation

1. Clone the repository:
```bash
git clone [your-repository-url]
cd keyboard-matching
```

2. Install required packages:
```bash
pip install opencv-python numpy matplotlib
```

## Usage

1. Place your keyboard images in the project directory:
   - `mac.jpeg`: Mac keyboard image
   - `windows.jpeg`: Windows keyboard image
   - `pink_mac.jpg`: Pink Mac keyboard image (for Mac-to-Mac comparison)

2. Run the main script:
```bash
python main.py
```

## How It Works

The matching process consists of several sophisticated steps:

1. **Image Preprocessing**
   - Converts images to binary format with adaptive thresholding
   - Applies morphological operations for noise reduction
   - Performs intelligent edge cropping

2. **Contour Processing**
   - Identifies keyboard pieces using contour detection
   - Filters nested contours to avoid duplicates
   - Extracts geometric features (area, dimensions, orientation)
   - Crops edges intelligently using the `crop_edges` function

3. **Feature Matching**
   - Uses SIFT to detect robust key features
   - Performs kNN matching with ratio test
   - Maps matches to specific contours within pieces
   - Implements contour-specific match filtering

4. **Smart Match Selection**
   - Uses a gap-based scoring system for match selection
   - Maintains top 3 different scores within specified gap
   - Limits total matches to 8 per piece
   - Supports different gap thresholds for different keyboard comparisons

5. **Visualization**
   - Shows matched pieces with feature correspondences
   - Displays location on target keyboard
   - Provides score-based match ranking

## Key Functions

- `find_contours()`: Advanced contour detection with filtering
- `crop_edges()`: Intelligent edge cropping for pieces
- `filter_contained_contours()`: Removes nested contours
- `match_contours()`: Matches contours between pieces using SIFT features
- `find_matching_pieces()`: Core matching algorithm with smart scoring
- `reduce_matches()`: Eliminates duplicate matches
- `visualize_matches_detailed()`: Creates comprehensive visualizations

## Parameters

Adjustable parameters for different scenarios:

- Binary thresholds:
  - Mac keyboard: 135
  - Windows keyboard: 170
  - Pink Mac keyboard: 100
- Contour filtering:
  - Minimum area: 100
  - Edge padding: 10 pixels
- Matching parameters:
  - SIFT ratio test: 0.875
  - Match score gap threshold: 2.5 (Mac-Windows), 1.0 (Mac-Mac)
  - Maximum matches per piece: 8
  - Maximum different scores: 3

## Visualization

The program generates detailed visualizations showing:
- Feature matches between corresponding keyboard pieces
- Location mapping on the target keyboard
- Score-based match ranking with gap thresholds
- Contour detection results

## Known Limitations

- Requires consistent lighting in input images
- Sensitive to keyboard orientation differences
- May require adjustment of gap thresholds for different keyboard pairs
- Processing time varies with image resolution and quality

## Acknowledgments

This project uses the following open-source libraries:
- OpenCV for computer vision operations
- NumPy for numerical computations
- Matplotlib for visualization