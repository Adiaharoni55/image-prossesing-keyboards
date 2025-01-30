# Keyboard Layout Matching

A Python-based computer vision project that matches corresponding keys between Mac and Windows keyboard layouts using feature detection and matching algorithms.

## Overview

This project implements an automated system to match keyboard pieces between Mac and Windows keyboards using computer vision techniques. It uses SIFT (Scale-Invariant Feature Transform) for feature detection and matching, along with various image processing techniques to improve accuracy.

## Features

- Image preprocessing and binary conversion
- Keyboard piece extraction and contour detection
- SIFT-based feature matching
- Parallel line detection for match verification
- Match reduction to eliminate duplicates
- Detailed visualization of matches
- Support for both Mac and Windows keyboard layouts

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

2. Run the main script:
```bash
python main.py
```

## How It Works

The matching process consists of several steps:

1. **Image Preprocessing**
   - Converts images to binary format
   - Applies morphological operations for noise reduction
   - Performs edge detection and cleaning

2. **Piece Extraction**
   - Identifies individual keys using contour detection
   - Crops and processes each key piece
   - Extracts relevant contour features

3. **Feature Matching**
   - Uses SIFT to detect key features
   - Performs initial matching using BFMatcher
   - Filters matches based on distance ratio

4. **Match Verification**
   - Finds parallel lines in matches
   - Calculates matching scores
   - Reduces duplicate matches
   - Verifies contour area ratios

5. **Visualization**
   - Displays matched pieces side by side
   - Shows location of matches on the original keyboard
   - Provides detailed match statistics

## Key Functions

- `preprocess_image()`: Handles image preprocessing
- `extract_pieces()`: Extracts keyboard pieces from binary images
- `find_matching_pieces()`: Core matching algorithm
- `reduce_matches()`: Eliminates duplicate matches
- `visualize_matches_detailed()`: Creates detailed visualizations

## Parameters

Key parameters that can be adjusted:

- Binary threshold values (Mac: 135, Windows: 170)
- Minimum contour area (default: 100)
- SIFT match ratio (default: 0.875)
- Score thresholds (minimum score: 4)
- Area ratio tolerances (0.55 and 2.0)

## Visualization

The program generates visualizations showing:
- Feature matches between corresponding pieces
- Location of matches on the Windows keyboard
- Match scores and statistics

## Known Limitations

- Requires good quality input images
- Sensitive to lighting conditions
- May produce some false matches in ambiguous cases
- Processing time increases with image resolution

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[Your chosen license]

## Acknowledgments

This project uses the following open-source libraries:
- OpenCV for computer vision operations
- NumPy for numerical computations
- Matplotlib for visualization