# Keyboard Layout Matching

A Python computer-vision project that finds where pieces of one keyboard belong on a different keyboard layout, using classical feature detection and matching (no deep learning). It supports matching between Mac and Windows keyboards, as well as between Mac keyboard variants (e.g., pink and white).

## Demo

**Mac ↔ Windows**

![Mac to Windows keyboard matching](images/mac_windows_demo.gif)

**Pink Mac ↔ White Mac**

![Pink Mac to white Mac keyboard matching](images/mac_mac_demo.gif)

## Overview

Given a source keyboard and a target keyboard, the program extracts individual keyboard pieces, describes them with SIFT features and contour geometry, and locates each piece's matching region on the target keyboard. A gap-based scoring system keeps only the strongest, most distinct matches, and the results are visualized with feature correspondences and their location on the target layout.

## Features

- Image preprocessing with binary thresholding and morphological noise reduction
- Keyboard piece extraction with intelligent edge cropping
- Contour-based feature analysis and nested-contour filtering
- SIFT feature matching with kNN and ratio test
- Gap-based match scoring and filtering
- Detailed visualizations mapping each match to its location on the target keyboard
- Multiple layout comparisons:
  - Mac → Windows
  - Mac variants (pink → white)

## Requirements

- Python 3.7+
- OpenCV (`opencv-python`)
- NumPy
- Matplotlib

## Installation

```bash
git clone https://github.com/Adiaharoni55/image-prossesing-keyboards.git
cd image-prossesing-keyboards
pip install opencv-python numpy matplotlib
```

## Usage

1. Place the keyboard images in the project directory:
   - `mac.jpeg` — Mac keyboard
   - `windows.jpeg` — Windows keyboard
   - `pink_mac.jpg` — pink Mac keyboard (for the Mac-to-Mac comparison)

2. Run:

```bash
python main.py
```

## How It Works

1. **Image preprocessing**
   - Converts images to binary with per-keyboard thresholds
   - Applies morphological operations to reduce noise
   - Crops piece edges to remove border artifacts

2. **Contour processing**
   - Detects keyboard pieces with contour detection
   - Removes nested contours to avoid duplicates
   - Extracts geometric features (area, dimensions, orientation)

3. **Feature matching**
   - Detects SIFT keypoints and descriptors
   - Performs kNN matching with a ratio test
   - Maps matches to specific contours within each piece and filters them per contour

4. **Match selection**
   - Scores candidate matches and keeps the top 3 distinct scores within a gap threshold
   - Limits results to at most 8 matches per piece
   - Uses different gap thresholds for different keyboard pairs

5. **Visualization**
   - Draws feature correspondences between matched pieces
   - Highlights the matched location on the target keyboard
   - Ranks matches by score

## Key Functions

| Function | Purpose |
|---|---|
| `find_contours()` | Contour detection with filtering |
| `crop_edges()` | Crops piece edges to remove border noise |
| `filter_contained_contours()` | Removes nested contours |
| `match_contours()` | Matches contours between pieces using SIFT features |
| `find_matching_pieces()` | Core matching algorithm with score-based selection |
| `reduce_matches()` | Removes duplicate matches |
| `visualize_matches_detailed()` | Generates the match visualizations |

## Parameters

| Category | Parameter | Value |
|---|---|---|
| Binary threshold | Mac keyboard | 135 |
| | Windows keyboard | 170 |
| | Pink Mac keyboard | 100 |
| Contour filtering | Minimum area | 100 |
| | Edge padding | 10 px |
| Matching | SIFT ratio test | 0.875 |
| | Score gap threshold | 2.5 (Mac–Windows), 1.0 (Mac–Mac) |
| | Max matches per piece | 8 |
| | Max distinct scores | 3 |

## Known Limitations

- Requires consistent lighting across input images
- Sensitive to differences in keyboard orientation
- Gap thresholds may need tuning for new keyboard pairs
- Processing time grows with image resolution

## Libraries Used

- [OpenCV](https://opencv.org/) — image processing and SIFT
- [NumPy](https://numpy.org/) — numerical computation
- [Matplotlib](https://matplotlib.org/) — visualization
