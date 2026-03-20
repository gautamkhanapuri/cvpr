# CS 5330 Project 4: Camera Calibration and Augmented Reality

**Author:** Gautam Ajey Khanapuri  
**Date:** 10 March 2026  
**Operating System:** macOS (Apple Silicon M1)  
**IDE/Editor:** CLion with Makefile. Run on Terminal
**Compiler:** clang++ (Apple Clang)

## Project Overview

Implementation of camera calibration system and augmented reality application that projects virtual 3D objects onto a checkerboard target in real-time video streams.

## Compilation
```bash
make clean all
```

Creates executable: `caar`

## Required Files

- `checkerboard` - 9×6 internal corners target (provided)
- `calibration_details.csv` - You need to create a csv file and provide it to the binary while running it from the command line.

## Running the System

### Basic Usage:
```bash
./caar <calibration_file.csv>
```

Example: `./caar calibration.csv`

**Note:** Calibration file must exist (can be empty). Program does not create it. You need to create it and pass it as the first (and only) command line argument.

## Operating Modes

### Mode 1: CALIBRATION (Default)

**Purpose:** Collect images to calibrate camera intrinsic parameters.

**Workflow:**
1. Display checkerboard to camera (on screen or printed)
2. Move checkerboard to different positions/angles
3. When corners detected (green overlay), press 's' to save frame
4. Repeat for at least 5 different positions
5. After 5 images, automatic calibration occurs
6. Review calibration matrix and reprojection error on display

**Best Practices for Calibration:**
- Use good lighting
- Vary checkerboard distance (near and far)
- Vary angles (tilt, rotate)
- Cover all regions of image
- Aim for reprojection error < 1.0 pixel

### Mode 2: AR PROJECTION

**Purpose:** Project virtual 3D objects onto checkerboard in real-time.

**Requirements:** Must have valid calibration (completed calibration mode first)

**Controls:**
- Press 'r' to enter AR mode
- Checkerboard must be visible (corners detected)
- System displays 3D axes and virtual objects
- Move checkerboard or camera - objects track correctly

## Keyboard Controls

### Calibration Mode:
- `s` - Save current frame for calibration (if corners detected)
- `r` - Switch to AR projection mode (requires completed calibration)
- `q` - Quit and save calibration to file

### AR Mode:
- `c` - Return to calibration mode
- `0` - Hide axes (axes only)
- `1` - Toggle display Object 1 (DNA Double Helix)
- `2` - Toggle display Object 2 (3D Pyramid)
- `3` - Toggle display Object 3 (Spiral Staircase)
- `s` - Save frame
- `q` - Quit program

**Note:** Multiple objects can be displayed simultaneously by pressing multiple number keys.

## Display Information

### Calibration Mode:
- Current mode
- Calibration status
- Number of images collected
- Calibration matrix
- Distortion coefficients
- Reprojection error (after calibration)

### AR Mode:
- Current mode
- Active objects
- Calibration matrix (K)
- Distortion coefficients
- Real-time rotation matrix (R)
- Real-time translation vector (t)

## Virtual Objects

**Object 1: DNA Double Helix**
- Two intertwined helical strands
- Used different colours.
- Rotates with time.

**Object 2: Spiral Staircase**
- These were practice objects
- Helical path rising from board
- 12 steps with inner/outer rails
- Radial connections
- Demonstrates continuous curves

**Object 2: 3D Pyramid**
- Practice object
- Square base on checkerboard
- Apex floating 4 units above
- Internal support structure (asymmetric)
- Shows geometric primitives

All objects float 1-4 units above the checkerboard Z-plane and maintain correct orientation as camera or target moves.

## Technical Details

**Checkerboard Target:**
- 9 columns × 6 rows of internal corners (54 total points)
- World coordinates: X ∈ [0, 8], Y ∈ [0, -5], Z = 0
- Origin at top-left corner
- 1 unit = 1 square width

**Calibration:**
- Uses `cv::calibrateCamera()` with fixed aspect ratio
- Estimates focal lengths (fx, fy), principal point (cx, cy)
- Optional radial distortion coefficients
- Automatic recalibration on each new saved image (after 5 minimum)

**Real-Time Tracking:**
- `cv::findChessboardCorners()` detects target each frame
- `cv::solvePNP()` computes camera pose (R, t)
- `cv::projectPoints()` maps 3D virtual objects to 2D image
- 30 FPS performance

## Task 7: Robust Features
### Basic Usage:
```bash
./task7_orb
```

```bash
.\task_ff
```

### Keyboard controls:
- `s` - Save frame
- `q` - Quit program

## Files Submitted

- main.cpp
- caar.cpp, caar.h
- calibrate.cpp, calibrate.h
- utils.cpp, utils.h
- ar.cpp, ar.h
- task7_orb.cpp, task7_fast_features.cpp
- Makefile
- README.md
- cvpr_proj4_report.pdf

## Extensions Completed

**Multi-Object AR:** Simultaneous display of multiple 3D virtual objects with independent toggle controls.

## Extensions In Progress

**3D Textured Model Rendering (OpenGL):** Attempting to render photorealistic glasses models with textures for virtual try-on application using OpenGL integration with OpenCV.

## Known Limitations

- Checkerboard must be reasonably flat
- AR mode requires calibration completion first

## Time Travel Days

## Notes

- Calibration improves with more images (5 minimum, 10-15 recommended)
- Virtual objects defined in checkerboard coordinate system (1 unit = 1 square)
