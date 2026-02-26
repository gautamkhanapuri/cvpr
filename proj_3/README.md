# CS 5330 Project 3: Real-Time Object Recognition
**Author:** Gautam Ajey Khanapuri  
**Date:** 23 February 26, 2026  
**Operating System:** macOS (Apple Silicon M1)  
**IDE/Editor:** CLion with  command-line compilation using Makefile
**Compiler:** clang++ (Apple Clang)

## Video Demonstration
https://youtu.be/98nBETLNPZk

## Compilation
```bash
make clean
make all
```

This creates the executable: `rtor`

## Running the System

```bash
./rtor <feature_database.csv> <dnn_database.csv>
```
Example: `./rtor feature_db.csv dnn_feature_db.csv`

**Requirements:**
- Both files need to be provided. These files must exist even if empty.
- `resnet18-v2-7.onnx` must be in the same directory as executable
- External USB webcam recommended (device ID 1)
- White background platform with diffused lighting

## Keyboard Controls

**Display Modes:**
- `t` - Toggle binary threshold view
- `m` - Toggle morphology view
- `q` - Quit program
- `s` - Save images


**Training:**
- `n` - Train hand-crafted features (prompts for labels)
- `N` - Train ResNet embeddings (prompts for labels)
- `w` - Reset white screen reference (if using background subtraction mode)

**Classification:**
- `r` - Toggle ResNet classification display (requires ResNet training data)

## Thresholding Modes

The system supports two thresholding methods (hardcoded at compile time):

**Mode 0 (Default):** Dynamic k-means clustering  
**Mode 1:** White background subtraction

To change mode, modify `threshold_mode` in constructor (default = 0).

## Training Instructions

1. Place one or multiple objects on white platform
2. Press 'n' for training hand-crafted features or 'N' for ResNet
3. System displays cropped objects one by one
4. Press 'a' to annotate, 'p' to skip when the program is in focus
5. Enter label when prompted (or create new category) in the terminal
6. Repeat for 3-5 examples per object class
7. Training data auto-saves to CSV on exit

## Testing Extensions

**Multi-Object Recognition:**
- Place multiple objects on platform
- System detects, tracks, and labels all objects simultaneously
- Each object maintains stable color across frames

**Background Subtraction:**
- Captures white platform at startup
- Successfully detects bright/light-colored objects
- Reset background with 'w' key if platform moves

**ResNet One-Shot Learning:**
- Train with single example per class
- Compare predictions with hand-crafted features
- Both labels displayed simultaneously when enabled

## Files Submitted
- main.cpp
- rtor.cpp, rtor.h
- threshold.cpp, threshold.h
- morph.cpp, morph.h
- segment.cpp, segment.h
- feature.cpp, feature.h
- resnetclassifier.cpp, resnetclassifier.h
- csv_util.cpp, csv_util.h
- utils.cpp, utils.h
- Makefile

## Time Travel Days

## Notes
- System performs best with dark objects on white background
- Shiny objects may fragment due to specular highlights
- Ensure adequate diffused lighting (paper over LED lamp recommended)
- Feature databases are CSV files that persist across sessions