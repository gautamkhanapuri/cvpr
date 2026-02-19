# CS 5030 Project 2: Content-Based Image Retrieval
**Author:** Gautam Ajey Khanapuri
**Email:** khanapuri.g@northeastern.edu
**Date:** February 11, 2026  
**Operating System:** macOS [VERSION - Tahoe 26.2]  
**Edorit:** Vim. Makefile. Command-line compilation  
**Compiler:** clang++ (Apple Clang version 20)

## Project Overview
Implementation of a content-based image retrieval (CBIR) system using custom color and texture features, with comparison to deep neural network embeddings.

## Video Links
None

## Code Files Submitted
**Program Part 1 (Feature Extraction):**
- p1.cpp, p1.h

**Program Part 2 (Matching):**
- p2.cpp, p2.h
- dist_utils.cpp, dist_utils.h

**Shared utilities:**
- mycv_utils.cpp, mycv_utils.h
- csv_util.cpp, csv_util.h
- utils.cpp, utils.h
- Makefile

## Compilation Instructions
```bash
# Build both programs:
make clean
make all

# This creates two executables: p1 and p2
```

## Running Instructions

### Program 1 (P1): Feature Extraction

**Purpose:** Pre-computes feature vectors for all images in a directory.

**Baseline Mode:**
```bash
./p1 -b- <image_directory>
```
Example: `./p1 -b- ~/datasets/olympus`

**Multi-Histogram Mode:**
```bash
./p1 -m-<spec> <image_directory>
```
Example: `./p1 -m-wrTR ~/datasets/olympus`

Spec format: Groups of 2 characters (part + histogram type)
- Parts: w=whole, t=top, T=bottom, l=left, L=right, c=center
- Histograms: r=RG, R=RGB, h=HS, u=intensity, s=sobel_mag_1d, S=sobel_mag_2d, g=GLCM, G=Laws


### Program 2 (P2): Image Matching

**Purpose:** Matches target image against pre-computed feature database.

**Baseline Mode:**
```bash
./p2 <target_image> -b-<metric> <baseline_csv>
```
Example: `./p2 pic.1016.jpg -b-i baseline_ft_vec_12345.csv`

**Classic Multi-Histogram Mode:**
```bash
./p2 <target_image> -m-<spec> <csv1> <csv2> ...
```
Example: `./p2 pic.0274.jpg -m-tRI1TRI1 top_rgb_*.csv bottom_rgb_*.csv`

Spec format: Groups of 4 characters (part + histogram + metric + weight)
- Distance metrics: i=SSD, I=intersection, q=chi-squared, o=cosine, O=correlation, y=bhattacharyya, Y=manhattan

**DNN Mode:**
```bash
./p2 <target_csv> -d-<metric> <embeddings_csv>
```
Example: `./p2 target.csv -d-o dnn_embeddings.csv`

**Note**: Target must be CSV with single entry containing target image name and embedding.

Output: Interactive display of ranked results

## Testing Task Results

### Task 1: Baseline Matching
```bash
./p1 -b- ~/datasets/olympus
./p2 pic.1016.jpg -b-i baseline_ft_vec_*.csv
```
Expected: pic.0986, pic.0641, pic.0547 in top results

### Task 2: Histogram Matching
```bash
./p1 -m-wr ~/datasets/olympus
./p2 pic.0164.jpg -m-wrI1 whole_rg_*.csv
```

### Task 3: Multi-Histogram
```bash
./p1 -m-tRTR ~/datasets/olympus
./p2 pic.0274.jpg -m-tRI1TRI1 top_rgb_*.csv bottom_rgb_*.csv
```

### Task 4: Texture and Color
```bash
./p1 -m-wrwS ~/datasets/olympus
./p2 pic.0535.jpg -m-wrI1wSO1 whole_rg_*.csv whole_sobel_*.csv
```

### Task 7: Custom Building Retrieval
```bash
./p1 -m-TGwSTRth ~/datasets/olympus
./p2 pic.0212.jpg -m-TGO7wSO5TRq4thI3 bottom_law_*.csv whole_sobel_*.csv bottom_rgb_*.csv top_hs_*.csv
```

## Extensions Implemented

1. **GLCM Texture Features** - Co-occurrence matrix analysis with 5 statistical measures
2. **Laws Filter Response Histograms** - 9 filter combinations for material texture
3. **Custom Distance Metrics** - Perceptual weighting and weighted combinations
4. **Comprehensive Metric Suite** - 10 distance metrics including EMD, Bhattacharyya
5. **Flexible Configuration System** - Extensible architecture supporting arbitrary feature combinations

## Time Travel Days
1 day used.

## Notes
- All images should be in standard formats: .jpg, .jpeg, .png
- Feature extraction (P1) takes 10-20 seconds (only if you are generating vectors for multiple parts) for full olympus dataset
- Matching (P2) takes 1-2 seconds per query
- **CSV files from P1 must be provided to P2 in correct order matching spec string**