//
// Created by Gautam Ajey Khanapuri
// 07 Feb 2026
// Header file for all opencv functionalities used by part 1 and part2 of the program. This file also contains all the
// constants defined for the functions.
//

#ifndef MYCV_UTILS_H
#define MYCV_UTILS_H

#include <opencv2/core.hpp>

#include <filesystem>
#include <string>
#include <vector>
#include <map>
#include <iterator>
#include <functional>

namespace fs = std::filesystem;

/**
 * Function pointer type for histogram computation functions.
 * Takes image path, output vector, and part name as parameters.
 */
typedef std::function<int(const fs::path&, std::vector<float>&, std::string&)> HistogramFunction;

// Sobel filter kernels for separable convolution
const int SOBEL_g[3] = {1, 2, 1};      // Gaussian smoothing kernel
const int SOBEL_h[3] = {-1, 0, 1};     // Horizontal derivative kernel
const int SOBEL_r[3] = {1, 0, -1};     // Reverse derivative kernel

// Default parameters for texture analysis
const int angle_bins = 9;              // Number of orientation bins (0-180 degrees)
const int quantize_levels = 16;       // Grayscale quantization levels for GLCM

/**
 * Spatial offsets for GLCM computation.
 * Format: {dx, dy} where dx is horizontal offset, dy is vertical offset.
 * Captures texture at different scales and orientations.
 */
const std::pair<int, int> offsets[4] = {
  {1, 0},  // Right neighbor (horizontal texture)
  {0, 1},  // Below neighbor (vertical texture)
  {1, 1},  // Diagonal neighbor (45-degree texture)
  {2, 0}   // 2 pixels right (coarser horizontal texture)
};
// const std::array<float, 5> L5 = {1, 4, 6, 4, 1};  // Level
// const std::array<float, 5> E5 = {-1, -2, 0, 2, 1};  // Edge
// const std::array<float, 5> S5 = {-1, 0, 2, 0, -1};  // Spot
// const std::array<float, 5> W5 = {-1, 2, 0, -2, 1};  // Wave
// const std::array<float, 5> R5 = {1, -4, 6, -4, 1};  // Ripple

// Laws filter 1D kernels for texture analysis
const std::array<float, 5> L5 = {1, 4, 6, 4, 1};      // Level (averaging)
const std::array<float, 5> E5 = {-1, -2, 0, 2, 1};    // Edge detection
const std::array<float, 5> S5 = {-1, 0, 2, 0, -1};    // Spot detection
const std::array<float, 5> W5 = {-1, 2, 0, -2, 1};    // Wave detection
const std::array<float, 5> R5 = {1, -4, 6, -4, 1};    // Ripple detection

/**
 * The 9 Laws filter combinations used for texture analysis.
 * Each entry is a pair of 1D kernels that are combined to create a 2D filter.
 * Format: {vertical_kernel, horizontal_kernel}
 */
const std::pair<std::array<float, 5>, std::array<float, 5>> laws_filters[9] = {
  {L5, E5},  // Vertical edges
  {E5, L5},  // Horizontal edges
  {E5, E5},  // All edges
  {S5, S5},  // Spots
  {L5, S5},  // Vertical spots
  {S5, L5},  // Horizontal spots
  {W5, W5},  // Waves
  {R5, R5},  // Ripples
  {E5, S5}   // Edge-spot combinations
};

/**
 * The allowed split types for dividing images into regions.
 */
enum SplitType {
  WHOLE,      // w - entire image
  TOP,        // t - top half
  BOTTOM,     // T - bottom half
  LEFT,       // l - left half
  RIGHT,      // L - right half
  QUADRANTS,  // a,A,b,B - four quadrants in anti-clockwise order
  CENTER,     // c - center region
  EDGE        // C - edge/border region
};

/**
 * Allowed histogram types for feature extraction.
 */
enum HistogramType {
  BASIC_BOX,         // B - 7x7 center square baseline
  RG_CHROMATICITY,   // r - RG chromaticity histogram
  RGB,               // R - RGB color histogram
  HS,                // h - Hue-Saturation histogram
  INTENSITY,         // u - grayscale intensity histogram
  SOBEL_MAG_1D,      // s - 1D gradient magnitude histogram
  SOBEL_MAGvORN_2D,  // S - 2D magnitude vs orientation histogram
  GLCM,              // g - Gray-level co-occurrence matrix features
  LAW                // G - Laws filter response histograms
};


/**
 * Maps histogram types to readable names used in output file naming.
 */
const std::unordered_map<HistogramType, std::string> HISTOGRAM_NAMES = {
  {BASIC_BOX, "basic_central_box"},
  {RG_CHROMATICITY, "rg"},
  {RGB, "rgb"},
  {HS, "hs"},
  {INTENSITY, "intensity"},
  {SOBEL_MAG_1D, "sobel_magnitude_1d"},
  {SOBEL_MAGvORN_2D, "sobel_magnitude_vs_orientation_2d"},
  {GLCM, "glcm"},
  {LAW, "law"}
};

/**
 * Holds configuration information for feature extraction including
 * image parts, histogram types, regions of interest, and bin counts.
 */
struct FeatureConfig {
  std::vector<std::string> parts;        // Part names (e.g., "top", "bottom")
  std::vector<HistogramType> hists;      // Histogram types for each part
  std::vector<cv::Rect> boxes;           // Region rectangles for each part
  std::vector<int> bins;                 // Bin counts for each histogram
};

/**
 * Parses the specification string and generates a FeatureConfig instance.
 *
 * @param spec_str obtained from command line after removing prefix (-m-)
 * @param config assigned values are stored in this reference
 * @return 0 if no error, exits otherwise
 */
int parse_config(std::string spec_str, FeatureConfig& config);

/**
 * Converts a character to its corresponding part name.
 *
 * @param c the character from spec corresponding to image part
 * @return the part name string (e.g., "top", "bottom", "whole")
 */
std::string parse_part_name(char c);

/**
 * Converts a character to its corresponding HistogramType.
 *
 * @param c the character from spec corresponding to histogram type
 * @return the corresponding HistogramType enum value
 */
HistogramType parse_hist_type(char c);

/**
 * Generates the cv::Rect region of interest for a given part name.
 *
 * @param ch the part name string (e.g., "top", "center")
 * @param r number of rows in the image
 * @param c number of columns in the image
 * @return the Rect defining the region boundaries
 */
cv::Rect parse_rect_size(std::string ch, int r, int c);

// Default bin counts for each histogram type
const int basic_box_size = 7;          // 7x7 square for baseline matching
const int rg_bins = 16;                // 32x32 bins for RG chromaticity
const int rgb_bins = 8;               // 32x32x32 bins for RGB (flattened)
const int hs_bins = 32;                // 32x32 bins for Hue-Saturation
const int intensity_bins = 32;         // 32 bins for grayscale intensity
const int sobel_mag_1d_bins = 16;      // 32 bins for gradient magnitude
const int sobel_mag_2d_bins = 16;      // 32 bins per dimension (mag and orientation)
const int glcm_bins = 16;              // 16 grayscale levels for co-occurrence matrix
const int law_bins = 16;               // 16 bins for Laws filter response histogram

/**
 * Maps each HistogramType to its default number of bins.
 */
const std::map<HistogramType, int> histogram_bins = {
  {HistogramType::BASIC_BOX, basic_box_size},
  {HistogramType::RG_CHROMATICITY, rg_bins },
  {HistogramType::RGB, rgb_bins },
  {HistogramType::HS, hs_bins },
  {HistogramType::INTENSITY, intensity_bins },
  {HistogramType::SOBEL_MAG_1D, sobel_mag_1d_bins },
  {HistogramType::SOBEL_MAGvORN_2D, sobel_mag_2d_bins },
  {HistogramType::GLCM, glcm_bins },
  {HistogramType::LAW, law_bins }
};

/**
 * Main dispatcher function that calls the appropriate histogram computation function
 * based on the histogram type specified.
 *
 * @param im_path path to the image file
 * @param vec output vector where histogram/features will be stored
 * @param hist_type type of histogram to compute
 * @param part name of the image part/region to process
 * @return 0 if successful, non-zero otherwise
 */
int compute_histogram(const fs::path& im_path, std::vector<float>& vec, HistogramType hist_type, std::string& part);

/**
 * Computes baseline 7x7 center square feature vector.
 *
 * @param img_path path to the image file
 * @param vec output vector (49 values for 7x7 square)
 * @param part name of the image part (typically "whole")
 * @param bins size of the center square (default 7)
 * @return 0 if successful
 */
int compute_basic_box_vector(const fs::path& img_path, std::vector<float>& vec, std::string& part, int bins=basic_box_size);

/**
 * Computes RG chromaticity histogram for the specified image region.
 *
 * @param img_path path to the image file
 * @param vec output vector (bins x bins values)
 * @param part name of the image part to process
 * @param bins number of bins per dimension (default 32)
 * @return 0 if successful
 */
int compute_rg_histogram(const fs::path& img_path, std::vector<float>& vec, std::string& part, int bins=rg_bins);

/**
 * Computes 3D RGB color histogram for the specified image region.
 *
 * @param img_path path to the image file
 * @param vec output vector (bins^3 values, flattened)
 * @param part name of the image part to process
 * @param bins number of bins per channel (default 32)
 * @return 0 if successful
 */
int compute_rgb_histogram(const fs::path& img_path, std::vector<float>& vec, std::string& part, int bins=rgb_bins);

/**
 * Computes Hue-Saturation histogram for the specified image region.
 *
 * @param img_path path to the image file
 * @param vec output vector (bins x bins values)
 * @param part name of the image part to process
 * @param bins number of bins per dimension (default 32)
 * @return 0 if successful
 */
int compute_hs_histogram(const fs::path& img_path, std::vector<float>& vec, std::string& part, int bins=hs_bins);

/**
 * Computes grayscale intensity histogram for the specified image region.
 *
 * @param img_path path to the image file
 * @param vec output vector (bins values)
 * @param part name of the image part to process
 * @param bins number of intensity bins (default 32)
 * @return 0 if successful
 */
int compute_intensity_histogram(const fs::path& img_path, std::vector<float>& vec, std::string& part, int bins=intensity_bins);

/**
 * Computes 1D histogram of Sobel gradient magnitudes.
 *
 * @param img_path path to the image file
 * @param vec output vector (bins values)
 * @param part name of the image part to process
 * @param bins number of magnitude bins (default 32)
 * @return 0 if successful
 */
int compute_sobel_mag_1d_histogram(const fs::path& img_path, std::vector<float>& vec, std::string& part, int bins=sobel_mag_1d_bins);

/**
 * Computes 2D histogram of Sobel gradient magnitude vs orientation.
 * Combines magnitude strength with edge direction for texture analysis.
 *
 * @param img_path path to the image file
 * @param vec output vector (mag_bins x angle_bins values, flattened)
 * @param part name of the image part to process
 * @param bins number of bins per dimension (default 32)
 * @return 0 if successful
 */
int compute_sobel_mag_vs_orn_2d_histogram(const fs::path& img_path, std::vector<float>& vec, std::string& part, int bins=sobel_mag_2d_bins);

/**
 * Computes GLCM (Gray-Level Co-occurrence Matrix) texture features.
 * Extracts 5 features (energy, contrast, homogeneity, entropy, max probability)
 * for each of 4 spatial offsets.
 *
 * @param img_path path to the image file
 * @param vec output vector (20 values: 5 features x 4 offsets)
 * @param part name of the image part to process
 * @param bins number of grayscale quantization levels (default 16)
 * @return 0 if successful
 */
int compute_glcm_features(const fs::path& img_path, std::vector<float>& vec, std::string& part, int bins=glcm_bins);

/**
 * Computes histograms of Laws filter responses for texture analysis.
 * Applies 9 different Laws filter combinations and creates response histograms.
 *
 * @param img_path path to the image file
 * @param vec output vector (9 filters x bins values)
 * @param part name of the image part to process
 * @param bins number of bins per filter response histogram (default 16)
 * @return 0 if successful
 */
int compute_law_histogram(const fs::path& img_path, std::vector<float>& vec, std::string& part, int bins=law_bins);

/**
 * Registry mapping histogram types to their computation functions.
 * Uses lambda functions to provide uniform interface for all histogram types.
 */
const std::map<HistogramType, HistogramFunction> histogram_functions = {
  {HistogramType::BASIC_BOX, [](const fs::path& p, std::vector<float>& s, std::string& v) {return compute_basic_box_vector(p, s, v);} },
  {HistogramType::RG_CHROMATICITY, [](const fs::path& p, std::vector<float>& s, std::string& v) {return compute_rg_histogram(p, s, v);} },
  {HistogramType::RGB, [](const fs::path& p, std::vector<float>& s, std::string& v) {return compute_rgb_histogram(p, s, v);} },
  {HistogramType::HS, [](const fs::path& p, std::vector<float>& s, std::string& v) {return compute_hs_histogram(p, s, v);} },
  {HistogramType::INTENSITY, [](const fs::path& p, std::vector<float>& s, std::string& v) {return compute_intensity_histogram(p, s, v);} },
  {HistogramType::SOBEL_MAG_1D, [](const fs::path& p, std::vector<float>& s, std::string& v) {return compute_sobel_mag_1d_histogram(p, s, v);} },
  {HistogramType::SOBEL_MAGvORN_2D, [](const fs::path& p, std::vector<float>& s, std::string& v) {return compute_sobel_mag_vs_orn_2d_histogram(p, s, v);} },
  {HistogramType::GLCM, [](const fs::path& p, std::vector<float>& s, std::string& v) {return compute_glcm_features(p, s, v);} },
  {HistogramType::LAW, [](const fs::path& p, std::vector<float>& s, std::string& v) {return compute_law_histogram(p, s, v);} }
};

/**
 * Computes Sobel X gradient using 3x3 separable filter.
 *
 * @param src input image
 * @param dst output gradient image (CV_16SC3 for signed values)
 * @return 0 if successful
 */
int sobelX3x3(cv::Mat &src, cv::Mat &dst);

/**
 * Computes Sobel Y gradient using 3x3 separable filter.
 *
 * @param src input image
 * @param dst output gradient image (CV_16SC3 for signed values)
 * @return 0 if successful
 */
int sobelY3x3(cv::Mat &src, cv::Mat &dst);

/**
 * Computes gradient magnitude from Sobel X and Y components.
 *
 * @param sx Sobel X gradient image
 * @param sy Sobel Y gradient image
 * @param dst output magnitude image (CV_8UC3)
 * @return 0 if successful
 */
int magnitude(cv::Mat &sx, cv::Mat &sy, cv::Mat &dst);

/**
 * Quantizes grayscale image to specified number of levels.
 * Reduces 256 gray levels to a smaller set for GLCM computation.
 *
 * @param src input grayscale image
 * @param dst output quantized image
 * @param levels number of quantization levels (e.g., 16)
 * @return 0 if successful
 */
int quantize_img(cv::Mat& src, cv::Mat& dst, int levels);

/**
 * Computes gray-level co-occurrence matrix for given spatial offset.
 *
 * @param src input quantized grayscale image
 * @param dst output co-occurrence matrix (levels x levels)
 * @param dx horizontal offset
 * @param dy vertical offset
 * @return 0 if successful
 */
int compute_cooccurrence_matrix(cv::Mat& src, cv::Mat& dst, int dx, int dy);

/**
 * Calculates five GLCM texture features from a co-occurrence matrix.
 *
 * @param co_mat input co-occurrence matrix
 * @param energy output: sum of squared elements
 * @param contrast output: intensity contrast between neighbors
 * @param homogeneity output: similarity of neighbors
 * @param entropy output: randomness of texture
 * @param max_prob output: most dominant co-occurrence pair
 * @return 0 if successful
 */
int calculate_glcm_vectors(cv::Mat& co_mat, float& energy, float& contrast, float& homogeneity, float& entropy, float& max_prob);

/**
 * Creates a 5x5 2D Laws filter from two 1D kernels.
 * Combines vertical and horizontal kernels via outer product.
 *
 * @param a vertical 1D kernel (5 elements)
 * @param b horizontal 1D kernel (5 elements)
 * @param filter output 5x5 filter matrix
 * @return 0 if successful
 */
int create_laws_filter(const float a[5], const float b[5], cv::Mat& filter);

/**
 * Computes histogram of Laws filter response values.
 *
 * @param src filtered image (response map)
 * @param hist output histogram vector
 * @param bins number of histogram bins
 * @return 0 if successful
 */
int calculate_law_response_histogram(cv::Mat& src, std::vector<float>& hist, int bins);

#endif //MYCV_UTILS_H
