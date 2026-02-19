//
// Created by Gautam Ajey Khanapuri
// 07 Feb 2026
// Header file for dist_utils.cpp. Defines distance metrics for comparing feature vectors
// and classes for handling different matching modes (Basic, Classic, DNN).
//

#ifndef DIST_UTILS_H
#define DIST_UTILS_H

#include <functional>
#include <iostream>
#include <algorithm>
#include <iomanip>

#include "mycv_utils.h"
#include "csv_util.h"

/**
 * Function pointer type for distance/similarity metric functions.
 * Takes two feature vectors and returns a distance value (lower = more similar).
 */
typedef std::function<double(const std::vector<float>&, const std::vector<float>&)> DistanceFunction;
// const int config_size = 4;

const int config_size = 4;  // Size of each configuration group in spec string (part + hist + metric + weight)

/**
 * Available distance metrics for comparing feature vectors.
 * Each metric has different properties suited for different feature types.
 */
enum DistanceMetric {
    SSD,                      // i - Sum of squared differences
    INTERSECTION,             // I - Histogram intersection
    CHI_SQUARED,              // q - Chi-squared distance
    EARTH_MOVER,              // Q - Earth mover's distance (1D)
    COSINE,                   // o - Cosine similarity
    CORELATION,               // O - Correlation coefficient
    BHATTACHARYA,             // y - Bhattacharyya distance
    MANHATTAN,                // Y - Manhattan (L1) distance
    CUSTOM_PERCEPTUAL,        // p - Custom perceptual weighting
    CUSTOM_WEIGHTED_COMBO     // P - Custom weighted combination
};

/**
 * Maps distance metrics to readable names for output display.
 */
const std::unordered_map<DistanceMetric, std::string> DISTMETRIC_NAMES = {
    {SSD, "ssd"},
    {INTERSECTION, "intersection"},
    {CHI_SQUARED, "chi_squared"},
    {EARTH_MOVER, "earth_mover"},
    {COSINE, "cosine"},
    {CORELATION, "corelation"},
    {BHATTACHARYA, "bhattacharya"},
    {MANHATTAN, "manhattan"},
    {CUSTOM_PERCEPTUAL, "custom_perceptual"},
    {CUSTOM_WEIGHTED_COMBO, "custom_weighted_combo"},
};

/**
 * Configuration for a single part/histogram/metric combination.
 * Used by Distance class to handle multi-part weighted matching.
 */
struct PartConfig {
    std::string part_name;                              // Name of image part (e.g., "top", "whole")
    HistogramType hist_type;                            // Type of histogram used
    DistanceMetric metric;                              // Distance metric to use
    double weight;                                      // Weight for this part in final combination (normalized)
    fs::path feature_file;                              // CSV file containing features for this part
    std::vector<std::vector<float>> img_vecs;           // Feature vectors for all database images
    std::vector<fs::path> img_names;                    // Image paths corresponding to feature vectors
    std::vector<float> target_vector;                   // Target image feature vector for this part
    std::vector<std::pair<double, fs::path>> diff_values;  // Distance values for all images
};

/**
 * Parses distance metric character to DistanceMetric enum.
 *
 * @param c character representing distance metric
 * @return corresponding DistanceMetric enum value
 */
DistanceMetric parse_distance_metric(char c);

/**
 * Dispatches to appropriate distance function based on metric type.
 *
 * @param x first feature vector
 * @param y second feature vector
 * @param metric distance metric to use
 * @return distance value (lower = more similar)
 */
double compute_distance(std::vector<float> const& x, std::vector<float> const& y, const DistanceMetric& metric);

/**
 * Computes sum of squared differences between two vectors.
 * Good for general purpose matching.
 *
 * @param x first feature vector
 * @param y second feature vector
 * @return SSD distance (lower = more similar)
 */
double compute_ssd(const std::vector<float>& x, const std::vector<float>& y);

/**
 * Computes histogram intersection distance.
 * Good for color histogram matching.
 *
 * @param x first histogram
 * @param y second histogram
 * @return intersection distance (lower = more similar)
 */
double compute_intersection(const std::vector<float>& x, const std::vector<float>& y);

/**
 * Computes chi-squared distance.
 * Good for statistical comparison of histograms.
 *
 * @param x first histogram
 * @param y second histogram
 * @return chi-squared distance (lower = more similar)
 */
double compute_chi_squared(const std::vector<float>& x, const std::vector<float>& y);

/**
 * Computes approximate earth mover's distance for multi-dimensional histograms.
 *
 * @param x first histogram
 * @param y second histogram
 * @return approximate EMD (lower = more similar)
 */
double compute_emd_approx(const std::vector<float>& x, const std::vector<float>& y);

/**
 * Computes earth mover's distance for 1D histograms.
 * Good for perceptual color similarity.
 *
 * @param x first histogram
 * @param y second histogram
 * @return EMD distance (lower = more similar)
 */
double compute_earth_mover(const std::vector<float>& x, const std::vector<float>& y);

/**
 * Computes cosine similarity distance.
 * Good for comparing distribution shapes regardless of magnitude.
 *
 * @param x first feature vector
 * @param y second feature vector
 * @return cosine distance (lower = more similar)
 */
double compute_cosine_similarity(const std::vector<float>& x, const std::vector<float>& y);

/**
 * Computes correlation coefficient distance.
 * Good for pattern matching regardless of offset.
 *
 * @param x first feature vector
 * @param y second feature vector
 * @return correlation distance (lower = more similar)
 */
double compute_corelation(const std::vector<float>& x, const std::vector<float>& y);

/**
 * Computes Bhattacharyya distance.
 * Good for probabilistic comparison of distributions.
 *
 * @param x first histogram
 * @param y second histogram
 * @return Bhattacharyya distance (lower = more similar)
 */
double compute_bhattacharya(const std::vector<float>& x, const std::vector<float>& y);

/**
 * Computes Manhattan (L1) distance.
 * Good for general purpose matching with robustness to outliers.
 *
 * @param x first feature vector
 * @param y second feature vector
 * @return Manhattan distance (lower = more similar)
 */
double compute_manhattan(const std::vector<float>& x, const std::vector<float>& y);

/**
 * Custom distance metric with perceptual weighting.
 * Emphasizes mid-range bins over extreme values.
 *
 * @param x first feature vector
 * @param y second feature vector
 * @return custom perceptual distance (lower = more similar)
 */
double compute_custom_perceptual_distance(const std::vector<float>& x, const std::vector<float>& y);

/**
 * Custom distance metric combining multiple standard metrics.
 * Application-specific weighted combination.
 *
 * @param x first feature vector
 * @param y second feature vector
 * @return custom combined distance (lower = more similar)
 */
double compute_custom_weighted_combo(const std::vector<float>& x, const std::vector<float>& y);

/**
 * Registry mapping distance metrics to their implementation functions.
 * Enables dynamic dispatch based on metric type.
 */
const std::map<DistanceMetric, DistanceFunction> distance_functions = {
    {SSD, [](const std::vector<float>& x, const std::vector<float>& y) {return compute_ssd(x, y);} },
    {INTERSECTION, [](const std::vector<float>& x, const std::vector<float>& y) {return compute_intersection(x, y);} },
    {CHI_SQUARED, [](const std::vector<float>& x, const std::vector<float>& y) {return compute_chi_squared(x, y);} },
    {EARTH_MOVER, [](const std::vector<float>& x, const std::vector<float>& y) {return compute_earth_mover(x, y);} },
    {COSINE, [](const std::vector<float>& x, const std::vector<float>& y) {return compute_cosine_similarity(x, y);} },
    {CORELATION, [](const std::vector<float>& x, const std::vector<float>& y) {return compute_corelation(x, y);} },
    {BHATTACHARYA, [](const std::vector<float>& x, const std::vector<float>& y) {return compute_bhattacharya(x, y);} },
    {MANHATTAN, [](const std::vector<float>& x, const std::vector<float>& y) {return compute_manhattan(x, y);} },
    {CUSTOM_PERCEPTUAL, [](const std::vector<float>& x, const std::vector<float>& y) {return compute_custom_perceptual_distance(x, y);} },
    {CUSTOM_WEIGHTED_COMBO, [](const std::vector<float>& x, const std::vector<float>& y) {return compute_custom_weighted_combo(x, y);} }
};

/**
 * Handles CLASSIC mode matching with multiple weighted histogram parts.
 * Loads features from multiple CSV files, computes target features,
 * calculates weighted combination of distances, and returns sorted results.
 */
class Distance {
private:
    fs::path tgt_file;                              // Path to target image
    std::string spec;                               // Specification string (part+hist+metric+weight groups)
    std::vector<fs::path> vec_files;                // Paths to feature CSV files
    int num_images;                                 // Number of images in database
    std::vector<std::pair<double, fs::path>>& op;    // Output: sorted (distance, path) pairs
    std::vector<PartConfig> pc;                     // Parsed configuration for each part

    /**
     * Parses specification string and prepares PartConfig entries.
     * Normalizes weights to sum to 1.0.
     *
     * @return 0 if successful
     */
    int prep_configs();

public:
    /**
     * Constructor for Distance class.
     *
     * @param tgt_file path to target image
     * @param spec specification string (groups of 4: part+hist+metric+weight)
     * @param vf vector of feature file paths (must match spec order)
     * @param op reference to output vector for storing results
     */
    Distance(fs::path& tgt_file, std::string& spec, std::vector<fs::path>& vf, std::vector<std::pair<double, fs::path>>& op);

    /**
     * Executes classic multi-part weighted matching.
     * Loads features, computes distances, combines with weights, and sorts results.
     *
     * @return 0 if successful
     */
    int calculate_classic();

};

/**
 * Handles DNN embedding matching with single distance metric.
 * Target and database embeddings stored in CSV format.
 */
class MyDNN {
private:
    fs::path tgt_file;                              // Path to CSV containing target embedding
    std::vector<float> tgt_vector;                  // Target image embedding vector
    std::string spec;                               // Single character distance metric
    std::vector<fs::path> vec_files;                // Path to database embeddings CSV
    std::vector<std::vector<float>> img_vecs;       // Database embedding vectors
    std::vector<fs::path> img_names;                // Image paths for database
    std::vector<std::pair<double, fs::path>>& op;    // Output: sorted (distance, path) pairs
    DistanceMetric metric;                          // Distance metric to use

    /**
     * Parses single-character metric specification.
     *
     * @return 0 if successful
     */
    int prep_config();

public:
    /**
     * Constructor for MyDNN class.
     *
     * @param tgt_file path to CSV with target embedding (first entry used)
     * @param spec single character distance metric
     * @param vf vector containing single path to embeddings CSV
     * @param op reference to output vector for storing results
     */
    MyDNN(fs::path& tgt_file, std::string& spec, std::vector<fs::path>& vf, std::vector<std::pair<double, fs::path>>& op);

    /**
     * Executes DNN embedding matching.
     * Loads target and database embeddings, computes distances, and sorts results.
     *
     * @return 0 if successful
     */
    int calculate_dnn();
};

/**
 * Handles BASIC mode matching using 7x7 center square baseline.
 * Computes features on-the-fly rather than loading from CSV.
 */
class Basic {
private:
    fs::path tgt_file;                              // Path to target image
    std::vector<float> tgt_vector;                  // Target 7x7 center square vector
    std::string spec;                               // Single character distance metric (typically 'd' for SSD)
    std::vector<fs::path> vec_files;                // Path to baseline features CSV
    std::vector<std::pair<double, fs::path>>& op;    // Output: sorted (distance, path) pairs
    DistanceMetric metric;                          // Distance metric to use
    std::vector<fs::path> img_paths;                // Database image paths

    /**
     * Parses single-character metric specification.
     *
     * @return 0 if successful
     */
    int prep_config();

public:
    /**
     * Constructor for Basic class.
     *
     * @param tgt_file path to target image
     * @param spec single character distance metric
     * @param vf vector containing path to baseline features CSV
     * @param op reference to output vector for storing results
     */
    Basic(fs::path& tgt_file, std::string& spec, std::vector<fs::path>& vf, std::vector<std::pair<double, fs::path>>& op);

    /**
     * Executes basic 7x7 square matching.
     * Computes SSD between center squares and sorts results.
     *
     * @return 0 if successful
     */
    int calculate_basic();
};

#endif //DIST_UTILS_H
