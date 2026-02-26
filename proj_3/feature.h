//
// Created by Ajey K on 23/02/26.
// Header file for Feature extraction and Classification modules.
// Implements geometric and moment-based feature computation with scaled Euclidean distance classification.
//

#ifndef FEATURE_H
#define FEATURE_H

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>

#include <filesystem>
#include <set>
#include <iostream>

#include "csv_util.h"
#include "segment.h"

namespace fs = std::filesystem;

// Window title for training mode display
inline const std::string training_img_display_window_name =
        "Training Image Display. 'p' - skip region / 'a' - add label";

// Number of features in feature vector (percent_filled, aspect_ratio, compactness, circularity, hu0, hu1, hu2)
inline const int num_features = 7;

// Distance threshold for accepting classification match (scaled Euclidean distance)
inline const float label_matching_threshold = 1.75;

// Label assigned when no match found within threshold
inline const std::string unknown = "Unknown";

/**
 * Feature extraction module.
 * Computes rotation, scale, and translation-invariant features from segmented regions.
 */
class Feature {
public:
    Feature() {
    };

    /**
     * Computes 7-dimensional feature vector for each region.
     * Features: percent_filled, aspect_ratio, compactness, circularity, and first 3 Hu moments.
     * @param regions vector of detected regions (features stored in each RegionStats)
     * @return 0 if successful
     */
    int calculate_basic_2d_features(std::vector<RegionStats> &regions);

    /**
     * Overlays feature values and predicted labels on video frame.
     * Displays classification results and selected feature values near each object.
     * @param src video frame to draw on
     * @param regions detected regions with computed features and labels
     * @return 0 if successful
     */
    int overlay_features(cv::Mat &src, std::vector<RegionStats> &regions);
};

/**
 * Classifier for hand-crafted features using nearest-neighbor with scaled Euclidean distance.
 * Maintains training database, computes feature statistics, and performs real-time classification.
 */
class Classifier {
    fs::path db_path; // Path to feature database CSV file
    std::vector<std::string> labels; // Training example labels
    std::vector<std::vector<float> > features; // Training example feature vectors
    std::vector<float> average; // Mean of each feature across training set
    std::vector<float> stddev; // Standard deviation of each feature
    int initial_datapoints_count; // Number of examples loaded from file at startup
    std::set<std::string> labels_set; // Set of unique labels for validation

    /**
     * Loads training examples from CSV file.
     * @return 0 if successful
     */
    int load_db_file();

    /**
     * Checks if label exists in current label set.
     * @param label label string to validate
     * @return true if label exists
     */
    bool is_valid_label(const std::string &label) const;

    /**
     * Adds new label to label set for validation.
     * @param label new label to register
     * @return 0 if successful
     */
    int register_new_label(const std::string &label);

    /**
     * Recalculates feature means from current training set.
     * @return 0 if successful
     */
    int recalculate_average();

    /**
     * Recalculates feature standard deviations from current training set.
     * @return 0 if successful
     */
    int recalculate_stddev();

public:
    /**
     * Constructor. Loads existing training database from CSV.
     * @param db_path path to feature database CSV file
     */
    Classifier(fs::path db_path);

    /**
     * Appends newly collected training examples to CSV file.
     * Only writes examples collected during current session.
     * @return 0 if successful
     */
    int write_new_trained_data();

    /**
     * Interactive training mode. Displays each region, prompts for labels,
     * and stores feature vectors with labels. Supports new label creation.
     * @param orig_img original color frame for display
     * @param label_map region label map from segmentation
     * @param regions detected regions with computed features
     * @return 0 if successful
     */
    int train_on_all_segments(cv::Mat &orig_img, cv::Mat &label_map, std::vector<RegionStats> &regions);

    /**
     * Classifies regions using nearest-neighbor with scaled Euclidean distance.
     * Assigns labels based on closest match in training database.
     * @param regions detected regions (labels stored in each RegionStats)
     * @return 0 if successful
     */
    int predict(std::vector<RegionStats> &regions);

    /**
     * Checks if training database has any examples.
     * @return true if training data exists
     */
    bool has_training_data() const;
};

#endif //FEATURE_H
