//
// Created by Ajey K on 23/02/26.
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

inline const std::string training_img_display_window_name = "Training Image Display. 'p' - skip region / 'a' - add label";
inline const int num_features = 7;
inline const float label_matching_threshold = 1.75;
inline const std::string unknown = "Unknown";

class Feature {

public:
    Feature() {};
    int calculate_basic_2d_features(std::vector<RegionStats>& regions);
    int overlay_features(cv::Mat& src, std::vector<RegionStats>& regions);
};

class Classifier {
    fs::path db_path;
    std::vector<std::string> labels;
    std::vector<std::vector<float>> features;
    std::vector<float> average;
    std::vector<float> stddev;
    int initial_datapoints_count;
    std::set<std::string> labels_set;
    int load_db_file();
    bool is_valid_label(const std::string &label) const;
    int register_new_label(const std::string &label);
    int recalculate_average();
    int recalculate_stddev();

    public:
    Classifier(fs::path db_path);
    int write_new_trained_data();
    int train_on_all_segments(cv::Mat& orig_img, cv::Mat& label_map, std::vector<RegionStats>& regions);
    int predict(std::vector<RegionStats>& regions);
    bool has_training_data() const;
};

#endif //FEATURE_H
