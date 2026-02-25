//
// Created by Ajey K on 21/02/26.
//

#ifndef RTOR_H
#define RTOR_H

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include <iostream>
#include <string>
#include <filesystem>
#include <cstdlib>

#include "threshold.h"
#include "morph.h"
#include "segment.h"
#include "feature.h"

namespace fs = std::filesystem;

inline const std::string window1 = "Primary Window-Real Time Object Recognition";
inline const std::string threshold_binary_window = "Thresholding-Binary Image";
inline const std::string cleaned_morph_window = "Morphology-Cleaned up image";
inline const int starter_threshold_mode = 0;
inline const std::string main_image_filename = "original_";
inline const std::string overlay_image_filename = "overlayed_";
inline const std::string threshold_image_filename = "threshold_";
inline const std::string morphed_image_filename = "morphed_";
inline const std::string image_save_format = ".png";

class RTObectRecognizer {
    cv::VideoCapture* vd_cap;
    int device_id;
    int api_id;
    cv::Size refS;
    int pressed;

    int threshold_mode;

    bool show_binary;
    bool show_morphology;

    cv::Mat main_frame;
    cv::Mat display_frame;
    bool white_screen_set;
    cv::Mat bin_frame;
    cv::Mat morph_frame;
    cv::Mat label_map;

    const fs::path& db_filepath;

    Threshold threshold;
    Segment segment;
    RegionTracker tracker;
    Feature feature;
    Classifier classifier;

    int vid_setup();
    int handle_key(int key, std::vector<RegionStats>& regions);

  public:
    RTObectRecognizer(const fs::path& db_filepath);
    ~RTObectRecognizer();
    int run();
};

#endif //RTOR_H
