//
// Created by Ajey K on 21/02/26.
// Header file for RTObectRecognizer - Main orchestrator class for real-time object recognition.
// Manages video capture, processing pipeline, feature extraction, classification, and display.
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
#include "resnetclassifier.h"

namespace fs = std::filesystem;

// Window titles for various display outputs
inline const std::string window1 = "Primary Window-Real Time Object Recognition";
inline const std::string threshold_binary_window = "Thresholding-Binary Image";
inline const std::string cleaned_morph_window = "Morphology-Cleaned up image";

// Default thresholding mode (0 = k-means, 1 = white background subtraction)
inline const int starter_threshold_mode = 0;

// Filename prefixes for saving captured frames
inline const std::string main_image_filename = "original_";
inline const std::string overlay_image_filename = "overlayed_";
inline const std::string threshold_image_filename = "threshold_";
inline const std::string morphed_image_filename = "morphed_";
inline const std::string image_save_format = ".png";

/**
 * Main orchestrator class for real-time object recognition system.
 * Manages video capture, processing pipeline (threshold → morph → segment → features → classify),
 * training modes, and real-time visualization with multiple display options.
 */
class RTObectRecognizer {
    cv::VideoCapture *vd_cap; // Video capture device
    int device_id; // Camera device ID
    int api_id; // Video capture API ID
    cv::Size refS; // Reference frame size
    int pressed; // Last key pressed by user

    int threshold_mode; // Current thresholding method (0 or 1)

    bool show_binary; // Toggle for binary threshold display
    bool show_morphology; // Toggle for morphology display

    cv::Mat main_frame; // Original color frame from camera
    cv::Mat display_frame; // Frame with overlays for display
    bool white_screen_set; // Flag for white background captured
    cv::Mat bin_frame; // Binary thresholded image
    cv::Mat morph_frame; // Morphologically cleaned image
    cv::Mat label_map; // Region labels from connected components

    const fs::path &db_filepath; // Path to hand-crafted features database

    Threshold threshold; // Thresholding module
    Segment segment; // Segmentation module
    RegionTracker tracker; // Region tracking and visualization
    Feature feature; // Feature extraction module
    Classifier classifier; // Hand-crafted feature classifier

    ResNetClassifier *resnet; // ResNet embedding classifier (optional)
    fs::path dnn_db_filepath; // Path to ResNet embeddings database
    bool resnet_available; // Flag for ResNet model loaded successfully
    bool classify_with_resnet; // Toggle for ResNet predictions

    /**
     * Initializes video capture device and configures camera parameters.
     * @return 0 if successful
     */
    int vid_setup();

    /**
     * Loads ResNet18 model and initializes embedding classifier.
     * @return 0 if successful
     */
    int resnet_setup();

    /**
     * Handles keyboard input for training, display toggles, and mode switching.
     * @param key pressed key code
     * @param regions detected regions in current frame
     * @return 0 if successful
     */
    int handle_key(int key, std::vector<RegionStats> &regions);

public:
    /**
     * Constructor for RTObectRecognizer.
     * @param db_filepath path to hand-crafted features database CSV
     * @param dnn_db_filepath path to ResNet embeddings database CSV
     */
    RTObectRecognizer(const fs::path &db_filepath, const fs::path &dnn_db_filepath);

    /**
     * Destructor. Saves training data and releases resources.
     */
    ~RTObectRecognizer();

    /**
     * Main processing loop. Captures frames, runs pipeline, displays results,
     * and handles user input until 'q' is pressed.
     * @return 0 on successful exit
     */
    int run();
};

#endif //RTOR_H
