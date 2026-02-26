//
// Created by Ajey K on 21/02/26.
// Header file for Threshold module - Implements adaptive thresholding methods
// for foreground-background separation including k-means clustering and background subtraction.
//

#ifndef THRESHOLD_H
#define THRESHOLD_H

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

#include <functional>

/**
 * Function pointer type for thresholding callback functions.
 * Takes source image and outputs binary image.
 */
typedef std::function<int(cv::Mat &, cv::Mat &)> ThresholdCallback;

// K-means algorithm parameters
inline const int kmeans_max_iter = 10; // Maximum iterations for k-means convergence
inline const int kmeans_stop_threshold = 1; // Convergence threshold (squared color distance)

// White background validation parameters
inline const int white_highness = 245; // Minimum brightness for valid white background
inline const int white_closeness = 5; // Maximum color channel difference for neutral white
inline const int sq_diff_with_white = 60; // Squared distance threshold for foreground classification

// Preprocessing parameters
inline const int blurring_kernel_size = 1; // Blur kernel size (1 = no blur)
inline const int gaussian_blur_kernel_size = 7; // Gaussian blur kernel size
inline const std::string bg_display_window_title = "Captured Bacground - Confirm satisfied? <y/n>";

/**
 * Threshold module implementing adaptive thresholding methods.
 * Supports dynamic k-means clustering and white background subtraction.
 */
class Threshold {
    cv::Mat bg; // Reference background color for white screen method

    /**
     * Dynamic k-means thresholding (mode 0).
     * Samples 1/16 of pixels and clusters into foreground/background using k-means.
     * @param src input color image
     * @param dst output binary image (foreground = 255, background = 0)
     * @return 0 if successful
     */
    int dynamic_threshold(cv::Mat &src, cv::Mat &dst);

    /**
     * White background subtraction thresholding (mode 1).
     * Classifies pixels by color distance from captured background reference.
     * @param src input color image
     * @param dst output binary image
     * @return 0 if successful
     */
    int white_screen_threshold(cv::Mat &src, cv::Mat &dst);

    /**
     * Dispatch map for selecting thresholding method based on mode.
     */
    const std::map<int, ThresholdCallback> callbacks = {
        {0, [this](cv::Mat &src, cv::Mat &dst) { return dynamic_threshold(src, dst); }},
        {1, [this](cv::Mat &src, cv::Mat &dst) { return white_screen_threshold(src, dst); }}
    };

public:
    /**
     * Default constructor.
     */
    Threshold() {
    };

    /**
     * Captures white background reference color for background subtraction method.
     * Validates background is bright and color-neutral before accepting.
     * @param src input frame of empty white platform
     * @return true if valid white background captured, false otherwise
     */
    bool pickup_white_screen(const cv::Mat &src);

    /**
     * Main thresholding dispatcher. Applies Gaussian blur then calls
     * appropriate thresholding method based on mode.
     * @param src input color image
     * @param dst output binary image
     * @param mode thresholding method (0 = k-means, 1 = background subtraction)
     * @return 0 if successful
     */
    int threshold(cv::Mat &src, cv::Mat &dst, const int mode = 0);
};

#endif //THRESHOLD_H
