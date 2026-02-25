//
// Created by Ajey K on 21/02/26.
//

#ifndef THRESHOLD_H
#define THRESHOLD_H

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

#include <functional>

typedef std::function<int(cv::Mat&, cv::Mat&)> ThresholdCallback;
inline const int kmeans_max_iter = 10;
inline const int kmeans_stop_threshold = 1;
inline const int white_highness = 245;
inline const int white_closeness = 5;
inline const int sq_diff_with_white = 60;
inline const int blurring_kernel_size = 1;
inline const int gaussian_blur_kernel_size = 7;
inline const std::string bg_display_window_title = "Captured Bacground - Confirm satisfied? <y/n>";

class Threshold {
    cv::Mat bg;
    int dynamic_threshold(cv::Mat& src, cv::Mat& dst);
    int white_screen_threshold(cv::Mat& src, cv::Mat& dst);
    const std::map<int, ThresholdCallback> callbacks = {
    {0, [this](cv::Mat& src, cv::Mat& dst) {return dynamic_threshold(src, dst);} },
    {1, [this](cv::Mat& src, cv::Mat& dst) {return white_screen_threshold(src, dst);} }
    };

    public:
    Threshold() {};
    bool pickup_white_screen(const cv::Mat& src);
    int threshold(cv::Mat& src, cv::Mat& dst, const int mode=0);

};
#endif //THRESHOLD_H
