//
// Created by Ajey K on 21/02/26.
//

#include "threshold.h"

int Threshold::dynamic_threshold(cv::Mat &src, cv::Mat &dst) {
    std::vector<cv::Vec3b> pixels;
    for (int i = 0; i < src.rows; i+=4) {
        cv::Vec3b* ptr = src.ptr<cv::Vec3b>(i);
        for (int j = 0; j < src.cols; j+=4) {
            pixels.push_back(ptr[j]);
        }
    }

    // K means
    cv::Vec3b mean1(0, 0, 0);  // Center of fg
    cv::Vec3b mean2(255, 255, 255);  // Center of bg

    for (int i = 0; i < kmeans_max_iter; i++) {
        float cluster1_b = 0.0, cluster2_b = 0.0;
        float cluster1_g = 0.0, cluster2_g = 0.0;
        float cluster1_r = 0.0, cluster2_r = 0.0;
        int cluster1_count = 0;
        int cluster2_count = 0;

        for (cv::Vec3b& pt : pixels) {
            long dist_from_mean1 = (mean1[0] - pt[0]) * (mean1[0] - pt[0]) + (mean1[1] - pt[1]) * (mean1[1] - pt[1]) + (mean1[2] - pt[2]) * (mean1[2] - pt[2]);
            long dist_from_mean2 = (mean2[0] - pt[0]) * (mean2[0] - pt[0]) + (mean2[1] - pt[1]) * (mean2[1] - pt[1]) + (mean2[2] - pt[2]) * (mean2[2] - pt[2]);
            if (dist_from_mean1 <= dist_from_mean2) {
                cluster1_b += pt[0];
                cluster1_g += pt[1];
                cluster1_r += pt[2];
                cluster1_count++;
            } else {
                cluster2_b += pt[0];
                cluster2_g += pt[1];
                cluster2_r += pt[2];
                cluster2_count++;
            }
        }
        auto c1_b = static_cast<uchar>(cluster1_b / cluster1_count);
        auto c1_g = static_cast<uchar>(cluster1_g / cluster1_count);
        auto c1_r = static_cast<uchar>(cluster1_r / cluster1_count);

        auto c2_b = static_cast<uchar>(cluster2_b / cluster2_count);
        auto c2_g = static_cast<uchar>(cluster2_g / cluster2_count);
        auto c2_r = static_cast<uchar>(cluster2_r / cluster2_count);

        cv::Vec3b new_mean1 = cv::Vec3b(c1_b, c1_g, c1_r);
        cv::Vec3b new_mean2 = cv::Vec3b(c2_b, c2_g, c2_r);

        long ch1 = (mean1[0] - new_mean1[0]) * (mean1[0] - new_mean1[0]) + (mean1[1] - new_mean1[1]) * (mean1[1] - new_mean1[1]) + (mean1[2] - new_mean1[2]) * (mean1[2] - new_mean1[2]);
        long ch2 = (mean2[0] - new_mean2[0]) * (mean2[0] - new_mean2[0]) + (mean2[1] - new_mean2[1]) * (mean2[1] - new_mean2[1]) + (mean2[2] - new_mean2[2]) * (mean2[2] - new_mean2[2]);
        bool m1_stop = ch1 <= kmeans_stop_threshold;
        bool m2_stop = ch2 <= kmeans_stop_threshold;
        mean1 = new_mean1;
        mean2 = new_mean2;
        // std::cout << cluster1_count << " " << cluster2_count << std::endl;
        if (m1_stop && m2_stop) {
            std::cout << "Kmeans converged in " << i << " iterations" << std::endl;
            break;
        }
    }

    std::cout << "FG Center: (" << (int)mean1[0] << ", " << (int)mean1[1] << ", " << (int)mean1[2] << ")" << std::endl;
    std::cout << "BG Center: (" << (int)mean2[0] << ", " << (int)mean2[1] << ", " << (int)mean2[2] << ")" << std::endl;

    dst.create(src.rows, src.cols, CV_8UC1);
    for (int i = 0; i < src.rows; i++) {
        cv::Vec3b* ptr = src.ptr<cv::Vec3b>(i);
        uchar* dst_ptr = dst.ptr<uchar>(i);
        for (int j = 0; j < src.cols; j++) {
            long dist_1 = (mean1[0] - ptr[j][0]) * (mean1[0] - ptr[j][0]) + (mean1[1] - ptr[j][1]) * (mean1[1] - ptr[j][1]) + (mean1[2] - ptr[j][2]) * (mean1[2] - ptr[j][2]);
            long dist_2 = (mean2[0] - ptr[j][0]) * (mean2[0] - ptr[j][0]) + (mean2[1] - ptr[j][1]) * (mean2[1] - ptr[j][1]) + (mean2[2] - ptr[j][2]) * (mean2[2] - ptr[j][2]);
            if (dist_1 <= dist_2) {
                // If dist_1 <= dist_2  ==> It means the pt is closer to mean1 than mean2.
                // mean1 was the center for black/darker objects. So it is a part of the fg.
                // Assign 255
                dst_ptr[j] = 255;
            } else {
                // Here dist2 > dist1. ==> Closer to mean2 than mean1.
                // mean2 was white when initialized, so it will be the center of background pixels.
                // Assign 0
                dst_ptr[j] = 0;
            }
        }
    }
    // std::cout << "Thresholding done!" << std::endl;
    return 0;
}

int Threshold::white_screen_threshold(cv::Mat &src, cv::Mat &dst) {
    dst.create(src.rows, src.cols, CV_8UC1);
    for (int i = 0; i < src.rows; i++) {
        cv::Vec3b* ptr = src.ptr<cv::Vec3b>(i);
        uchar* dst_ptr = dst.ptr<uchar>(i);
        for (int j = 0; j < src.cols; j++) {
            long dist_from_white = (this->bg_white[0] - ptr[j][0]) * (this->bg_white[0] - ptr[j][0]) + (this->bg_white[1] - ptr[j][1]) * (this->bg_white[1] - ptr[j][1]) + (this->bg_white[2] - ptr[j][2]) * (this->bg_white[2] - ptr[j][2]);
            if (dist_from_white >= sq_diff_with_white) {
                dst_ptr[j] = 255;
            } else {
                dst_ptr[j] = 0;
            }
        }
    }
    return 0;
}

bool Threshold::pickup_white_screen(cv::Mat &src) {
    float b = 0.0;
    float g = 0.0;
    float r = 0.0;
    int count = 0;

    for (int i = 0; i < src.rows; i+=10) {
        cv::Vec3b* ptr = src.ptr<cv::Vec3b>(i);
        for (int j = 0; j < src.cols; j+=10) {
            b += ptr[j][0];
            g += ptr[j][1];
            r += ptr[j][2];
            count++;
        }
    }
    b /= count;
    g /= count;
    r /= count;

    bool high_enough = (b >=white_highness) && (g >=white_highness) && (r >=white_highness);
    float max_diff = std::max({std::abs(b-g), std::abs(r-g), std::abs(b-r)});
    bool close_enough = (max_diff <= white_closeness);
    if (high_enough && close_enough) {
        auto b_ = static_cast<uchar>(b);
        auto g_ = static_cast<uchar>(g);
        auto r_ = static_cast<uchar>(r);
        this->bg_white = cv::Vec3b(b_, g_, r_);
        return true;
    }
    return false;
}

int Threshold::threshold(cv::Mat &src, cv::Mat &dst) {
    cv::blur(src, src, cv::Size(blurring_kernel_size, blurring_kernel_size));
    std::map<int, ThresholdCallback>::const_iterator it = this->callbacks.find(this->mode);
    if (it == this->callbacks.end()) {
        return dynamic_threshold(src, dst);
    }
    return it->second(src, dst);
}

