//
// Created by Ajey K on 21/02/26.
//

#ifndef SEGMENT_H
#define SEGMENT_H

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <cmath>
#include <iostream>

inline const int min_area_pix = 9000;
inline const float acceptable_centroid_x_ch = 5.0;
inline const float acceptable_centroid_y_ch = 5.0;
inline const float acceptable_box_width_ch  = 5.0;
inline const float acceptable_box_height_ch = 5.0;
inline const float acceptable_box_angle_ch  = 5.0;

struct RegionStats {
    int region_id;
    cv::Moments moments;
    cv::Point centroid;
    float angle;  // In degrees
    cv::RotatedRect oriented_box;
    std::vector<std::vector<cv::Point>> contours;
    std::vector<float> features;
    std::string label;
    cv::Scalar color;
    float confidence;
};

class Segment {

    public:
    Segment() {};
    int make_segments(const cv::Mat& src, cv::Mat& label_map, std::vector<RegionStats>& regions);
    // int get_obb(cv::Mat& src, int region_id, RegionStats& reg);
};

class RegionTracker {
    std::vector<RegionStats> old_regions;
    int find_closest_prev(RegionStats& reg);
    int draw_oriented_box(cv::Mat& src, const cv::RotatedRect& oriented_box, const cv::Scalar& colour);
    int draw_axes(cv::Mat& src, const cv::Point2f& cent, float angle_rad, const cv::RotatedRect& oriented_box, const cv::Scalar& colour);
    public:
    RegionTracker() {};
    int match_and_draw_box(cv::Mat& src, std::vector<RegionStats>& new_regions);
};

#endif //SEGMENT_H
