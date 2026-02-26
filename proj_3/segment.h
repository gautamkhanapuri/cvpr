//
// Created by Ajey K on 21/02/26.
// Header file for Segmentation and Region Tracking modules.
// Implements connected components analysis, region property computation, and temporal tracking.
//

#ifndef SEGMENT_H
#define SEGMENT_H

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <cmath>
#include <iostream>

// Minimum region area (pixels) for object detection - filters out noise
inline const int min_area_pix = 4000;

// Tolerance thresholds for region tracking across frames
inline const float acceptable_centroid_x_ch = 5.0; // Maximum centroid x-shift (pixels)
inline const float acceptable_centroid_y_ch = 5.0; // Maximum centroid y-shift (pixels)
inline const float acceptable_box_width_ch = 5.0; // Maximum width change (pixels)
inline const float acceptable_box_height_ch = 5.0; // Maximum height change (pixels)
inline const float acceptable_box_angle_ch = 5.0; // Maximum angle change (degrees)

/**
 * Extent values along principal and secondary axes.
 * Used for oriented bounding box creation and ResNet preprocessing.
 */
struct AxisExtent {
    float minE1; // Minimum extent along major axis (negative value)
    float maxE1; // Maximum extent along major axis (positive value)
    float minE2; // Minimum extent along minor axis (negative value)
    float maxE2; // Maximum extent along minor axis (positive value)
};

/**
 * Complete statistics and properties for a detected region.
 * Stores geometric, moment-based, and classification information.
 */
struct RegionStats {
    int region_id; // Region label from connected components
    cv::Moments moments; // Image moments (spatial, central, normalized)
    cv::Point centroid; // Center of mass
    float angle; // Principal axis angle in radians
    cv::RotatedRect oriented_box; // Oriented bounding box
    std::vector<std::vector<cv::Point> > contours; // Region boundary contours
    std::vector<float> features; // Feature vector (7 values)
    std::string label; // Predicted label from hand-crafted features
    std::string dnn_label; // Predicted label from ResNet embeddings
    cv::Scalar color; // Display color (stable across frames)
    float confidence; // Classification confidence
    AxisExtent axisExtent; // Object extents along principal axes
};

/**
 * Segmentation module using connected components analysis.
 * Extracts regions, computes moments, principal axes, and extent values.
 */
class Segment {
public:
    Segment() {
    };

    /**
     * Performs connected components analysis and computes region properties.
     * Filters small regions, computes moments, principal axes, and extents.
     * @param src binary input image (foreground = 255)
     * @param label_map output region map (each region has unique ID)
     * @param regions output vector of region statistics
     * @return 0 if successful
     */
    int make_segments(const cv::Mat &src, cv::Mat &label_map, std::vector<RegionStats> &regions);

    /**
     * Computes actual object extents by projecting all region pixels onto principal axes.
     * Provides precise measurements for asymmetric or irregular shapes.
     * @param label_map region label map
     * @param region_id ID of region to analyze
     * @param centroid center of mass
     * @param angle_rad principal axis angle in radians
     * @return AxisExtent with min/max projections along both axes
     */
    AxisExtent compute_extents(const cv::Mat &label_map, int region_id, const cv::Point2f &centroid, float angle_rad);

    /**
     * Creates oriented bounding box from computed extents.
     * Determines box center accounting for asymmetric mass distribution.
     * @param centroid center of mass
     * @param angle_rad principal axis angle
     * @param ext axis extents
     * @return RotatedRect with correct center, size, and orientation
     */
    cv::RotatedRect create_obb_from_extents(const cv::Point2f &centroid, float angle_rad, const AxisExtent &ext);
};

/**
 * Region tracking and visualization module.
 * Maintains region identity across frames and draws overlays.
 */
class RegionTracker {
    std::vector<RegionStats> old_regions; // Regions from previous frame for tracking

    /**
     * Finds matching region from previous frame based on proximity and similarity.
     * Matches by centroid position, size, and orientation.
     * @param reg current frame region to match
     * @return 0 if match found (color assigned), 1 if new region
     */
    int find_closest_prev(RegionStats &reg);

    /**
     * Draws oriented bounding box on frame.
     * @param src frame to draw on
     * @param oriented_box rotated rectangle to draw
     * @param colour line color
     * @return 0 if successful
     */
    int draw_oriented_box(cv::Mat &src, const cv::RotatedRect &oriented_box, const cv::Scalar &colour);

    /**
     * Draws principal and secondary axes through centroid.
     * Uses computed extents for accurate axis lengths.
     * @param src frame to draw on
     * @param cent centroid point
     * @param angle_rad principal axis angle
     * @param oriented_box bounding box (for reference)
     * @param colour line color
     * @return 0 if successful
     */
    int draw_axes(cv::Mat &src, const cv::Point2f &cent, float angle_rad, const cv::RotatedRect &oriented_box,
                  const cv::Scalar &colour);

public:
    RegionTracker() {
    };

    /**
     * Matches regions to previous frame for color stability, then draws overlays.
     * Assigns persistent colors and draws bounding boxes with axes.
     * @param src frame to draw on
     * @param new_regions current frame regions
     * @return 0 if successful
     */
    int match_and_draw_box(cv::Mat &src, std::vector<RegionStats> &new_regions);
};

#endif //SEGMENT_H
