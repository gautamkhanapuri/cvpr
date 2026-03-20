//
// Created by Gautam Khanapuri on 10th March 2026
// Header file for AR module - handles real-time pose estimation and 3D object projection.
// Projects virtual objects onto checkerboard target using calibrated camera parameters.
//

#ifndef AR_H
#define AR_H

#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>

#include <iostream>
#include <string>

#include "calibrate.h"

// Length of displayed 3D axes in checkerboard units
inline const int axis_length = 3;

// Line connectivity for rocket object (8 base points + 1 apex + 3 fins)
inline const std::vector<std::pair<int, int>> rocket_lines = {
    {0,1}, {1,2}, {2,3}, {3,4}, {4,5}, {5,6}, {6,7}, {7,0},  // Base circle
    {0,8}, {2,8}, {4,8}, {6,8},  // To nose cone
    {2,9}, {9,3}, {5,10}, {10,6}, {7,11}, {11,0}  // Fins
};

/**
 * AR module for real-time augmented reality.
 * Estimates camera pose using solvePNP and projects virtual 3D objects onto checkerboard.
 */
class AR {
private:
    bool axes;              // Toggle for 3D axes display
    bool display_1;         // Toggle for DNA helix
    bool display_2;         // Toggle for pyramid
    bool display_3;         // Toggle for spiral
    cv::Mat calibration_matrix;     // Camera intrinsic parameters (K)
    cv::Mat distortion_matrix;      // Lens distortion coefficients
    cv::Mat rotation_matrix;        // Current camera rotation (R) from solvePNP
    cv::Mat translation_matrix;     // Current camera translation (t) from solvePNP
    std::chrono::time_point<std::chrono::steady_clock> start_time;  // For time-based animation

    /**
     * Draws 3D coordinate axes at checkerboard origin.
     * X-axis (red), Y-axis (green), Z-axis (blue).
     * @param src frame to draw on
     */
    void draw_3d_axes(cv::Mat& src);

    /**
     * Draws circles at the four outer corners of checkerboard for verification.
     * @param src frame to draw on
     * @param cols number of checkerboard columns
     * @param rows number of checkerboard rows
     */
    void draw_corner_points(cv::Mat& src, int cols=checkboard_width, int rows=checkboard_height);

    /**
     * Creates 3D spiral staircase structure.
     * @return vector of 3D points defining spiral
     */
    std::vector<cv::Vec3f> create_spiral();

    /**
     * Generates line connectivity for spiral staircase.
     * @return pairs of point indices to connect
     */
    std::vector<std::pair<int, int>> spiral_lines();

    /**
     * Creates DNA double helix base structure (before rotation/incline).
     * Two intertwined strands with base pair connection points.
     * @return vector of 3D points (4 points per level: strand1, strand2, inner1, inner2)
     */
    std::vector<cv::Vec3f> create_dna_helix();

    /**
     * Creates wireframe rocket with base, nose cone, and fins.
     * @return vector of 3D points defining rocket geometry
     */
    std::vector<cv::Vec3f> create_rocket();

    /**
     * Draws DNA double helix with time-based rotation and spatial incline.
     * Applies transformations, projects to 2D, and renders with multiple colors.
     * @param src frame to draw on
     * @param time_seconds elapsed time for rotation animation
     */
    void draw_dna_special(cv::Mat& src, double time_seconds);

    /**
     * Generic 3D object renderer. Projects points and draws connecting lines.
     * @param src frame to draw on
     * @param points_3d 3D world coordinates of object vertices
     * @param lines pairs of indices defining edges
     * @param color line color
     */
    void draw_3d_object(cv::Mat& src, const std::vector<cv::Vec3f>& points_3d,
                   const std::vector<std::pair<int, int>>& lines, cv::Scalar color);

    /**
     * Draws AR mode UI overlay showing calibration parameters and pose matrices.
     * @param src frame to draw on
     * @param found whether checkerboard was detected in current frame
     * @return 0 if successful
     */
    int draw_ar_ui(cv::Mat& src, bool found);

public:
    /**
     * Constructor. Initializes AR module and animation timer.
     */
    AR();

    /**
     * Main display function. Renders enabled objects and UI overlay.
     * @param src frame to render on
     * @return 0 if successful
     */
    int display_objects(cv::Mat& src);

    /**
     * Sets camera calibration matrix from calibration module.
     * @param calib_mat 3×3 intrinsic parameter matrix
     * @return 0 if successful
     */
    int set_calibration_matrix(const cv::Mat& calib_mat);

    /**
     * Sets distortion coefficients from calibration module.
     * @param dist_mat distortion coefficient vector
     * @return 0 if successful
     */
    int set_distortion_matrix(const cv::Mat& dist_mat);

    /**
     * Toggles 3D axes display.
     * @return 0 if successful
     */
    int show_axes();

    /**
     * Toggles DNA helix display.
     * @return 0 if successful
     */
    int ar_object1();

    /**
     * Toggles pyramid display.
     * @return 0 if successful
     */
    int ar_object2();

    /**
     * Toggles spiral display.
     * @return 0 if successful
     */
    int ar_object3();
};

#endif //AR_H
