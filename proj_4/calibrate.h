//
// Created by Gautam Khanapuri on 10th March 2026
// Header file for Calibrate module - implements camera calibration using checkerboard target.
// Collects calibration images, estimates intrinsic parameters, and manages calibration data persistence.
//

#ifndef CALIBRATE_H
#define CALIBRATE_H

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/types_c.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>

#include <filesystem>
#include <iostream>
#include <string>
#include <fstream>

#include "utils.h"

namespace fs = std::filesystem;

// Checkerboard dimensions (internal corners)
inline const int checkboard_width = 9;   // Columns
inline const int checkboard_height = 6;  // Rows

// ArUco marker dimensions
inline const int Aruco_width = 6;
inline const int Aruco_height = 5;

// Minimum images required before calibration can be performed
inline const int min_calibration_images_required = 5;

// Number of distortion coefficients to estimate (0, 4, 5, or 8)
inline const int distortion_matrix_size = 5;

// World coordinate units (1 = one checkerboard square)
inline const int square_size = 1;

// Filename prefix for saved calibration images
inline const std::string calibration_file_name = "Calibration_";
inline const std::string image_save_format = ".png";

/**
 * World coordinates for 9×6 checkerboard (54 internal corners).
 * Order matches findChessboardCorners output: left-to-right, then top-to-bottom.
 * Coordinates: (col, -row, 0) with origin at top-left, Y-down = negative.
 */
inline const std::vector<cv::Vec3f> chessboard_points_9x6 = {
//     {0.0, 0.0, 0.0},
// {1.0, 0.0, 0.0},
// {2.0, 0.0, 0.0},
// {3.0, 0.0, 0.0},
// {4.0, 0.0, 0.0},
// {5.0, 0.0, 0.0},
// {6.0, 0.0, 0.0},
// {7.0, 0.0, 0.0},
// {8.0, 0.0, 0.0},
// {0.0, -1.0, 0.0},
// {1.0, -1.0, 0.0},
// {2.0, -1.0, 0.0},
// {3.0, -1.0, 0.0},
// {4.0, -1.0, 0.0},
// {5.0, -1.0, 0.0},
// {6.0, -1.0, 0.0},
// {7.0, -1.0, 0.0},
// {8.0, -1.0, 0.0},
// {0.0, -2.0, 0.0},
// {1.0, -2.0, 0.0},
// {2.0, -2.0, 0.0},
// {3.0, -2.0, 0.0},
// {4.0, -2.0, 0.0},
// {5.0, -2.0, 0.0},
// {6.0, -2.0, 0.0},
// {7.0, -2.0, 0.0},
// {8.0, -2.0, 0.0},
// {0.0, -3.0, 0.0},
// {1.0, -3.0, 0.0},
// {2.0, -3.0, 0.0},
// {3.0, -3.0, 0.0},
// {4.0, -3.0, 0.0},
// {5.0, -3.0, 0.0},
// {6.0, -3.0, 0.0},
// {7.0, -3.0, 0.0},
// {8.0, -3.0, 0.0},
// {0.0, -4.0, 0.0},
// {1.0, -4.0, 0.0},
// {2.0, -4.0, 0.0},
// {3.0, -4.0, 0.0},
// {4.0, -4.0, 0.0},
// {5.0, -4.0, 0.0},
// {6.0, -4.0, 0.0},
// {7.0, -4.0, 0.0},
// {8.0, -4.0, 0.0},
// {0.0, -5.0, 0.0},
// {1.0, -5.0, 0.0},
// {2.0, -5.0, 0.0},
// {3.0, -5.0, 0.0},
// {4.0, -5.0, 0.0},
// {5.0, -5.0, 0.0},
// {6.0, -5.0, 0.0},
// {7.0, -5.0, 0.0},
// {8.0, -5.0, 0.0}
   {8.0, -5.0, 0.0},
{7.0, -5.0, 0.0},
{6.0, -5.0, 0.0},
{5.0, -5.0, 0.0},
{4.0, -5.0, 0.0},
{3.0, -5.0, 0.0},
{2.0, -5.0, 0.0},
{1.0, -5.0, 0.0},
{0.0, -5.0, 0.0},
{8.0, -4.0, 0.0},
{7.0, -4.0, 0.0},
{6.0, -4.0, 0.0},
{5.0, -4.0, 0.0},
{4.0, -4.0, 0.0},
{3.0, -4.0, 0.0},
{2.0, -4.0, 0.0},
{1.0, -4.0, 0.0},
{0.0, -4.0, 0.0},
{8.0, -3.0, 0.0},
{7.0, -3.0, 0.0},
{6.0, -3.0, 0.0},
{5.0, -3.0, 0.0},
{4.0, -3.0, 0.0},
{3.0, -3.0, 0.0},
{2.0, -3.0, 0.0},
{1.0, -3.0, 0.0},
{0.0, -3.0, 0.0},
{8.0, -2.0, 0.0},
{7.0, -2.0, 0.0},
{6.0, -2.0, 0.0},
{5.0, -2.0, 0.0},
{4.0, -2.0, 0.0},
{3.0, -2.0, 0.0},
{2.0, -2.0, 0.0},
{1.0, -2.0, 0.0},
{0.0, -2.0, 0.0},
{8.0, -1.0, 0.0},
{7.0, -1.0, 0.0},
{6.0, -1.0, 0.0},
{5.0, -1.0, 0.0},
{4.0, -1.0, 0.0},
{3.0, -1.0, 0.0},
{2.0, -1.0, 0.0},
{1.0, -1.0, 0.0},
{0.0, -1.0, 0.0},
{8.0, 0.0, 0.0},
{7.0, 0.0, 0.0},
{6.0, 0.0, 0.0},
{5.0, 0.0, 0.0},
{4.0, 0.0, 0.0},
{3.0, 0.0, 0.0},
{2.0, 0.0, 0.0},
{1.0, 0.0, 0.0},
{0.0, 0.0, 0.0}
};

/**
 * Calibration module for camera parameter estimation.
 * Detects checkerboard corners, collects calibration images, and computes intrinsic parameters.
 */
class Calibrate {
private:
    cv::Size frame_size;                        // Video frame dimensions
    fs::path calibration_file;                  // Path to calibration data CSV
    cv::Mat detected_frame;                     // Most recent frame with detected corners
    cv::Mat calibration_matrix;                 // Camera matrix K (3×3)
    cv::Mat distortion_matrix;                  // Distortion coefficients (5 values)
    double error;                               // Reprojection error from calibration

    int chessboard_width;                       // Checkerboard columns
    int chessboard_height;                      // Checkerboard rows
    int aruco_width;                            // ArUco marker array width
    int aruco_height;                           // ArUco marker array height

    int points_found;                           // Number of corners detected in current frame
    int mode;                                   // Target type: 1=checkerboard, 0=ArUco
    bool is_calibrated;                         // Flag for valid calibration available
    int calibrated_images_count;                // Number of calibration images collected

    std::vector<std::vector<cv::Vec3f>> point_list;     // 3D world coordinates for all saved images
    std::vector<cv::Point2f> corners;                   // Most recent detected 2D corners
    std::vector<std::vector<cv::Point2f>> corner_list;  // 2D corners for all saved images

    /**
     * Initializes calibration matrix with default values.
     * Sets focal lengths to frame width and principal point to frame center.
     * @return 0 if successful
     */
    int set_default_calibration_matrix();

    /**
     * Draws calibration mode UI overlay showing status and parameters.
     * @param src frame to draw on
     * @return 0 if successful
     */
    int draw_calibration_ui(cv::Mat& src);

public:
    /**
     * Constructor. Initializes calibration module and loads existing data if available.
     * @param calibration_file path to calibration data CSV
     */
    Calibrate(const fs::path& calibration_file);

    /**
     * Reads calibration matrix and distortion coefficients from file.
     * @param frame_size video frame dimensions for validation
     * @return 0 if successful, non-zero if file empty or invalid
     */
    int read_calibration_data(cv::Size frame_size);

    /**
     * Writes current calibration matrix and distortion coefficients to CSV file.
     * @return 0 if successful
     */
    int write_calibration_data();

    /**
     * Detects checkerboard corners in frame and draws overlay if found.
     * Updates internal corner storage for potential calibration image save.
     * @param src input frame (corners drawn on this frame)
     * @return 0 if successful
     */
    int detect_and_mark_corners(cv::Mat& src);

    /**
     * Returns reference to camera calibration matrix.
     * @return 3×3 calibration matrix (K)
     */
    cv::Mat& get_calibration_matrix();

    /**
     * Returns reference to distortion coefficients.
     * @return distortion coefficient vector
     */
    cv::Mat& get_distortion_matrix();

    /**
     * Checks if valid calibration data exists.
     * @return true if calibration completed
     */
    bool is_calibration_matrix_available() const;

    /**
     * Saves current detected corners as calibration image.
     * Automatically triggers calibration after minimum images collected.
     * Recalibrates on each subsequent save.
     * @return 0 if successful, 1 if no corners detected
     */
    int save_calibration_image();

    /**
     * Returns number of calibration images collected.
     * @return count of saved calibration frames
     */
    int get_calibration_images_count();
};

/**
 * Generates 3D world coordinates for checkerboard corners.
 * Creates grid of points in Z=0 plane with origin at top-left.
 * @param width number of columns (internal corners)
 * @param height number of rows (internal corners)
 * @return vector of 54 3D points for standard 9×6 board
 */
std::vector<cv::Vec3f> generate_chesssboard_world_coordinates(int width=checkboard_width, int height=checkboard_height);

#endif //CALIBRATE_H
