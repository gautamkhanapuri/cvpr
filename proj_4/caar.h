//
// Created by Gautam Khanapuri on 10th March 2026
// Header file for CAAR (Calibration and Augmented Reality) - main orchestrator class.
// Manages video capture, mode switching, and coordinates calibration and AR modules.
//

#ifndef CAAR_H
#define CAAR_H

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include <filesystem>
#include <string>
#include <iostream>
#include <utility>
#include <set>
#include <map>

#include "utils.h"
#include "calibrate.h"
#include "ar.h"

namespace fs = std::filesystem;

// Allowed file extensions for calibration data file
inline const std::set<std::string> calibration_file_formats = {".csv"};

// Mapping of mode numbers to display names
inline const std::map<int, std::string> mode_number_to_name_map = {
    {1, "CALIBRATION"},
    {2, "REPROJECTION"}
};

// Video capture configuration
inline int device_id = 0;                           // Camera device ID
inline int api_id = cv::CAP_AVFOUNDATION;          // Video API (macOS)

// Window title
inline const std::string window1 = "Calibration and Augmented Reality";

// Filename prefix for AR screenshots
inline const std::string ar_image_filename = "AR_";

// Duration to display mode switch notification (frames)
inline const int timer_mode_switch_notification = 50;

/**
 * Main orchestrator class coordinating calibration and AR modes.
 * Manages video capture, user input, and delegates to Calibrate and AR modules.
 */
class CAAR {
private:
    cv::VideoCapture* vd_cap;                   // Video capture device
    cv::Size refS;                              // Reference frame size

    int mode;                                   // Current mode (1=calibration, 2=AR)

    cv::Mat main_frame;                         // Current video frame
    cv::Mat display_frame;                      // Frame with overlays for display

    Calibrate calib;                            // Calibration module
    bool display_mode_switch_notification;      // Flag to show mode change message
    int mode_switch_notification_countdown_timer;  // Frames remaining for notification

    AR ar;                                      // AR projection module

    int pressed;                                // Last key pressed by user

    /**
     * Initializes video capture device and configures camera.
     * @return 0 if successful
     */
    int vid_setup();

    /**
     * Dispatches key handling to appropriate mode handler.
     * @return 0 if successful
     */
    int handle_key();

    /**
     * Handles keyboard input during calibration mode.
     * 's' - save calibration image, 'r' - switch to AR mode
     * @return 0 if successful
     */
    int handle_key_in_calibration_mode();

    /**
     * Handles keyboard input during AR mode.
     * '0-3' - toggle objects, 'a' - toggle axes, 'c' - return to calibration
     * @return 0 if successful
     */
    int handle_key_in_reprojection_mode();

    /**
     * Displays temporary mode switch notifications.
     * @return 0 if successful
     */
    int display_notifications();

public:
    /**
     * Constructor. Initializes modules and loads existing calibration if available.
     * @param calibration_file_path path to calibration data CSV
     */
    CAAR(fs::path& calibration_file_path);

    /**
     * Destructor. Saves calibration data to file.
     */
    ~CAAR();

    /**
     * Main processing loop. Captures frames, detects checkerboard,
     * performs calibration or AR projection based on mode.
     * @return 0 on successful exit
     */
    int run();
};

#endif //CAAR_H
