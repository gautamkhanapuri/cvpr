//
// Created by Ajey K on 15/03/26.
//

#include "calibrate.h"

Calibrate::Calibrate(const fs::path& calibration_file) {
    this->calibration_file = calibration_file;
    this->chessboard_width = checkboard_width;
    this->chessboard_height = checkboard_height;
    this->aruco_width = Aruco_width;
    this->aruco_height = Aruco_height;
    this->points_found = 0;
    this->mode = 1;
    this->is_calibrated = false;
    this->calibrated_images_count = 0;
    this->distortion_matrix = cv::Mat::zeros(1, distortion_matrix_size, CV_64FC1);
    this->error = 10.0;

    this->point_list.clear();
    this->corners.clear();
    this->corner_list.clear();
}

int Calibrate::set_default_calibration_matrix() {
    this->calibration_matrix = cv::Mat::eye(cv::Size(3, 3), CV_64FC1);
    this->calibration_matrix.at<double>(0, 2) = this->frame_size.width / 2.0;
    this->calibration_matrix.at<double>(1, 2) = this->frame_size.height / 2.0;
    return 0;
}

int Calibrate::read_calibration_data(cv::Size frame_size) {
    this->frame_size = frame_size;
    this->set_default_calibration_matrix();

    if (!fs::is_empty(this->calibration_file)) {
        std::ifstream inFile(this->calibration_file);

        if (inFile.is_open()) {
            std::string line, val;
            if (std::getline(inFile, line)) {
                std::stringstream ss(line);
                int index = 0;
                while (std::getline(ss, val, ',') && index < 9) {
                    int row = index / 3;
                    int col = index % 3;
                    this->calibration_matrix.at<double>(row, col) = std::stod(val);
                    index++;
                }
            }

            // 2. Read Distortion Coefficients (Second Line)
            if (std::getline(inFile, line)) {
                std::stringstream ss(line);
                int i = 0;
                while (std::getline(ss, val, ',')) {
                    // Check if string is not just whitespace
                    if (!val.empty()) {
                        this->distortion_matrix.at<double>(0, i) = std::stod(val);
                        i++;
                    }
                }
            }

            inFile.close();
            this->is_calibrated = true;
        }
    }
    return 0;
}

int Calibrate::write_calibration_data() {
    bool was_intially_empty = fs::is_empty(this->calibration_file);
    if (was_intially_empty) {
        std::cout << "Overwriting calibration file with new values." << std::endl;
    }
    std::ofstream outFile(this->calibration_file, std::ios::trunc);  // Overwrite mode
    if (outFile.is_open()) {
        for (int i = 0; i < 9; ++i) {
            // Access data as a flat array of floats
            outFile << this->calibration_matrix.at<double>(i);
            if (i < 8) outFile << ","; // Comma separated
        }
        outFile << "\n"; // Move to second line

        // 2. Write Distortion Coefficients (Second Line)
        for (size_t i = 0; i < this->distortion_matrix.cols; ++i) {
            outFile << this->distortion_matrix.at<double>(0, i);
            if (i < this->distortion_matrix.cols - 1) outFile << ",";
        }
        outFile.close();
    }
    std::cout << "Updated Calibration matrix written to: " << fs::absolute(this->calibration_file)
            << std::endl;
    return 0;
}

int Calibrate::detect_and_mark_corners(cv::Mat &src) {
    cv::Size pattern_size = cv::Size(this->chessboard_width, this->chessboard_height);
    bool all_corners_detected = cv::findChessboardCorners(src, pattern_size, this->corners, cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_FAST_CHECK);
    // std::cout << "Corners detected: " << all_corners_detected << std::endl;
    if (all_corners_detected) {
        cv::Mat grey;
        cv::cvtColor(src, grey, cv::COLOR_BGR2GRAY);
        cv::cornerSubPix(grey, this->corners, cv::Size(11, 11), cv::Size(-1, -1),
        cv::TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));
    }
    cv::drawChessboardCorners(src, pattern_size, cv::Mat(this->corners), all_corners_detected);
    this->draw_calibration_ui(src);
    if (all_corners_detected) {
        this->detected_frame = src.clone();
    }
    return 0;
}

int Calibrate::draw_calibration_ui(cv::Mat& dst) {
    // 1. Define Fixed Panel Rectangles
    cv::Rect leftRect(10, 10, 350, 100);
    cv::Rect rightRect(dst.cols - 310, 10, 300, 180);

    // 2. Fast ROI Blending function (lambda for reuse)
    auto applyOverlay = [&](cv::Rect roiRect) {
        // Ensure ROI stays within frame boundaries
        roiRect &= cv::Rect(0, 0, dst.cols, dst.rows);
        if (roiRect.empty()) return;

        cv::Mat roi = dst(roiRect);
        cv::Mat color(roi.size(), roi.type(), cv::Scalar(0, 0, 0)); // Black
        double alpha = 0.5;
        cv::addWeighted(color, alpha, roi, 1.0 - alpha, 0, roi);
    };

    applyOverlay(leftRect);
    applyOverlay(rightRect);

    // Styling Constants
    int font = cv::QT_FONT_NORMAL;
    cv::Scalar yellow(0, 255, 255), green(0, 255, 0), red(0, 0, 255), white(255, 255, 255);

    // --- LEFT SIDE: STATUS ---
    int lx = leftRect.x + 10, ly = leftRect.y + 25;
    cv::putText(dst, "Mode: CALIBRATION", {lx, ly}, font, 0.6, yellow, 2);
    ly += 25;
    cv::putText(dst, this->is_calibrated ? "Calibration: TRUE" : "Calibration: FALSE",
                {lx, ly}, font, 0.5, this->is_calibrated ? green : red, 2);
    if (this->is_calibrated) {
        ly += 25;
        cv::putText(dst, "Calibrated Images: " + std::to_string(this->calibrated_images_count) + " / " + std::to_string(min_calibration_images_required),
                    {lx, ly}, font, 0.45, white, 1);
    }

    // --- RIGHT SIDE: DATA ---
    int rx = rightRect.x + 10, ry = rightRect.y + 20;
    cv::putText(dst, "Calibration Matrix:", {rx, ry}, font, 0.5, white, 1);
    // std::cout << "Reached line 149 in calibrate" << std::endl;
    for (int i = 0; i < 3; i++) {
        ry += 18;
        std::string row = cv::format("[ %.1f, %.1f, %.1f ]", this->calibration_matrix.at<double>(i,0), this->calibration_matrix.at<double>(i,1), this->calibration_matrix.at<double>(i,2));
        cv::putText(dst, row, {rx + 10, ry}, font, 0.4, white, 1);
    }

    ry += 25;
    cv::putText(dst, "Distortion:", {rx, ry}, font, 0.5, white, 1);
    ry += 18;
    std::string dStr = "[ ";
    for(int i=0; i < distortion_matrix_size; i++) dStr += cv::format("%.2f ", this->distortion_matrix.at<double>(0, i));
    dStr += "]";
    cv::putText(dst, dStr, {rx + 10, ry}, font, 0.4, white, 1);

    if (this->is_calibrated) {
        ry += 30;
        cv::putText(dst, cv::format("Reprojection Error: %.2f", this->error),
                    {rx, ry}, font, 0.5, (error < 1.0 ? green : red), 2);
    }
    return 0;
}

cv::Mat& Calibrate::get_calibration_matrix() {
    return this->calibration_matrix;
}

cv::Mat& Calibrate::get_distortion_matrix() {
    return this->distortion_matrix;
}

bool Calibrate::is_calibration_matrix_available() const {
    return this->is_calibrated;
}

int Calibrate::save_calibration_image() {
    if (this->corners.empty()) return 1;
    this->corner_list.push_back(this->corners);
    this->point_list.push_back(chessboard_points_9x6);
    // Save image.
    const long long time_now = get_time_instant();
    const std::string time_now_str = std::to_string(time_now);
    const std::string calibrated_image_path = calibration_file_name + time_now_str + image_save_format;
    cv::imwrite(calibrated_image_path, this->detected_frame);
    std::cout << "Image stored to: " << calibrated_image_path << std::endl;

    // Print corners and world coordinates with corresponding images file names.
    // for (const cv::Point2f& corner : this->corners) {
    //     std::cout << corner << "\t";
    // }
    // std::cout << std::endl;

    this->calibrated_images_count++;
    if (this->calibrated_images_count >= min_calibration_images_required) {
        // Recalibrate.
        std::cout << "Calibration matrix BEFORE calibration: " << std::endl;
        for (int i = 0; i < this->calibration_matrix.rows; i++) {
            std::cout << this->calibration_matrix.at<double>(i, 0) << "\t" << this->calibration_matrix.at<double>(i, 1)
            << "\t" << this->calibration_matrix.at<double>(i, 2)<<  std::endl;
        }
        std::vector<cv::Mat> Rs;
        std::vector<cv::Mat> ts;
        // Generate reprojection error. Update variable. Print recalibration error.
        this->error = cv::calibrateCamera(this->point_list, this->corner_list, this->frame_size,
            this->calibration_matrix, this->distortion_matrix, Rs, ts, cv::CALIB_FIX_ASPECT_RATIO);
        std::cout << "Calibration matrix AFTER calibration: " << std::endl;
        for (int i = 0; i < this->calibration_matrix.rows; i++) {
            std::cout << this->calibration_matrix.at<double>(i, 0) << "\t" << this->calibration_matrix.at<double>(i, 1) << "\t" << this
            ->calibration_matrix.at<double>(i, 2)<<  std::endl;
        }
        std::cout << "Re-projection error: " << this->error << std::endl;

        // Update the image_count, is_calibrated.
        this->is_calibrated = true;
    }
    return 0;
}

int Calibrate::get_calibration_images_count() {
    return this->calibrated_images_count;
}


std::vector<cv::Vec3f> generate_chesssboard_world_coordinates(int width, int height) {
    std::vector<cv::Vec3f> objectPoints;
    objectPoints.reserve(width * height);

    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            // Coordinate: (x, y, z)
            objectPoints.push_back(cv::Vec3f(j * square_size, i * square_size, 0.0f));
        }
    }
    return objectPoints;
}
