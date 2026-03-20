//
// Created by Ajey K on 15/03/26.
//

#include "ar.h"

AR::AR() {
    this->axes = true;
    this->display_1 = false;
    this->display_2 = false;
    this->display_3 = false;
    this->rotation_matrix = cv::Mat::zeros(3, 1, CV_64FC1);
    this->translation_matrix = cv::Mat::zeros(3, 1, CV_64FC1);
}

void AR::draw_3d_axes(cv::Mat& src) {
    // Define 3D axis endpoints
    std::vector<cv::Vec3f> axis_points = {
        cv::Vec3f(0, 0, 0),   // Origin
        cv::Vec3f(axis_length, 0, 0),   // X-axis end (red)
        cv::Vec3f(0, -axis_length, 0),  // Y-axis end (green)
        cv::Vec3f(0, 0, axis_length)    // Z-axis end (blue)
    };

    // Project to 2D
    std::vector<cv::Point2f> projected;
    cv::projectPoints(axis_points, this->rotation_matrix, this->translation_matrix, this->calibration_matrix, this->distortion_matrix, projected);

    // Draw axes
    cv::line(src, projected[0], projected[1], cv::Scalar(0, 0, 255), 3);    // X: Red
    cv::line(src, projected[0], projected[2], cv::Scalar(0, 255, 0), 3);    // Y: Green
    cv::line(src, projected[0], projected[3], cv::Scalar(255, 0, 0), 3);    // Z: Blue

    // Label axes
    cv::putText(src, "X", projected[1], cv::QT_FONT_NORMAL, 0.6, cv::Scalar(0, 0, 255), 2);
    cv::putText(src, "Y", projected[2], cv::QT_FONT_NORMAL, 0.6, cv::Scalar(0, 255, 0), 2);
    cv::putText(src, "Z", projected[3], cv::QT_FONT_NORMAL, 0.6, cv::Scalar(255, 0, 0), 2);
}

void AR::draw_corner_points(cv::Mat& src, int cols, int rows) {
    // Define 4 outer corners of checkerboard (9 cols, 6 rows → 0-8, 0 to -5)
    std::vector<cv::Vec3f> corner_points = {
        cv::Vec3f(0, 0, 0),           // Top-left
        cv::Vec3f(cols-1, 0, 0),      // Top-right (8, 0, 0)
        cv::Vec3f(0, -(rows-1), 0),   // Bottom-left (0, -5, 0)
        cv::Vec3f(cols-1, -(rows-1), 0)  // Bottom-right (8, -5, 0)
    };

    // Project to 2D
    std::vector<cv::Point2f> projected;
    cv::projectPoints(corner_points, this->rotation_matrix, this->translation_matrix, this->calibration_matrix, this->distortion_matrix, projected);

    // Draw circles at corners
    for (const cv::Point2f& pt : projected) {
        cv::circle(src, pt, 8, cv::Scalar(255, 255, 0), -1);  // Cyan filled circles
        cv::circle(src, pt, 10, cv::Scalar(0, 0, 0), 2);      // Black outline
    }
}

// std::vector<cv::Vec3f> AR::create_arrow() {
//     std::vector<cv::Vec3f> points;
//
//     // Arrow shaft (from back to front)
//     points.push_back(cv::Vec3f(4, -2, 1));    // 0: Back of shaft
//     points.push_back(cv::Vec3f(4, -2, 3));    // 1: Front of shaft
//
//     // Arrowhead (triangle pointing forward)
//     points.push_back(cv::Vec3f(3, -2, 3.5));  // 2: Left point of arrowhead
//     points.push_back(cv::Vec3f(5, -2, 3.5));  // 3: Right point
//     points.push_back(cv::Vec3f(4, -2, 4.5));  // 4: Tip of arrow
//
//     // Vertical fin on top (for asymmetry)
//     points.push_back(cv::Vec3f(4, -1, 2));    // 5: Top of fin
//
//     return points;
// }

std::vector<cv::Vec3f> AR::create_spiral() {
    std::vector<cv::Vec3f> points;
    int num_steps = 12;
    float radius = 1.5;

    for (int i = 0; i < num_steps; i++) {
        float angle = i * 2 * CV_PI / num_steps;  // Rotate around
        float height = i * 0.3;                    // Rise up

        // Outer point
        points.push_back(cv::Vec3f(
            4 + radius * cos(angle),
            -2 + radius * sin(angle),
            1 + height
        ));

        // Inner point (half radius)
        points.push_back(cv::Vec3f(
            4 + (radius/2) * cos(angle),
            -2 + (radius/2) * sin(angle),
            1 + height
        ));
    }

    return points;
}

std::vector<std::pair<int, int>> AR::spiral_lines() {
    std::vector<std::pair<int, int>> lines;
    int num_steps = 12;

    for (int i = 0; i < num_steps - 1; i++) {
        int outer = i * 2;
        int inner = i * 2 + 1;

        // Connect outer to outer (next step)
        lines.push_back({outer, outer + 2});

        // Connect inner to inner
        lines.push_back({inner, inner + 2});

        // Connect outer to inner (radial)
        lines.push_back({outer, inner});
    }

    return lines;
}

// std::vector<cv::Vec3f> AR::create_letter_N() {
//     std::vector<cv::Vec3f> points;
//     float base_x = 3;
//     float base_y = -3;
//     float base_z = 1;
//     float letter_height = 2;
//     float letter_width = 1.5;
//
//     // Left vertical bar (bottom to top)
//     points.push_back(cv::Vec3f(base_x, base_y, base_z));                    // 0: Bottom left
//     points.push_back(cv::Vec3f(base_x, base_y, base_z + letter_height));    // 1: Top left
//
//     // Diagonal (bottom-left to top-right)
//     points.push_back(cv::Vec3f(base_x + letter_width, base_y, base_z + letter_height));  // 2: Top right
//
//     // Right vertical bar (top to bottom)
//     points.push_back(cv::Vec3f(base_x + letter_width, base_y, base_z));     // 3: Bottom right
//
//     // Add some depth (make it 3D, not flat)
//     float depth = 0.3;
//     points.push_back(cv::Vec3f(base_x, base_y - depth, base_z));                    // 4: Back bottom left
//     points.push_back(cv::Vec3f(base_x, base_y - depth, base_z + letter_height));    // 5: Back top left
//     points.push_back(cv::Vec3f(base_x + letter_width, base_y - depth, base_z + letter_height));  // 6: Back top right
//     points.push_back(cv::Vec3f(base_x + letter_width, base_y - depth, base_z));     // 7: Back bottom right
//
//     return points;
// }

std::vector<cv::Vec3f> AR::create_dna_helix() {
    std::vector<cv::Vec3f> points;
    int num_base_pairs = 24;
    float radius = 1.2;
    float height_per_turn = 4.0;
    float base_x = 2.0, base_y = -2.0, base_z = 1.5;

    for (int i = 0; i < num_base_pairs; i++) {
        float t = i / (float)num_base_pairs;
        float angle = t * 8 * CV_PI;  // 1.5 turns
        float height = base_z + t * height_per_turn;

        // Strand 1
        points.push_back(cv::Vec3f(
            base_x + radius * cos(angle),
            base_y + radius * sin(angle),
            height
        ));

        // Strand 2 (opposite)
        points.push_back(cv::Vec3f(
            base_x - radius * cos(angle),
            base_y - radius * sin(angle),
            height
        ));

        // Inner connection point 1 (30% inward from strand 1)
        points.push_back(cv::Vec3f(
            base_x + 0.3 * radius * cos(angle),
            base_y + 0.3 * radius * sin(angle),
            height
        ));

        // Inner connection point 2 (30% inward from strand 2)
        points.push_back(cv::Vec3f(
            base_x - 0.3 * radius * cos(angle),
            base_y - 0.3 * radius * sin(angle),
            height
        ));
    }

    return points;
}

void AR::draw_dna_special(cv::Mat& src, double time_seconds) {
    // Create base structure
    auto base_points = this->create_dna_helix();

    // Apply incline + rotation
    std::vector<cv::Vec3f> transformed_points;
    float center_x = 4.0, center_y = -3.0;
    float incline_angle = 45 * CV_PI / 180;  // 15 degree tilt
    float rotation_angle = time_seconds * 0.3;  // 0.5 rad/sec rotation speed

    for (const cv::Vec3f& p : base_points) {
        // Step 1: Apply incline (shift X based on height)
        float x_inclined = p[0] + (p[2] - 0.5) * tan(incline_angle);

        // Step 2: Rotate around vertical axis through center
        float dx = x_inclined - center_x;
        float dy = p[1] - center_y;

        float x_rotated = dx * cos(rotation_angle) - dy * sin(rotation_angle);
        float y_rotated = dx * sin(rotation_angle) + dy * cos(rotation_angle);

        transformed_points.push_back(cv::Vec3f(
            x_rotated + center_x,
            y_rotated + center_y,
            p[2]
        ));
    }

    // Project to 2D
    std::vector<cv::Point2f> projected;
    cv::projectPoints(transformed_points, this->rotation_matrix,
                     this->translation_matrix, this->calibration_matrix,
                     this->distortion_matrix, projected);

    // Draw with multiple colors
    int num_base_pairs = 24;
    for (int i = 0; i < num_base_pairs - 1; i++) {
        int strand1 = i * 4;
        int strand2 = i * 4 + 1;
        int inner1 = i * 4 + 2;
        int inner2 = i * 4 + 3;

        // Strand 1 backbone (RED)
        cv::line(src, projected[strand1], projected[strand1 + 4],
                cv::Scalar(0, 0, 255), 4);

        // Strand 2 backbone (BLUE)
        cv::line(src, projected[strand2], projected[strand2 + 4],
                cv::Scalar(255, 0, 0), 4);

        // Base pairs - alternate colors
        cv::Scalar rung_color = (i % 2 == 0) ? cv::Scalar(0, 255, 255) : cv::Scalar(0, 255, 0);
        cv::line(src, projected[strand1], projected[inner1], rung_color, 2);
        cv::line(src, projected[inner1], projected[inner2], rung_color, 2);
        cv::line(src, projected[inner2], projected[strand2], rung_color, 2);
    }
}

std::vector<cv::Vec3f> AR::create_rocket() {
    std::vector<cv::Vec3f> points = {
        // Base circle (8 points)
        {3.5, -2.5, 1.0}, {4.2, -1.8, 1.0}, {4.5, -2.5, 1.0}, {4.2, -3.2, 1.0},
        {3.5, -3.2, 1.0}, {2.8, -3.2, 1.0}, {2.5, -2.5, 1.0}, {2.8, -1.8, 1.0},

        // Nose cone tip
        {3.5, -2.5, 4.0},  // 8: Top center

        // Fins (3 asymmetric fins)
        {5.0, -2.5, 1.5},  // 9: Right fin
        {2.5, -3.5, 1.5},  // 10: Left-back fin
        {2.5, -1.5, 1.5}   // 11: Left-front fin
    };

    return points;
}

void AR::draw_3d_object(cv::Mat& src, const std::vector<cv::Vec3f>& points_3d, const std::vector<std::pair<int, int>>& lines, cv::Scalar color) {
    // Project all 3D points to 2D
    std::vector<cv::Point2f> projected_points;
    cv::projectPoints(points_3d, this->rotation_matrix, this->translation_matrix, this->calibration_matrix, this->distortion_matrix, projected_points);

    // Draw lines
    for (const auto& [start, end] : lines) {
        cv::line(src, projected_points[start], projected_points[end], color, 2);
    }
}

int AR::draw_ar_ui(cv::Mat& src, bool found) {
// Define panel rectangles
    cv::Rect leftRect(10, 10, 300, 120);
    cv::Rect rightRect(src.cols - 360, 10, 350, 250);

    // Semi-transparent overlay
    auto applyOverlay = [&](cv::Rect roiRect) {
        roiRect &= cv::Rect(0, 0, src.cols, src.rows);
        if (roiRect.empty()) return;
        cv::Mat roi = src(roiRect);
        cv::Mat color(roi.size(), roi.type(), cv::Scalar(0, 0, 0));
        cv::addWeighted(color, 0.5, roi, 0.5, 0, roi);
    };

    applyOverlay(leftRect);
    applyOverlay(rightRect);

    // Colors
    cv::Scalar yellow(0, 255, 255), green(0, 255, 0), white(255, 255, 255);
    int font = cv::QT_FONT_NORMAL;

    // --- LEFT PANEL: Mode & Status ---
    int lx = leftRect.x + 10, ly = leftRect.y + 25;
    cv::putText(src, "Mode: AR", {lx, ly}, font, 0.6, yellow, 2);
    ly += 30;

    // Show which objects are active
    std::string objects = "Objects: ";
    if (axes) objects += "Axes ";
    if (display_1) objects += "Arrow ";
    if (display_2) objects += "Spiral ";
    if (display_3) objects += "Letter ";
    cv::putText(src, objects, {lx, ly}, font, 0.45, white, 1);

    ly += 25;
    cv::putText(src, "Press 0-3 to toggle", {lx, ly}, font, 0.4, white, 1);

    // --- RIGHT PANEL: Matrices ---
    int rx = rightRect.x + 10, ry = rightRect.y + 20;

    // Calibration Matrix
    cv::putText(src, "Calibration Matrix (K):", {rx, ry}, font, 0.5, white, 1);
    for (int i = 0; i < 3; i++) {
        ry += 18;
        std::string row = cv::format("[ %.1f  %.1f  %.1f ]",
                                     this->calibration_matrix.at<double>(i, 0),
                                     this->calibration_matrix.at<double>(i, 1),
                                     this->calibration_matrix.at<double>(i, 2));
        cv::putText(src, row, {rx + 10, ry}, font, 0.35, white, 1);
    }

    ry += 25;
    // Distortion
    cv::putText(src, "Distortion:", {rx, ry}, font, 0.5, white, 1);
    ry += 18;
    std::string dStr = "[ ";
    for (int i = 0; i < distortion_matrix_size; i++) {
        dStr += cv::format("%.3f ", this->distortion_matrix.at<double>(0, i));
    }
    dStr += "]";
    cv::putText(src, dStr, {rx + 10, ry}, font, 0.35, white, 1);

    ry += 30;
    // Rotation Matrix (3×3)
    cv::Mat rot_mat;
    cv::Rodrigues(this->rotation_matrix, rot_mat);
    cv::putText(src, "Rotation Matrix (R):", {rx, ry}, font, 0.5, green, 1);
    for (int i = 0; i < 3; i++) {
        ry += 18;
        std::string row = found ?
            cv::format("[ %.3f  %.3f  %.3f ]",
                rot_mat.at<double>(i, 0),
                rot_mat.at<double>(i, 1),
                rot_mat.at<double>(i, 2)) :
            "Not detected";
        cv::putText(src, row, {rx + 10, ry}, font, 0.35, green, 1);
    }

    ry += 25;
    // Translation Vector (3×1)
    cv::putText(src, "Translation (t):", {rx, ry}, font, 0.5, green, 1);
    ry += 18;
    std::string tStr = found ?
        cv::format("[ %.2f  %.2f  %.2f ]",
        this->translation_matrix.at<double>(0),
        this->translation_matrix.at<double>(1),
        this->translation_matrix.at<double>(2)) :
        "Not detected";
    cv::putText(src, tStr, {rx + 10, ry}, font, 0.35, green, 1);

    return 0;
}

int AR::display_objects(cv::Mat& src) {
    std::vector<cv::Point2f> corners;
    bool all_corners_detected = cv::findChessboardCorners(src, {checkboard_width, checkboard_height}, corners, cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_FAST_CHECK);
    if (all_corners_detected) {
        cv::Mat grey;
        cv::cvtColor(src, grey, cv::COLOR_BGR2GRAY);
        cv::cornerSubPix(grey, corners, cv::Size(11, 11), cv::Size(-1, -1),
        cv::TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));
        cv::solvePnP(chessboard_points_9x6, corners, this->calibration_matrix, this->distortion_matrix, this->rotation_matrix, this->translation_matrix);


        if (this->axes) {
            this->draw_3d_axes(src);
            this->draw_corner_points(src);
        }
        if (this->display_1) {
            auto now = std::chrono::steady_clock::now();
            double elapsed = std::chrono::duration<double>(now - start_time).count();
            this->draw_dna_special(src, elapsed);
            // auto dna = this->create_dna_helix();
            // const auto helix_lines = this->dna_lines();
            // this->draw_3d_object(src, dna, helix_lines, cv::Scalar(0, 255, 0));
            // auto arrow = this->create_arrow();
            // this->draw_3d_object(src, arrow, arrow_lines, cv::Scalar(0, 255, 0));
        }
        if (this->display_2) {
            const auto spiral = this->create_spiral();
            const auto s_lines = this->spiral_lines();
            this->draw_3d_object(src, spiral, s_lines, cv::Scalar(255, 0, 255));
        }
        if (this->display_3) {
            auto rocket = this->create_rocket();
            this->draw_3d_object(src, rocket, rocket_lines, cv::Scalar(0, 165, 255));
            // auto letter = this->create_letter_N();
            // this->draw_3d_object(src, letter, letter_N_lines, cv::Scalar(0, 165, 255));
        }
    }
    this->draw_ar_ui(src, all_corners_detected);
    return 0;
}

int AR::set_calibration_matrix(const cv::Mat &calib_mat) {
    this->calibration_matrix = calib_mat;
    return 0;
}

int AR::set_distortion_matrix(const cv::Mat &dist_mat) {
    this->distortion_matrix = dist_mat;
    return 0;
}

int AR::show_axes() {
    this->axes = !this->axes;
    return 0;
}

int AR::ar_object1() {
    this->display_1 = !this->display_1;
    return 0;
}

int AR::ar_object2() {
    this->display_2 = !this->display_2;
    return 0;
}

int AR::ar_object3() {
    this->display_3 = !this->display_3;
    return 0;
}


