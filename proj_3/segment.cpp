//
// Created by Ajey K on 21/02/26.
//

#include "segment.h"

int Segment::make_segments(const cv::Mat& src, cv::Mat& label_map, std::vector<RegionStats>& regions) {
    // std::cout << "Entered Segment::make_segments" << std::endl;
    cv::Mat stats;
    cv::Mat centroids;

    const int num_regions = cv::connectedComponentsWithStats(src, label_map, stats, centroids);
    // std::cout << "Detected Regions: " << num_regions << std::endl;
    // region id = 0 is background. Skipping it.
    for (int i=1; i<num_regions; i++) {
        cv::Mat mask = (label_map == i);
        const cv::Moments mi = cv::moments(mask, true);

        // std::cout << "Moment area " << i << ": " << mi.m00 << std::endl;
        if (mi.m00 < min_area_pix) {
            continue;
        }

        RegionStats r;
        r.region_id = i;
        r.moments = mi;
        r.centroid = cv::Point2f(mi.m10/mi.m00, mi.m01/mi.m00);

        double mu20 = mi.mu20/mi.m00;
        double mu02 = mi.mu02/mi.m00;
        double mu11 = mi.mu11/mi.m00;
        r.angle = 0.5 * std::atan2(2 * mu11, mu20 - mu02);  // in radians

        std::vector<cv::Point> pts;
        cv::findNonZero(mask, pts);
        r.oriented_box = cv::minAreaRect(pts);
        r.contours.clear();
        cv::findContours(mask, r.contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        regions.push_back(r);
    }
    // std::cout << "Completed Segment::make_segments" << std::endl;
    return 0;
}

// int Segment::get_obb(cv::Mat& src, int region_id, RegionStats& reg) {
//     std::vector<cv::Point> pts;
//     for (int i=0; i<src.rows; i++) {
//         int* ptr = src.ptr<int>(i);
//         for (int j=0; j<src.cols; j++) {
//             if (ptr[j] == region_id) {
//                 pts.push_back(cv::Point(j, i));
//             }
//         }
//     }
//     if (pts.size() > min_area_pix) {
//         reg.oriented_box = cv::minAreaRect(pts);
//         reg.angle = reg.oriented_box.angle;
//         reg.area = reg.oriented_box.size.width * reg.oriented_box.size.height;
//         return 0;
//     }
//     return 1;
// }

int RegionTracker::find_closest_prev(RegionStats &reg) {
    for (const RegionStats& old_reg : this->old_regions) {
        float x_ch = std::abs(old_reg.centroid.x - reg.centroid.x);
        float y_ch = std::abs(old_reg.centroid.y - reg.centroid.y);
        bool x_ok = x_ch < acceptable_centroid_x_ch;
        bool y_ok = y_ch < acceptable_centroid_y_ch;
        bool width_ok = std::abs(old_reg.oriented_box.size.width - reg.oriented_box.size.width) < acceptable_box_width_ch;
        bool height_ok = std::abs(old_reg.oriented_box.size.height - reg.oriented_box.size.height) < acceptable_box_height_ch;
        bool angle_ok = std::abs(old_reg.angle - reg.angle) < acceptable_box_angle_ch;
        if (x_ok && y_ok && width_ok && height_ok && angle_ok) {
            reg.color = old_reg.color;
            return 0;
        }
    }
    return 1;
}

int RegionTracker::draw_oriented_box(cv::Mat &src, const cv::RotatedRect &oriented_box, const cv::Scalar &colour) {
    cv::Point2f vertices[4];
    oriented_box.points(vertices);  // The order of returned points is bottom left, top left, top right, bottom right.

    for (int i=0; i<4; i++) {
        cv::line(src, vertices[i], vertices[(i+1) % 4], colour, 2);
    }
    return 0;
}

int RegionTracker::draw_axes(cv::Mat &src, const cv::Point2f &cent, const float angle_rad, const cv::RotatedRect &oriented_box,
    const cv::Scalar &colour) {
    const float major = std::max(oriented_box.size.width, oriented_box.size.height) / 2.0f;
    const float minor = std::min(oriented_box.size.width, oriented_box.size.height) / 2.0f;

    cv::Point2f major1(cent.x + std::cos(angle_rad) * major, cent.y + std::sin(angle_rad) * major);
    cv::Point2f major2(cent.x - std::cos(angle_rad) * major, cent.y - std::sin(angle_rad) * major);
    cv::line(src, major1, major2, colour, 1);

    float minor_angle = angle_rad + CV_PI / 2.0f;
    cv::Point2f minor1(cent.x + std::cos(minor_angle) * minor, cent.y + std::sin(minor_angle) * minor);
    cv::Point2f minor2(cent.x - std::cos(minor_angle) * minor, cent.y - std::sin(minor_angle) * minor);
    cv::line(src, minor1, minor2, colour, 1);
    return 0;
}

int RegionTracker::match_and_draw_box(cv::Mat &src, std::vector<RegionStats> &new_regions) {
    // std::cout << "Drawing regions..." << std::endl;
    for (RegionStats &r : new_regions) {
        int matched = find_closest_prev(r);
        if (matched != 0) {
            r.color = cv::Scalar(std::rand() % 255, std::rand() % 255, std::rand() % 255);
        }
        this->draw_oriented_box(src, r.oriented_box, r.color);
        this->draw_axes(src, r.centroid, r.angle, r.oriented_box, r.color);
    }
    this->old_regions = new_regions;
    // std::cout << "Finished drawing regions!" << std::endl;
    return 0;
}


