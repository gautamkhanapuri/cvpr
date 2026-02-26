//
// Created by Ajey K on 21/02/26.
//

#include "segment.h"

int Segment::make_segments(const cv::Mat &src, cv::Mat &label_map, std::vector<RegionStats> &regions) {
    // std::cout << "Entered Segment::make_segments" << std::endl;
    cv::Mat stats;
    cv::Mat centroids;

    const int num_regions = cv::connectedComponentsWithStats(src, label_map, stats, centroids);
    // std::cout << "Detected Regions: " << num_regions << std::endl;
    // region id = 0 is background. Skipping it.
    for (int i = 1; i < num_regions; i++) {
        cv::Mat mask = (label_map == i);
        const cv::Moments mi = cv::moments(mask, true);

        // std::cout << "Moment area " << i << ": " << mi.m00 << std::endl;
        if (mi.m00 < min_area_pix) {
            continue;
        }

        RegionStats r;
        r.region_id = i;
        r.moments = mi;
        r.centroid = cv::Point2f(mi.m10 / mi.m00, mi.m01 / mi.m00);
        r.dnn_label = "";

        double mu20 = mi.mu20 / mi.m00;
        double mu02 = mi.mu02 / mi.m00;
        double mu11 = mi.mu11 / mi.m00;
        r.angle = 0.5 * std::atan2(2 * mu11, mu20 - mu02); // in radians

        r.axisExtent = compute_extents(label_map, i, r.centroid, r.angle);
        // std::cout << "minE1 should be negative, maxE1 should be positive" << std::endl;
        // std::cout << "Computed extents: minE1=" << r.axisExtent.minE1
        //   << " maxE1=" << r.axisExtent.maxE1 << std::endl;
        // std::cout << "Computed extents: minE2=" << r.axisExtent.minE2
        //   << " maxE2=" << r.axisExtent.maxE2 << std::endl;
        r.oriented_box = create_obb_from_extents(r.centroid, r.angle, r.axisExtent);
        // std::vector<cv::Point> pts;
        // cv::findNonZero(mask, pts);
        // r.oriented_box = cv::minAreaRect(pts);
        r.contours.clear();
        cv::findContours(mask, r.contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        regions.push_back(r);
        // std::cout << "Computed extents at end: minE1=" << r.axisExtent.minE1
        //   << " maxE1=" << r.axisExtent.maxE1 << std::endl;
        // std::cout << "Computed extents: minE2=" << r.axisExtent.minE2
        //   << " maxE2=" << r.axisExtent.maxE2 << std::endl;
    }
    // std::cout << "Completed Segment::make_segments" << std::endl;
    return 0;
}

AxisExtent Segment::compute_extents(const cv::Mat &label_map, int region_id,
                                    const cv::Point2f &centroid, float angle_rad) {
    AxisExtent ext;
    ext.minE1 = FLT_MAX;
    ext.maxE1 = -FLT_MAX;
    ext.minE2 = FLT_MAX;
    ext.maxE2 = -FLT_MAX;

    float cos_theta = cos(angle_rad);
    float sin_theta = sin(angle_rad);

    // Project ALL pixels in this region
    for (int i = 0; i < label_map.rows; i++) {
        const int *ptr = label_map.ptr<int>(i);
        for (int j = 0; j < label_map.cols; j++) {
            if (ptr[j] != region_id) continue; // Skip other regions

            // Vector from centroid to this pixel
            float dx = j - centroid.x;
            float dy = i - centroid.y;

            // Project onto major axis
            float proj_major = dx * cos_theta + dy * sin_theta;

            // Project onto minor axis (perpendicular)
            float proj_minor = -dx * sin_theta + dy * cos_theta;

            // Track extents
            ext.minE1 = std::min(ext.minE1, proj_major);
            ext.maxE1 = std::max(ext.maxE1, proj_major);
            ext.minE2 = std::min(ext.minE2, proj_minor);
            ext.maxE2 = std::max(ext.maxE2, proj_minor);
        }
    }

    return ext;
}

cv::RotatedRect Segment::create_obb_from_extents(const cv::Point2f &centroid, float angle_rad, const AxisExtent &ext) {
    // std::cout << "start create obb " << std::endl;
    // std::cout << " minE1=" << ext.minE1
    //           << " maxE1=" << ext.maxE1
    //           << " minE2=" << ext.minE2
    //           << " maxE2=" << ext.maxE2 << std::endl;
    // Size of the box
    float width = ext.maxE1 - ext.minE1;
    float height = ext.maxE2 - ext.minE2;

    // Center of box (offset from centroid in principal axis coords)
    float center_offset_major = (ext.maxE1 + ext.minE1) / 2.0;
    float center_offset_minor = (ext.maxE2 + ext.minE2) / 2.0;

    // Convert to image coordinates
    float center_x = centroid.x + center_offset_major * cos(angle_rad)
                     - center_offset_minor * sin(angle_rad);
    float center_y = centroid.y + center_offset_major * sin(angle_rad)
                     + center_offset_minor * cos(angle_rad);

    cv::Point2f box_center(center_x, center_y);

    // std::cout << "after create obb " << std::endl;
    // std::cout << " minE1=" << ext.minE1
    //           << " maxE1=" << ext.maxE1
    //           << " minE2=" << ext.minE2
    //           << " maxE2=" << ext.maxE2 << std::endl;
    return cv::RotatedRect(box_center,
                           cv::Size2f(width, height),
                           angle_rad * 180 / CV_PI); // Convert to degrees
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
    for (const RegionStats &old_reg: this->old_regions) {
        float x_ch = std::abs(old_reg.centroid.x - reg.centroid.x);
        float y_ch = std::abs(old_reg.centroid.y - reg.centroid.y);
        bool x_ok = x_ch < acceptable_centroid_x_ch;
        bool y_ok = y_ch < acceptable_centroid_y_ch;
        bool width_ok = std::abs(old_reg.oriented_box.size.width - reg.oriented_box.size.width) <
                        acceptable_box_width_ch;
        bool height_ok = std::abs(old_reg.oriented_box.size.height - reg.oriented_box.size.height) <
                         acceptable_box_height_ch;
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
    oriented_box.points(vertices); // The order of returned points is bottom left, top left, top right, bottom right.

    for (int i = 0; i < 4; i++) {
        cv::line(src, vertices[i], vertices[(i + 1) % 4], colour, 3);
    }
    return 0;
}

int RegionTracker::draw_axes(cv::Mat &src, const cv::Point2f &cent, const float angle_rad,
                             const cv::RotatedRect &oriented_box, const cv::Scalar &
                             colour) {
    // const float major = std::max(oriented_box.size.width, oriented_box.size.height) / 2.0f;
    // const float minor = std::min(oriented_box.size.width, oriented_box.size.height) / 2.0f;
    //
    // cv::Point2f major1(cent.x + std::cos(angle_rad) * major, cent.y + std::sin(angle_rad) * major);
    // cv::Point2f major2(cent.x - std::cos(angle_rad) * major, cent.y - std::sin(angle_rad) * major);
    // cv::line(src, major1, major2, colour, 2);
    //
    // float minor_angle = angle_rad + CV_PI / 2.0f;
    // cv::Point2f minor1(cent.x + std::cos(minor_angle) * minor, cent.y + std::sin(minor_angle) * minor);
    // cv::Point2f minor2(cent.x - std::cos(minor_angle) * minor, cent.y - std::sin(minor_angle) * minor);
    // cv::line(src, minor1, minor2, colour, 1);

    // --- 2nd attempt ---
    // Vector from centroid to OBB center
    float dx = oriented_box.center.x - cent.x;
    float dy = oriented_box.center.y - cent.y;

    // Project onto principal axis (dot product)
    float offset = dx * cos(angle_rad) + dy * sin(angle_rad);

    float major_total = std::max(oriented_box.size.width, oriented_box.size.height);
    float half_major = major_total / 2.0;

    // Distance from centroid to each edge
    float extent_positive = half_major + offset; // In positive direction
    float extent_negative = half_major - offset; // In negative direction

    cv::Point2f major1(cent.x + extent_positive * cos(angle_rad),
                       cent.y + extent_positive * sin(angle_rad));

    cv::Point2f major2(cent.x - extent_negative * cos(angle_rad),
                       cent.y - extent_negative * sin(angle_rad));

    cv::line(src, major1, major2, colour, 2);

    // Project onto minor axis (perpendicular direction)
    float offset_minor = dx * cos(angle_rad + CV_PI / 2) + dy * sin(angle_rad + CV_PI / 2);
    // Or equivalently: offset_minor = -dx * sin(angle_rad) + dy * cos(angle_rad);

    float minor_total = std::min(oriented_box.size.width, oriented_box.size.height);
    float half_minor = minor_total / 2.0;

    float minor_extent_pos = half_minor + offset_minor;
    float minor_extent_neg = half_minor - offset_minor;

    // Draw
    cv::Point2f minor1(cent.x + minor_extent_pos * cos(angle_rad + CV_PI / 2),
                       cent.y + minor_extent_pos * sin(angle_rad + CV_PI / 2));

    cv::Point2f minor2(cent.x - minor_extent_neg * cos(angle_rad + CV_PI / 2),
                       cent.y - minor_extent_neg * sin(angle_rad + CV_PI / 2));

    cv::line(src, minor1, minor2, colour, 1);
    // float cos_theta = std::cos(angle_rad);
    // float sin_theta = std::sin(angle_rad);
    //
    // // Major axis - use ACTUAL extents
    // cv::Point2f major_pos(cent.x + ae.maxE1 * cos_theta,
    //                       cent.y + ae.maxE1 * sin_theta);
    //
    // cv::Point2f major_neg(cent.x + ae.minE1 * cos_theta,
    //                       cent.y + ae.minE1 * sin_theta);
    //
    // cv::line(src, major_pos, major_neg, colour, 2);
    //
    // // Minor axis - perpendicular
    // float cos_perp = std::cos(angle_rad + CV_PI/2);
    // float sin_perp = std::sin(angle_rad + CV_PI/2);
    //
    // cv::Point2f minor_pos(cent.x + ae.maxE2 * cos_perp,
    //                       cent.y + ae.maxE2 * sin_perp);
    //
    // cv::Point2f minor_neg(cent.x + ae.minE2 * cos_perp,
    //                       cent.y + ae.minE2 * sin_perp);
    //
    // cv::line(src, minor_pos, minor_neg, colour, 1);
    return 0;
}

int RegionTracker::match_and_draw_box(cv::Mat &src, std::vector<RegionStats> &new_regions) {
    // std::cout << "Drawing regions..." << std::endl;
    for (RegionStats &r: new_regions) {
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
