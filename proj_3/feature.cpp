//
// Created by Ajey K on 23/02/26.
//

#include "feature.h"

int Feature::calculate_basic_2d_features(std::vector<RegionStats>& regions) {
    // std::cout << "Entered calculate_basic_2d_features" << std::endl;
    // std::cout << "Detected regions (after ignoring small regions below threshold) = " << regions.size() << std::endl;
    for (RegionStats& r : regions) {
        // Percent filled
        float percent_filled = r.moments.m00 / r.oriented_box.size.width * r.oriented_box.size.height;
        float ar = r.oriented_box.size.height / r.oriented_box.size.width;
        float compactness = 0.0;
        float circularity = 0.0;
        if (!r.contours.empty()) {
            double perimeter = cv::arcLength(r.contours[0], true);
            compactness = (perimeter * perimeter) / (4 * CV_PI * r.moments.m00);
            circularity = (4 * CV_PI * r.moments.m00) / (perimeter * perimeter);
        }

        double hu[num_features];
        cv::HuMoments(r.moments, hu);
        double hu0 = std::log(std::abs(hu[0]));
        double hu1 = std::log(std::abs(hu[1]));
        double hu2 = std::log(std::abs(hu[2]));

        r.features.clear();
        r.features = {percent_filled, ar, compactness, circularity, static_cast<float>(hu0), static_cast<float>(hu1), static_cast<float>(hu2)};
    }
    // std::cout << "Completed calculate_basic_2d_features" << std::endl;
    return 0;
}

int Feature::overlay_features(cv::Mat &src, std::vector<RegionStats> &regions) {
    int count = 20;
    for (RegionStats& r : regions) {
        std::string lab = r.label.empty() ? "Unknown" : r.label;
        std::string str = std::format("Label: {} | Features: PF = {:0.2f} | AR = {:0.2f} | CO = {:0.2f} | CR = {:0.2f} | HU1 = {:0.2f} | HU2 = {:0.2f} | HU3 = {:0.2f} | ANGLE = {:0.2f}", lab, r.features[0], r.features[1], r.features[2], r.features[3], r.features[4], r.features[5], r.features[6], r.angle);
        cv::Point p(10, count);
        cv::putText(src, str, p, cv::QT_FONT_NORMAL, 0.6, r.color, 1);
        count += 20;
    }
    return 0;
}

int Classifier::load_db_file() {
    std::vector<char*> temp_labels;
    read_image_data_csv(this->db_path.string().c_str(), temp_labels, this->features);
    for (int i = 0; i < temp_labels.size(); i++) {
        this->labels.push_back(temp_labels[i]);
        this->initial_datapoints_count++;
        this->labels_set.insert(temp_labels[i]);
    }
    this->recalculate_average();
    this->recalculate_stddev();
    return 0;
}

bool Classifier::is_valid_label(const std::string &label) const {
    return labels_set.contains(label);
}

int Classifier::register_new_label(const std::string &label) {
    this->labels_set.insert(label);
    return 0;
}

int Classifier::recalculate_average() {
    int n_samples = this->features.size();
    if (n_samples == 0) {
        return 0;
    }
    this->average.assign(num_features, 0.0);
    for (const std::vector<float>& feature_vec : this->features) {
        for (int i = 0; i <num_features; i++) {
            this->average[i] += feature_vec[i];
        }
    }
    for (int i = 0; i < num_features; i++) {
        this->average[i] /= num_features;
    }
    std::cout << "Recalculated means for " << n_samples << " training examples" << std::endl;
    return 0;
}

int Classifier::recalculate_stddev() {
    int n_samples = this->features.size();
    if (n_samples < 2) {
        this->stddev.assign(num_features, 1.0);
        return 0;
    }
    this->stddev.assign(num_features, 0.0);

    for (const std::vector<float>& feature_vec : this->features) {
        for (int i = 0; i <num_features; i++) {
            double diff = feature_vec[i] - this->average[i];
            this->stddev[i] += diff * diff;
        }
    }

    for (int i = 0; i < num_features; i++) {
        this->stddev[i] = sqrt(this->stddev[i] / n_samples);
        if (this->stddev[i] < 0.0001) {
            this->stddev[i] = 1.0;
        }
    }
    std::cout << "Recalculated std dev" << std::endl;
    return 0;
}

Classifier::Classifier(fs::path db_path) {
    this->db_path = db_path;
    this->labels.clear();
    this->features.clear();
    this->labels_set.clear();
    this->average.assign(num_features, 0.0);
    this->stddev.assign(num_features, 1.0);
    this->initial_datapoints_count = 0;
    this->load_db_file();
}

int Classifier::write_new_trained_data() {
    size_t current_size = this->labels.size();
    if (current_size != this->initial_datapoints_count) {
        for (int i = this->initial_datapoints_count; i < current_size; i++) {
            append_image_data_csv(this->db_path.string().c_str(), this->labels[i].c_str(), this->features[i]);
        }
    }
    this->initial_datapoints_count = current_size;
    return 0;
}

int Classifier::train_on_all_segments(cv::Mat &orig_img, cv::Mat &label_map, std::vector<RegionStats> &regions) {
    std::cout << "Entered training mode." << std::endl;
    size_t number_of_regions = regions.size();
    std::cout << "Detected " << number_of_regions << " regions." << std::endl;
    std::cout << "Please enter the correct label for the displayed region in lower case." << std::endl;
    std::cout << "The region will be displayed to you. Press 'a' to be able to enter the label in the teminal or press 'p' to skip the region." << std::endl;
    std::cout << "If you press 'a', type into the terminal." << std::endl;

    int counter = 1;
    for (RegionStats& r : regions) {
        cv::Mat mask = (label_map == r.region_id);
        cv::Rect roi = cv::boundingRect(mask);
        cv::Mat display_mat;
        orig_img.copyTo(display_mat, mask);
        display_mat = display_mat(roi).clone();
        std::string title = training_img_display_window_name + " - Region " + std::to_string(counter) + " out of " + std::to_string(number_of_regions);
        cv::imshow(training_img_display_window_name, display_mat);
        cv::setWindowTitle(training_img_display_window_name, title);
        int key = cv::waitKey(0);
        if (key == 'p') {
            std::cout << "Skipping region" << counter << std::endl;
            continue;
        }
        if (key == 'a') {
            int retry_count = 3;
            std::string user_input_label;
            std::cout << "Available labels:" << std::endl;
            int idx = 0;
            for (const std::string& label : this->labels_set) {
                std::cout << idx << ". " << label << std::endl;
                idx++;
            }
            std::cin >> user_input_label;
            while (!is_valid_label(user_input_label) && retry_count > 0) {
                std::cout << "You typed: '" << user_input_label << "'" << std::endl;
                std::cout << "Is this a new label? <y/n>" << std::endl;
                std::string is_new_label;
                std::cin >> is_new_label;
                if (is_new_label == "y") {
                    this->register_new_label(user_input_label);
                    break;
                }
                std::cout << "Invalid label entered." << std::endl;
                std::cout << "Remaining retries = " << retry_count << ". Try again. Copy paste the correct label in lower case." << std::endl;
                int idx1 = 0;
                for (const std::string& label : this->labels_set) {
                    std::cout << idx1 << ". " << label << std::endl;
                    idx1++;
                }
                retry_count--;
            }
            if (!is_valid_label(user_input_label) && retry_count == 0) {
                std::cout << "You typed: '" << user_input_label << "'" << std::endl;
                std::cout << "Unrecognized label entered." << std::endl;
                std::cout << "Ignoring this region. Try again later." << std::endl;
                continue;
            }
            std::cout << "Successfully noted down region as: " << user_input_label << std::endl;
            this->labels.push_back(user_input_label);
            this->features.push_back(r.features);
        }
        counter++;
    }
    cv::destroyWindow(training_img_display_window_name);
    this->write_new_trained_data();
    this->recalculate_average();
    this->recalculate_stddev();
    return 0;
}

int Classifier::predict(std::vector<RegionStats> &regions) {
    // std::cout << "Entered predicting mode." << std::endl;
    // std::cout << regions.size() << " regions." << std::endl;
    for (RegionStats &r : regions) {
        float min_dist = std::numeric_limits<float>::max();
        std::string min_label = unknown;
        std::vector<float> this_region_features = r.features;
        for (int i=0; i < this->features.size(); i++) {
            float dist_sum = 0.0;
            std::string this_label = this->labels[i];
            std::vector<float> this_features = this->features[i];
            for (int j=0; j < num_features; j++) {
                double diff = this_features[j] - this_region_features[j];
                double scaled_diff = diff / this->stddev[j];
                dist_sum += scaled_diff * scaled_diff;
            }
            double distance = sqrt(dist_sum);
            if (distance < min_dist) {
                min_dist = distance;
                min_label = this_label;
            }
        }
        if (min_dist < label_matching_threshold) {
            // std::cout << min_label << std::endl;
            // std::cout << min_dist << std::endl;
            r.label = min_label;
        } else {
            r.label = unknown;
        }
        r.confidence = 1.0f / (1.0 + min_dist);
    }
    return 0;
}

bool Classifier::has_training_data() const {
    // std::cout << "Classifier::has_training_data()" << std::endl;
    // std::cout << "Labels: " << this->labels.size() << std::endl;
    // std::cout << "Features: " << this->features.size() << std::endl;
    return this->features.empty();
}
