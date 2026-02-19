//
// Created by Ajey K on 09/02/26.
//

#include "dist_utils.h"

#include "p2.h"


DistanceMetric parse_distance_metric(char c) {
    switch (c) {
        case 'i': return SSD;
        case 'I': return INTERSECTION;
        case 'q': return CHI_SQUARED;
        case 'Q': return EARTH_MOVER;
        case 'o': return COSINE;
        case 'O': return CORELATION;
        case 'y': return BHATTACHARYA;
        case 'Y': return MANHATTAN;
        case 'p': return CUSTOM_PERCEPTUAL;
        case 'P': return CUSTOM_WEIGHTED_COMBO;
        default:
            std::cout << "Unknown distance metric: " << c << std::endl;
        std::exit(0);
    }
}

double compute_distance(std::vector<float> const &x, std::vector<float> const &y, const DistanceMetric& metric) {
    std::map<DistanceMetric, DistanceFunction>::const_iterator it = distance_functions.find(metric);
    double ret = 0.0;
    if (it == distance_functions.end()) {
        ret = compute_ssd(x, y);
    } else {
        ret = it->second(x, y);
    }
    return ret;
}

double compute_ssd(const std::vector<float> &x, const std::vector<float> &y) {
    double sum = 0.0;
    for (size_t i = 0; i < x.size(); i++) {
        double diff = x[i] - y[i];
        sum += diff * diff;
    }
    return sum / x.size();  // Lower => more similar
}

double compute_intersection(const std::vector<float> &x, const std::vector<float> &y) {
    double intersection = 0.0;
    for (size_t i = 0; i < x.size(); i++) {
        intersection += std::min(x[i], y[i]);
    }
    return 1.0 - intersection;  // Convert to distance (0 = identical)
    // Lower = more similar
}

double compute_chi_squared(const std::vector<float> &x, const std::vector<float> &y) {
    double chi_squared = 0.0;
    for (size_t i = 0; i < x.size(); i++) {
        float sum = x[i] + y[i];
        if (sum > 0.0001) {
            float diff = x[i] - y[i];
            chi_squared += (diff * diff) / sum;
        }
    }
    return chi_squared;  // Lower => more similar
}

double compute_emd_approx(const std::vector<float> &x, const std::vector<float> &y) {
    // TODO
    return 0.0;
}

double compute_earth_mover(const std::vector<float> &x, const std::vector<float> &y) {
    size_t n  = x.size();
    if (n > 32) {
        compute_emd_approx(x, y);
        return 0;
    }
    std::vector<double> cdf1(n);
    std::vector<double> cdf2(n);

    cdf1[0] = x[0];
    cdf2[0] = y[0];

    for (size_t i = 1; i < n; i++) {
        cdf1[i] = cdf1[i - 1] + x[i];
        cdf2[i] = cdf2[i - 1] + y[i];
    }
    double sum = 0.0;
    for (size_t i = 0; i < n; i++) {
        sum += std::abs(cdf1[i] - cdf2[i]);
    }
    return sum;

}

double compute_cosine_similarity(const std::vector<float> &x, const std::vector<float> &y) {
    double dot = 0.0;
    double mag1 = 0.0;
    double mag2 = 0.0;

    for (size_t i = 0; i < x.size(); i++) {
        dot += x[i] * y[i];
        mag1 += x[i] * x[i];
        mag2 += y[i] * y[i];
    }
    double cosine = dot / (std::sqrt(mag1) * std::sqrt(mag2));
    return 1.0 - cosine;  // Since normalized, all vectors are in a single quadrant. Therefore, cosine similarity will be between 0 and 1. So subtracting from 1 will make it distance.
}

double compute_corelation(const std::vector<float> &x, const std::vector<float> &y) {
    int n = x.size();

    double mean1 = 0.0;
    double mean2 = 0.0;
    for (int i = 0; i < n; i++) {
        mean1 += x[i];
        mean2 += y[i];
    }
    mean1 /= n;
    mean2 /= n;

    double numerator = 0.0;
    double sum_sq1 = 0.0;
    double sum_sq2 = 0.0;

    for (int i = 0; i < n; i++) {
        double diff1 = x[i] - mean1;
        double diff2 = y[i] - mean2;

        numerator += diff1 * diff2;
        sum_sq1 += diff1 * diff1;
        sum_sq2 += diff2 * diff2;
    }
    double corelation = numerator / (sqrt(sum_sq1) * sqrt(sum_sq2));
    return 1.0 - corelation;
}

double compute_bhattacharya(const std::vector<float> &x, const std::vector<float> &y) {
    double b = 0.0;
    for (size_t i = 0; i < x.size(); i++) {
        b += std::sqrt(x[i] * y[i]);
    }

    return -std::log(b);  // Lower = more similar
}

double compute_manhattan(const std::vector<float> &x, const std::vector<float> &y) {
    double sum = 0.0;
    for (size_t i = 0; i < x.size(); i++) {
        sum += std::abs(y[i] - x[i]);
    }
    return sum;  // Lower => similar
}

double compute_custom_perceptual_distance(const std::vector<float> &x, const std::vector<float> &y) {
    return 0;  // TODO
}

double compute_custom_weighted_combo(const std::vector<float> &x, const std::vector<float> &y) {
    return 0;  // TODO
}

int Distance::prep_configs() {
    double total_weights = 0.0;
    for (size_t i = 0; i < this->spec.size(); i += 4) {
        PartConfig p;
        char a0 = this->spec[i];
        char a1 = this->spec[i + 1];
        char a2 = this->spec[i + 2];
        char a3 = this->spec[i + 3];
        p.part_name = parse_part_name(a0);
        p.hist_type = parse_hist_type(a1);
        p.metric = parse_distance_metric(a2);
        p.weight = a3 - '0';
        total_weights += p.weight;
        int fl_index = static_cast<int>(i / config_size);
        p.feature_file = this->vec_files[fl_index];
        this->pc.push_back(p);
    }

    for (PartConfig& p : this->pc) {
        p.weight /= total_weights;
    }
    return 0;
}

Distance::Distance(fs::path& tgt_file, std::string& spec, std::vector<fs::path>& vf, std::vector<std::pair<double, fs::path>>& op) : tgt_file(tgt_file), spec(spec), vec_files(vf), op(op) {
    // this->tgt_file = tgt_file;
    // this->spec = spec;
    // this->vec_files = vf;
    // this->op = op;
    prep_configs();
    // std::cout << "Distance object created with " << this->pc.size()
    //           << " part configs" << std::endl;
}

int Distance::calculate_classic() {
    this->num_images = 0;
    for (PartConfig& pcfg : this->pc) {
        std::cout << "Working on:\nPart: " << pcfg.part_name << "\nHistogram_type: " << HISTOGRAM_NAMES.at(pcfg.hist_type) << "\nDistance Metric: " << DISTMETRIC_NAMES.at(pcfg.metric) << "\nWeight: " << pcfg.weight << std::endl;
        std::vector<char*> db_imgs;
        int read_resp = read_image_data_csv(pcfg.feature_file.string().c_str(), db_imgs, pcfg.img_vecs);
        if (read_resp != 0) {
            std::cout << "Unable to read file containing feature vectors: " << pcfg.feature_file << std::endl;
            std::exit(-1);
        }
        std::cout << "Images to be compared." << std::endl;
        for (const char* p : db_imgs) {
            std::cout << p << std::endl;
            fs::path tp(p);
            pcfg.img_names.push_back(tp);
        }
        size_t num_imgs_in_part_cfg = pcfg.img_names.size();
        std::cout << "Number of images in this part: " << num_imgs_in_part_cfg << std::endl;
        if (this->num_images == 0) {
            this->num_images = num_imgs_in_part_cfg;
        }
        if (this->num_images != num_imgs_in_part_cfg) {
            std::cout << "Not all configs have the same number of images!\nSize 1 = " << this->num_images << "\nSize in this PartConfig = " << num_imgs_in_part_cfg << std::endl;
        }
        int res = compute_histogram(this->tgt_file, pcfg.target_vector, pcfg.hist_type, pcfg.part_name);
        if (res != 0) {
            std::cout << "Error computing histogram for target image." << std::endl;
            std::exit(-1);
        }
        for (size_t i = 0; i < pcfg.img_names.size(); i++) {
            double diff = compute_distance(pcfg.target_vector, pcfg.img_vecs[i], pcfg.metric);
            // diff *= pcfg.weight;
            std::pair<double, fs::path> tmp_pair(diff, pcfg.img_names[i]);
            pcfg.diff_values.push_back(tmp_pair);
        }
    }
    std::cout << "Validating all CSVs have same images..." << std::endl;
    for (size_t i = 0; i < this->num_images; i++) {
        fs::path reference = this->pc[0].img_names[i];

        for (const PartConfig& pcfg : this->pc) {
            if (pcfg.img_names[i] != reference) {
                std::cout << "Image order mismatch at index " << i << std::endl;
                std::cout << "Expected: " << reference << std::endl;
                std::cout << "Got: " << pcfg.img_names[i] << std::endl;
                std::exit(-1);
            }
        }
    }
    std::cout << "Validation passed!" << std::endl;
    for (int i = 0; i < this->num_images; i++) {
        // std::cout << i << std::endl;
        double total_diff = 0.0;
        fs::path tmp;
        for (PartConfig& pcfg : this->pc) {
            // std::cout << "Part: " << pcfg.part_name << std::endl;
            // std::cout << std::setprecision(10) << pcfg.diff_values[i].first << std::endl;
            total_diff += pcfg.diff_values[i].first * pcfg.weight;
            if (tmp.empty()) {
                tmp = pcfg.img_names[i];
            }
            if (tmp != pcfg.img_names[i]) {
                std::cout << "Filenames do not match." << std::endl;
                std::exit(-1);
            }
        }
        std::pair<double, fs::path> tmp_pair(total_diff, tmp);
        this->op.push_back(tmp_pair);
    }
    std::ranges::sort(this->op);
    return 0;

}

int MyDNN::prep_config() {
    char a0 = this->spec[0];
    this->metric = parse_distance_metric(a0);
    return 0;
}

MyDNN::MyDNN(fs::path &tgt_file, std::string &spec, std::vector<fs::path> &vf,
             std::vector<std::pair<double, fs::path>>& op) : tgt_file(tgt_file), spec(spec), vec_files(vf), op(op) {
    // this->tgt_file = tgt_file;
    // this->spec = spec;
    // this->vec_files = vf;
    // this->op = op;
    this->prep_config();
}

int MyDNN::calculate_dnn() {
    std::cout << "Running Distance Metric: " << DISTMETRIC_NAMES.at(this->metric) << " for DNN embeddings." << std::endl;
    std::vector<char*> db_imgs;
    int target_read_resp = read_image_data_csv(tgt_file.string().c_str(), db_imgs, this->img_vecs);
    if (target_read_resp != 0 || this->img_vecs.empty()) {
        std::cout << "Unable to read Target file containing feature vectors: " << this->tgt_file << std::endl;
        std::exit(-1);
    }
    fs::path tgt_img_name(db_imgs[0]);
    this->tgt_vector = this->img_vecs[0];
    std::cout << "Target image name: " << tgt_img_name << std::endl;

    db_imgs.clear();
    this->img_vecs.clear();
    // std::cout << "First five values Target vector: " << std::endl;
    // for (int i = 0; i  < 5; i++) {
    //     std::cout << this->tgt_vector[i] << std::endl;
    // }
    // Ensure img_vecs is empty.
    std::cout << "Size of image_vecs after clearing = " << this->img_vecs.size() << std::endl;
    if (this->img_vecs.size() != 0) {
        std::cout << "image_vecs after clearing is not empty!" << std::endl;
        std::exit(-1);
    }
    int read_resp = read_image_data_csv(vec_files[0].string().c_str(), db_imgs, this->img_vecs);
    if (read_resp != 0) {
        std::cout << "Unable to read file containing feature vectors: " << vec_files[0] << std::endl;
        std::exit(-1);
    }
    std::cout << "Images to be compared." << std::endl;
    for (const char* p : db_imgs) {
        std::cout << p << std::endl;
        fs::path tp(p);
        this->img_names.push_back(tp);
    }
    size_t num_imgs = this->img_names.size();

    for (size_t i = 0; i < num_imgs; i++) {
        double diff = compute_distance(this->tgt_vector, this->img_vecs[i], this->metric);
        std::pair<double, fs::path> tmp_pair(diff, this->img_names[i]);
        this->op.push_back(tmp_pair);
    }
    std::ranges::sort(this->op);
    return 0;
}


Basic::Basic(fs::path &tgt_file, std::string &spec, std::vector<fs::path> &vf,
             std::vector<std::pair<double, fs::path>>& op) : tgt_file(tgt_file), spec(spec), vec_files(vf), op(op) {
    // this->tgt_file = tgt_file;
    // this->spec = spec;
    // this->vec_files = vf;
    // this->op = op;
    this->prep_config();
}

int Basic::calculate_basic() {
    std::cout << "Running Distance Metric: " << DISTMETRIC_NAMES.at(this->metric) << " for Basic matching." << std::endl;
    std::vector<char*> db_imgs;
    std::vector<std::vector<float>> dummy;
    int csv_read_resp = read_image_data_csv(vec_files[0].string().c_str(), db_imgs, dummy);
    if (csv_read_resp != 0 || db_imgs.empty()) {
        std::cout << "Unable to read CSV file containing feature vectors: " << vec_files[0] << std::endl;
        std::exit(-1);
    }
    std::cout << "Images to be compared." << std::endl;
    for (const char* p : db_imgs) {
        std::cout << p << std::endl;
        fs::path tp(fs::absolute(p));
        this->img_paths.push_back(tp);
    }

    std::string basic_box_part_name = "W";
    compute_histogram(this->tgt_file, this->tgt_vector, HistogramType::BASIC_BOX, basic_box_part_name);
    std::cout << "Vector of target calculated. Size =  " << this->tgt_vector.size() << std::endl;

    for (const fs::path &p : this->img_paths) {
        std::vector<float> img_vec;
        compute_histogram(p, img_vec, HistogramType::BASIC_BOX, basic_box_part_name);
        double diff = compute_distance(this->tgt_vector, img_vec, this->metric);  // -b-i
		std::cout << "Diff = " << diff << std::endl << "File: " << p << std::endl;
        std::pair<double, fs::path> tmp_pair(diff, p);
        this->op.push_back(tmp_pair);
    }
	std::cout << "Size of output: " << this->op.size() << std::endl;
    std::ranges::sort(this->op);
	std::cout << "Size of output after sorting " << this->op.size() << std::endl;
	std::cout << "printing top 3 from dist_utils.cpp..." << std::endl;
	for (int i = 0; i < 3; i++) {
		std::cout << this->op[i].first << ", " << this->op[i].second << std::endl;
	}
    return 0;
}


int Basic::prep_config() {
    char a0 = this->spec[0];
    this->metric = parse_distance_metric(a0);
    return 0;
}



