// Gautam Ajey Khanapuri
// 03 February 2026
// Accepts a directory of images and a feature set(distance metric). Then it writes the feature set of each image to a file. The program will print the the location of the file containing the feature set. 
// This feature file must be passed to the ./arrange binary.


#include "p1.h"


int main(int argc, char *argv[]) {

    std::cout << "Starting up Program Part 1..." << std::endl;
    if (argc < 3) {
     std::cout << "Incorrect usage!" << std::endl;
     std::cout << "Correct usage: ./p1 [distance_metric_type(a/b/c)] [file_path_to_images_dataset_directory]" << std::endl;
     std::exit(-1);
    }

    if (argc > 3) {
     std::cout << "Incorrect usage!" << std::endl;
     std::cout << "Correct usage: ./preprocess [distance_metric_type(a/b/c)] [file_path_to_images_dataset_directory]" << std::endl;
     std::cout << "Please make sure to escape spaces in the file path" << std::endl;
     std::exit(-1);
    }

    std::string arg1 = argv[1];
    std::string arg2 = argv[2];

    P1 p1;
    p1.parse_mode(arg1);
    p1.parse_dir(arg2);
    p1.run();
    return 0;
}


int P1::parse_mode(std::string& arg) {
    if (!arg.starts_with("-")) {
        std::cout << "Incorrect usage! Mode must begin with '-' " << std::endl;
        std::exit(-1);
    }
    if (valid_modes.count(arg[1]) == 0) {
        std::cout << "Allowed modes include: [m, b]" << std::endl;
        std::exit(-1);
    }
    const char hyphen = '-';
    if (arg[2] != hyphen) {
        std::cout << "Mode should be followed by '-'" << std::endl;
        std::exit(-1);
    }
    this->mode = arg[1];
    this->spec = arg.substr(3);
    return 0;
}


int P1::parse_dir(std::string& arg) {
    bool exists = fs::exists(arg);
    bool is_dir = fs::is_directory(arg);
    bool is_empty = fs::is_empty(arg);
    if ( !exists || !is_dir || is_empty) {
	 std::cout << "Exists: " << exists << std::endl;
	 std::cout << "Is a directory: " << is_dir << std::endl;
	 std::cout << "Is empty: " << is_empty << std::endl;
	 std::cout << "Dir path received: " << arg << std::endl;
     std::cout << "Invalid directory path provided." << std::endl;
     std::exit(-1);
    }
    this->dir = fs::absolute(arg);
    this->find_img_paths();
    return 0;
}


int P1::find_img_paths() {
    int count = 0;
    for (const fs::directory_entry& entry : fs::directory_iterator(this->dir)) {
        bool is_regular = entry.is_regular_file();
        std::string ext = entry.path().extension();
        bool ext_is_allowed = allowed_img_formats.count(ext);

        if (!(is_regular && ext_is_allowed)) {
            std::cout << "Either not a regular file or ext is not allowed: " << entry.path() << std::endl;
            continue;
        }
        this->img_paths.push_back(fs::absolute(entry.path()));
        count++;
    }
    if (count == 0) {
        std::cout << "No images found." << std::endl;
        std::exit(-1);
    }
    return 0;
}


int P1::run() {
    if (this->mode == "b") {
        std::cout << "Running Baseline match mode." << std::endl;
        this->run_bsm();  // baseline match mode
    }
    else if (this->mode == "m") {
        this->run_mhs();  // multiple histogram match mode
    }
    // else if (this->mode == "h") {
    //     std::cout << "Running Single Histogram match mode." << std::endl;
    //     this->run_bhs();  // basic histogram match mode
    // }

    std::cout << "Feature vectors are being written to the following files..." << std::endl;
    for (fs::path op_path : this->op_files) {
        std::cout << op_path << std::endl;
    }
    return 0;
}


int P1::run_bsm() {
	std::cout << "Inside run_bsm" << std::endl;
    std::string ts = std::to_string(get_time_instant());
    std::string op_filename = bsm_op_file_name + ts + op_file_format;
	std::cout << op_filename << std::endl;
    fs::path parent_dir = this->dir.parent_path();
    fs::path op_file_path = parent_dir / op_filename;
    this->op_files.push_back(fs::absolute(op_file_path));

    std::vector<float> dummy = {0.0};
    for (const fs::path& file : this->img_paths) {
        append_image_data_csv(op_file_path.string().c_str(), file.string().c_str(), dummy);
    }
    return 0;
}


// int P1::run_bhs() {
//
// }

int P1::run_mhs() {
    FeatureConfig config;
    parse_config(this->spec, config);

    for (std::size_t i = 0; i < config.parts.size(); i++) {
        std::string part = config.parts[i];
        HistogramType ht = config.hists[i];
        std::cout << "Processing part: " << part << std::endl;
        std::cout << "Histogram type: " << HISTOGRAM_NAMES.at(ht) << std::endl;

        std::string ts = std::to_string(get_time_instant());
        std::string op_file_name = part + "_" + HISTOGRAM_NAMES.at(ht) + "_" + mhs_op_file_name + ts + op_file_format;
        fs::path op_path = this->dir.parent_path() / op_file_name;
        std::cout << "Writing vectors to: " << op_path << std::endl;
    	this->op_files.push_back(op_path);
        for (fs::path img_path : this->img_paths) {
            std::vector<float> img_vec;
            std::cout << "Processing Image: " << img_path << std::endl;
            int res = compute_histogram(img_path, img_vec, ht, part);
            if (res != 0) {
                std::cout << "Error computing histogram." << std::endl;
                continue;
            }
            append_image_data_csv(fs::absolute(op_path).string().c_str(), fs::absolute(img_path).string().c_str(), img_vec);
        }
    }
    return 0;
}


// int main(int argc, char* argv[]) {
//
// 	fs::path op_path;
//
// 	if (argc < 3) {
// 	 std::cout << "Incorrect usage!" << std::endl;
// 	 std::cout << "Correct usage: ./preprocess [distance_metric_type(a/b/c)] [file_path_to_images_dataset_directory]" << std::endl;
// 	 std::exit(-1);
// 	} else if (argc > 3) {
// 	 std::cout << "Incorrect usage!" << std::endl;
// 	 std::cout << "Correct usage: ./preprocess [distance_metric_type(a/b/c)] [file_path_to_images_dataset_directory]" << std::endl;
// 	 std::cout << "Please make sure to escape spaces in the file path" << std::endl;
// 	 std::exit(-1);
// 	}
//
// 	std::string dm = argv[1];
// 	fs::path dir_path = argv[2];
//
// 	bool exists = fs::exists(dir_path);
// 	bool is_dir = fs::is_directory(dir_path);
// 	bool is_empty = fs::is_empty(dir_path);
// 	if (!(exists && is_dir && !is_empty)){
// 	 std::cout << "Invalid directory path provided." << std::endl;
// 	 std::exit(-1);
// 	}
//
// 	DistMetric base(dir_path);
// 	if (dm == "b") {
// 		op_path = base.baseline_match();
// 	}
// 	else if (dm == "h") {
// 		op_path = base.histogram_rg();
// 	}
// 	else {
// 	 std::cout << "Invalid feature vector: "<< dm << std::endl;
// 	 std::cout << "Valid options: 1. b - baseline matching on central 7x7 square using SSD." << std::endl;
// 	 std::exit(-1);
// 	}
//
// 	std::cout << "Feature vectors have been written to: " << fs::absolute(op_path) << std::endl;
// 	return 0;
//
// }
