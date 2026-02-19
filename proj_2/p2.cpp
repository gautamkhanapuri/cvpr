// Gautam Ajey Khanapuri
// 03 February 2026
// Accepts path to target image, distance metric type, path to file containing feature vectors.
// File containing feature vectors is obtained from ./preprocess


#include <iostream>
#include <string>
#include <cstdlib>
#include <filesystem>

// #include "similarity.h"
#include "p2.h"
#include "dist_utils.h"

namespace fs = std::filesystem;


int main(int argc, char *argv[]) {
	std::cout << "Starting up Program Part 2..." << std::endl;
	if (argc < 4) {
		std::cout << "Incorrect usage!" << std::endl;
		std::cout << "Correct usage (classic features): ./p2 [path_to_target_img] [distance_metric_type(-m-____)] [file_path_to_feature_vectors] <file_path_to_feature_vectors...>" << std::endl;
		std::cout << "In case using DNN embeddings -->" << std::endl;
		std::cout << "Correct usage (DNN embeddings): ./p2 [path_to_file] [distance_metric_type(-d-_)] [file_path_to_dnn_embeddings]" << std::endl;
		std::cout << "file should contain the names of images as csv. \nEach line should have the name of the image.\nThe first image will be considered the target." << std::endl;
		std::exit(-1);
	}
	std::string arg1 = argv[1];
	std::string arg2 = argv[2];

	P2 p2;
	p2.parse_mode(arg2);
	p2.validate_tgt_path(arg1);

	int start_index = 3;
	std::vector<std::string> fi(argv + start_index, argv + argc);
	p2.validate_spec(fi);
	p2.run();
	std::cout << "Terminating Program Part 2..." << std::endl;
	return 0;
}


int P2::validate_tgt_path_classic(const std::string &arg) {
	fs::path f(arg);
	bool is_img = check_file(f, allowed_img_formats);

	if (!(is_img)) {
		std::cout << "Problem with target image. Maybe file doesn't exist or may be empty: " << f << std::endl;
		std::exit(-1);
	}
	this->tgt_path = fs::absolute(f);
	return 0;
}

int P2::validate_tgt_path_dnn(const std::string &arg) {
	fs::path f(arg);
	std::set<std::string> csv = {".csv"};
	bool is_csv = check_file(arg, csv);
	if (!is_csv) {
		std::cout << "Target File " << arg << " should be a CSV with a single entry for the target. This is only for DNN mode." << std::endl;
		std::exit(-1);
	}
	this->tgt_path = fs::absolute(f);
	return 0;
}

int P2::validate_tgt_path(const std::string& arg) {
	std::map<Mode, TargetPathValidatorFunc>::const_iterator it = target_path_validators.find(this->mode);
	if (it == target_path_validators.end()) {
		std::cout << "No Target Validator found." << std::endl;
		std::exit(-1);
	}
	int val_ret = it->second(arg);
	return val_ret;
}


int P2::parse_mode(const std::string& arg) {
	if (!arg.starts_with('-')) {
		std::cout << "Mode flag must start with -" << std::endl;
	}
	char md = arg[1];
	// Mode it = modes.find(md);
	std::map<char, Mode>::const_iterator it = str_to_mode.find(md);
	if (it == str_to_mode.end()) {
		std::cout << "Valid modes are b, m and d" << std::endl;
		std::exit(-1);
	}
	this->mode = it->second;
	// std::map<Mode, SpecValidatorFunc>::const_iterator val = spec_validators.find(this->mode);
	// if (val == spec_validators.end()) {
	// 	std::cout << "No SpecValidatorFunc found." << std::endl;
	// 	std::exit(-1);
	// }
	// int val_ret = val->second();
	char char3 = arg[2];
	if (char3 != '-') {
		std::cout << "Mode flags (b, m, d) should be separated from the spec with a hyphen" << std::endl;
		std::exit(-1);
	}
	this->spec = arg.substr(3);
	return 0;
}


int P2::validate_spec(std::vector<std::string>& files) {
	std::map<Mode, SpecValidatorFunc>::const_iterator it = spec_validators.find(this->mode);
	if (it == spec_validators.end()) {
		std::cout << "No SpecValidatorFunc found." << std::endl;
		std::exit(-1);
	}
	int val_ret = it->second(files);
	return val_ret;
}


int P2::validate_basic_spec(std::vector<std::string>& files) {
	size_t num_files = files.size();
	if (num_files > 1) {
		std::cout << "Please provide a single file containing the list of images." << std::endl;
		std::exit(-1);
	}
	std::set<std::string> csv = {".csv"};
	bool is_csv = check_file(files[0], csv);
	if (!is_csv) {
		std::cout << "File " << files[0] << " is not in a valid csv format." << std::endl;
		std::exit(-1);
	}
	size_t num_of_configs = this->spec.size();
	bool diff = std::isalpha(this->spec.at(0));
	if ((num_of_configs != 1) || !diff) {
		std::cout << "Specification does not match expected pattern. try -b-i for SSD." << std::endl;
		std::exit(-1);
	}
	for (const std::string& file : files) {
		this->vec_files.push_back(fs::absolute(file));
	}
	return 0;
}


int P2::validate_classic_spec(std::vector<std::string>& files) {
	size_t num_files = files.size();
	size_t num_of_configs = this->spec.size() / config_size;
	bool div_by_four = (this->spec.size() % config_size) == 0;
	if (!div_by_four || (num_of_configs != num_files)) {
		std::cout << "Number of configs does not match number of files." << std::endl;
		std::exit(-1);
	}
	for (int i=0; i < this->spec.size(); i += config_size) {
		bool part = std::isalpha(this->spec.at(i));
		bool hist = std::isalpha(this->spec.at(i + 1));
		bool diff = std::isalpha(this->spec.at(i + 2));
		bool wt = std::isdigit(this->spec.at(i + 3));
		std::set<std::string> csv = {".csv"};
		bool is_csv = check_file(files[i / config_size], csv);
		if (!(part && hist && diff && wt && is_csv)) {
			std::cout << "Specification does not match expected pattern." << std::endl;
			std::exit(-1);
		}
	}
	for (const std::string& file : files) {
		this->vec_files.push_back(fs::absolute(file));
	}
	return 0;
}


int P2::validate_dnn_spec(std::vector<std::string>& files) {
	size_t num_files = files.size();
	if (num_files > 1) {
		std::cout << "Please provide a single file containing the DNN embeddings." << std::endl;
		std::exit(-1);
	}
	std::set<std::string> csv = {".csv"};
	bool is_csv = check_file(files[0], csv);
	if (!is_csv) {
		std::cout << "File " << files[0] << " is not a valid DNN embeddings in csv format." << std::endl;
		std::exit(-1);
	}
	size_t num_of_configs = this->spec.size();
	bool diff = std::isalpha(this->spec.at(0));
	if ((num_of_configs != 1) || !diff) {
		std::cout << "Specification does not match expected pattern." << std::endl;
		std::exit(-1);
	}
	for (const std::string& file : files) {
		this->vec_files.push_back(fs::absolute(file));
	}
	return 0;
}

int P2::run() {
	std::cout << "Parameters passed look correct so far. Starting comparison..." << std::endl;
	std::map<Mode, RunnerFunc>::const_iterator it = runners.find(this->mode);
	if (it == runners.end()) {
		std::cout << "No runner found." << std::endl;
	}
	int run_result = it->second();
	return run_result;
}


int P2::run_basic() {
	std::cout << "BASIC RUNNING" << std::endl;
	std::cout << "Target: " << this->tgt_path << std::endl;
	std::cout << "Spec: " << this->spec << std::endl;
	std::cout << "Files: ";
	for (const auto& f : this->vec_files) {
		std::cout << f.filename() << " ";
	}
	std::cout << std::endl;

	Basic basic(this->tgt_path, this->spec, this->vec_files, this->neighbours);
	basic.calculate_basic();
	int counter = 0;
	std::cout << "Printing out the images in decreasing order of similarity. \nProgram will print first five images by default. \nPress 'q' after that to terminate the program." << std::endl;
	std::cout << std::endl;
	std::cout << "Size of neighbour (in P2 file) = " << this->neighbours.size() << std::endl;
	for (const std::pair<double, fs::path>& p : this->neighbours) {
		if (counter >= top_n) {
			std::cout <<"Want to see the next image? Press 'n'. \nIf you want to exit, press 'q'." << std::endl;
			char inp;
			std::cin >> inp;
			if (inp == 'q') {
				break;
			}
		}
		counter++;
		std::cout << counter << ". " << p.second << " ---\t--- " << p.first << std::endl;
	}
	return 0;
}


int P2::run_classic() {
	std::cout << "CLASSIC RUNNING" << std::endl;
	std::cout << "Target: " << this->tgt_path << std::endl;
	std::cout << "Spec: " << this->spec << std::endl;
	std::cout << "Files: ";
	for (const auto& f : this->vec_files) {
		std::cout << f.filename() << " ";
	}
	std::cout << std::endl;
	Distance dist(this->tgt_path, this->spec, this->vec_files, this->neighbours);
	dist.calculate_classic();

	int counter = 0;
	std::cout << "Printing out the images in decreasing order of similarity. \nProgram will print first five images by default. \nPress 'q' after that to terminate the program." << std::endl;
	for (const std::pair<double, fs::path>& p : this->neighbours) {
		if (counter >= top_n) {
			std::cout <<"Want to see the next image? Press 'n'. \nIf you want to exit, press 'q'." << std::endl;
			char inp;
			std::cin >> inp;
			if (inp == 'q') {
				break;
			}
		}
		counter++;
		std::cout << counter << ". "  << p.second << " ---\t--- " << std::setprecision(10) << p.first << std::endl;
	}
	return 0;
}

int P2::run_dne() {
	std::cout << "DNN RUNNING" << std::endl;
	std::cout << "Target: " << this->tgt_path << std::endl;
	std::cout << "Spec: " << this->spec << std::endl;
	std::cout << "Files: ";
	for (const auto& f : this->vec_files) {
		std::cout << f.filename() << " ";
	}
	std::cout << std::endl;
	MyDNN mydnn(this->tgt_path, this->spec, this->vec_files, this->neighbours);
	mydnn.calculate_dnn();

	int counter = 0;
	std::cout << "Printing out the images in decreasing order of similarity. \nProgram will print first five images by default. \nPress 'q' after that to terminate the program." << std::endl;
	std::cout << std::endl;
	for (const std::pair<double, fs::path>& p : this->neighbours) {
		if (counter >= top_n) {
			std::cout <<"Want to see the next image? Press 'n'. \nIf you want to exit, press 'q'." << std::endl;
			char inp;
			std::cin >> inp;
			if (inp == 'q') {
				break;
			}
		}
		counter++;
		std::cout << counter << ". " << p.second << " ---\t--- " << std::setprecision(10) << p.first << std::endl;
	}
	return 0;
}


// int main(int argc, char* argv[]) {
//
// 	std::vector<std::pair<double,fs::path>> sorted_images;
// 	if (argc < 4) {
// 	 std::cout << "Incorrect usage!" << std::endl;
// 	 std::cout << "Correct usage: ./arrange [file_path_to_target_image] [distance_metric_type(a/b/c)] [file_path_to_feature_vectors]" << std::endl;
// 	 std::exit(-1);
// 	} else if (argc > 4) {
// 	 std::cout << "Incorrect usage!" << std::endl;
// 	 std::cout << "Correct usage: ./arrange [file_path_to_target_image] [distance_metric_type(a/b/c)] [file_path_to_feature_vectors]" << std::endl;
// 	 std::cout << "Please make sure to escape spaces in the file path" << std::endl;
// 	 std::exit(-1);
// 	}
//
// 	fs::path target(argv[1]);
// 	std::string dm = argv[2];
// 	fs::path ft_vec(argv[3]);
//
// 	bool target_exists = fs::exists(target);
// 	bool target_is_file = fs::is_regular_file(target);
//
// 	if (!(target_exists && target_is_file)) {
// 		std::cout << "Invalid target path provided." << std::endl;
// 		std::exit(-1);
// 	}
//
// 	bool ft_vec_exists = fs::exists(ft_vec);
// 	bool ft_vec_is_file = fs::is_regular_file(ft_vec);
//
// 	if (!(ft_vec_exists && ft_vec_is_file)) {
// 		std::cout << "Invalid path to feature vector provided." << std::endl;
// 		std::exit(-1);
// 	}
//
// 	if (dm == "b") {
// 		base_line_match(target, ft_vec, sorted_images);
// 	}
// 	else if (dm == "h") {
// 		rg_histogram_match(target, ft_vec, sorted_images);
// 	}
// 	else {
// 		std::cout << "Invalid feature vector: "<< dm << std::endl;
// 		std::cout << "Valid options: \n1. b - baseline matching on central 7x7 square using SSD." << std::endl;
// 		std::exit(-1);
// 	}
//
// 	int count = 1;
// 	if (!sorted_images.empty()) {
// 		std::cout << "Rank \t-*-\t Filename \t-*-\t Distance " << std::endl;
// 		for (const std::pair<double, fs::path>& res : sorted_images) {
// 			std::cout << count << ". " << res.second << "- \t" << res.first << std::endl;
// 			count += 1;
// 		}
// 	} else {
// 		std::cout << "List of sorted images not returned." << std::endl;
// 	}
// 	return 0;
// }
