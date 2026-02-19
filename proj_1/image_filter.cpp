#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

#include <string>
#include <iostream>
#include <chrono>
#include "filter.h"


int main(int argc, char* argv[]) {
	if (argc < 3) {
		std::cout << "Usage: ./image_filter [flag] [file_path]" << std::endl;
		return 1;
	}
	std::string file_path = argv[2];
	std::string flag = argv[1];
	auto frame_instant = std::chrono::system_clock::now().time_since_epoch();
	auto frame_instant_str = std::chrono::duration_cast<std::chrono::seconds>(frame_instant).count();
	std::string saved_filename = "image_filter_" + std::to_string(frame_instant_str) + ".png";

	cv::Mat frame = cv::imread(file_path, cv::IMREAD_COLOR);
	if (frame.empty()) {
		std::cout << "Could not read the image: '" << file_path << " Make sure you are escaping any spaces. If a space ' ' exists in the name of the file, write it as'\ '" << std::endl;
	}
	cv::Mat sobelx_frame;
	cv::Mat sobely_frame;
	cv::Mat mag_frame;

	sobelX3x3(frame, sobelx_frame);
	sobelY3x3(frame, sobely_frame);
	magnitude(sobelx_frame, sobely_frame, mag_frame);

	cv::imwrite(saved_filename, mag_frame);
	std::cout << "Gradient magnitude of the input image has been saved at: " << saved_filename << std::endl;
	std::cout << "Terminating..." << std::endl;
}
