//
// Created by Ajey K on 19/03/26.
//
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video.hpp>
#include <opencv2/features2d.hpp>
#include <stdlib.h>
#include <string>
#include <iostream>

#include "utils.h"

const std::string task7_file_name = "task7_hf_";
const std::string image_save_format = ".png";

int main(int argc, char* argv[]) {
    cv::VideoCapture cap(0);
    cv::Mat frame, grey;

    // Harris parameters
    int blockSize = 2;
    int apertureSize = 3;
    double k = 0.04;
    int thresh = 100;  // Adjust this (50-200)

    while (true) {
        cap >> frame;
        cv::cvtColor(frame, grey, cv::COLOR_BGR2GRAY);

        // Detect Harris corners
        cv::Mat harris_response;
        cv::cornerHarris(grey, harris_response, blockSize, apertureSize, k);

        cv::Mat harris_norm;
        cv::normalize(harris_response, harris_norm, 0, 255, cv::NORM_MINMAX, CV_32F);

        cv::Mat output = frame.clone();
        int count = 0;

        for (int i = 0; i < harris_norm.rows; i++) {
            const float* ptr = harris_norm.ptr<float>(i);
            for (int j = 0; j < harris_norm.cols; j++) {
                if (ptr[j] > thresh) {
                    cv::circle(output, cv::Point(j, i), 5, cv::Scalar(0, 0, 255), 2);
                    count++;
                }
            }
        }

        cv::putText(output, "Harris Corners: " + std::to_string(count),
                   cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1,
                   cv::Scalar(0, 0, 255), 2);

        cv::imshow("Harris Corners", output);

        int key = cv::waitKey(10);
        if (key == 'q') {
            break;
        } else if (key == 's') {
            const long long time_now = get_time_instant();
            const std::string time_now_str = std::to_string(time_now);
            const std::string task7_image_path = task7_file_name + time_now_str + image_save_format;
            cv::imwrite(task7_image_path, frame);
            std::cout << "Image stored to: " << task7_image_path << std::endl;
        }

        // Adjust threshold with keys
        if (key == '+') thresh += 10;
        if (key == '-') thresh = std::max(10, thresh - 10);
    }

    return 0;
}