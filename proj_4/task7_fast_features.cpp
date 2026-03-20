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

const std::string task7_file_name = "task7_ff_";
const std::string image_save_format = ".png";

int main(int argc, char* argv[]) {
    cv::VideoCapture cap(0);
    cv::Mat frame, grey;

    // FAST detector
    int threshold = 20;  // Adjust (10-50)
    cv::Ptr<cv::FastFeatureDetector> detector = cv::FastFeatureDetector::create(threshold);

    while (true) {
        cap >> frame;
        cv::cvtColor(frame, grey, cv::COLOR_BGR2GRAY);

        // Detect FAST features
        std::vector<cv::KeyPoint> keypoints;
        detector->detect(grey, keypoints);

        // Draw
        cv::Mat output;
        cv::drawKeypoints(frame, keypoints, output,
                         cv::Scalar(255, 0, 255),  // Magenta
                         cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

        cv::putText(output, "FAST Features: " + std::to_string(keypoints.size()),
                   cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1,
                   cv::Scalar(255, 0, 255), 2);

        cv::imshow("FAST Features", output);

        int key = cv::waitKey(10);
        if (key == 'q') {
            break;
        } else if (key == 's') {
            const long long time_now = get_time_instant();
            const std::string time_now_str = std::to_string(time_now);
            const std::string task7_image_path = task7_file_name + time_now_str + image_save_format;
            cv::imwrite(task7_image_path, output);
            std::cout << "Image stored to: " << task7_image_path << std::endl;
        }

        // Adjust threshold
        if (key == '+') {
            threshold += 5;
            detector = cv::FastFeatureDetector::create(threshold);
        }
        if (key == '-') {
            threshold = std::max(5, threshold - 5);
            detector = cv::FastFeatureDetector::create(threshold);
        }
    }

    return 0;
}