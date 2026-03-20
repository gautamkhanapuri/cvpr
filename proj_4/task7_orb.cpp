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

const std::string task7_file_name = "task7_";
const std::string image_save_format = ".png";

int main(int argc, char* argv[]) {
    cv::VideoCapture cap(0);
    cv::Mat frame, grey;

    // Create feature detector (pick one)
    cv::Ptr<cv::ORB> detector = cv::ORB::create(70);  // 500 features
    // Or: cv::Ptr<cv::GoodFeaturesToTrack> detector = ...

    while (true) {
        cap >> frame;
        cv::cvtColor(frame, grey, cv::COLOR_BGR2GRAY);

        // Detect features
        std::vector<cv::KeyPoint> keypoints;
        detector->detect(grey, keypoints);

        // Draw features
        cv::Mat output;
        cv::drawKeypoints(frame, keypoints, output,
                         cv::Scalar(0, 255, 0),  // Green circles
                         cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

        // Show info
        cv::putText(output, "Features: " + std::to_string(keypoints.size()),
                   cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1,
                   cv::Scalar(0, 255, 0), 2);

        cv::imshow("Feature Detection", output);

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
    }

    return 0;
}