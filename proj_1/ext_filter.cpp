// Gautam Ajey Khanapuri
// 19 January 2026
// Program to load, and open a video. Display images continuously. Key strokes have different functions.


#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include <iostream>
#include <string>
#include <cmath>

const std::string ascii_chars = " .,-~:;=!*#$@";
const int num_chars = 13;
const float norm = 255.0f / num_chars;
const int cell_width = 10;
const int cell_height = 15;


int ascii_video(cv::Mat &src, cv::Mat &dst) {

} 
