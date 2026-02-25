//
// Created by Ajey K on 21/02/26.
//

#ifndef MORPH_H
#define MORPH_H

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

// #include <iostream>

inline const int kernel_size = 5;

int morph(const cv::Mat& src, cv::Mat& dst);

#endif //MORPH_H
