//
// Created by Ajey K on 21/02/26.
// Header file for morphological filtering operations.
// Applies opening and closing operations to clean binary images.
//

#ifndef MORPH_H
#define MORPH_H

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

// Structuring element kernel size for morphological operations
inline const int kernel_size = 5;

/**
 * Applies morphological filtering to clean binary images.
 * Uses opening (erosion → dilation) to remove noise followed by
 * closing (dilation → erosion) to fill holes and connect fragmented regions.
 *
 * @param src input binary image
 * @param dst output cleaned binary image
 * @return 0 if successful
 */
int morph(const cv::Mat &src, cv::Mat &dst);

#endif //MORPH_H
