//
// Created by Ajey K on 21/02/26.
//

#include "morph.h"

int morph(const cv::Mat &src, cv::Mat &dst) {
    // std::cout << "Start morph" << std::endl;
    // std::cout << "src type: " << src.type() << std::endl;
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(kernel_size, kernel_size));

    // std::cout << "Opening image" << std::endl;
    cv::morphologyEx(src, dst, cv::MORPH_OPEN, kernel);
    // std::cout << "Closing image" << std::endl;
    cv::morphologyEx(dst, dst, cv::MORPH_CLOSE, kernel);

    // std::cout << "dst type after morphology: " << dst.type() << std::endl;
    // std::cout << "Morphology done!" << std::endl;
    return 0;
}
