// Gautam Ajey Khanapuri
// 20 January 2026
// Header for the helper file 'filter.cpp' which contains methods that implement different kinds of filters.
#ifndef FILTER_H
#define FILTER_H

#include <opencv2/core.hpp>

// Converts a standard RGB image into greyscale image. This is a custom implementation. For all functions I have performed operations separately for each of the channels.
// I have provided two options: One takes the average of all three channels and assigned it to each channel. The second channel subtracts the red channel from 255 and assigns it to all three channels.
// Input params - src and dst cv::Mat and an optional paramter to select the type of conversion. Both matrices are passed by reference.
// returns 0 on success.
int greyscale(cv::Mat &src, cv::Mat &dst, int option=0);


// Converts a standard RGB image into sepia image. This gives us the feel that the image has been taken by an old camera. I used the matrix provided to us in the HW description.
// Input params - src and dst cv::Mat. Both matrices are passed by reference. THis is the case for all functions in the project. Even opencv does this to avoid having to copy all image matrices for all function calls.
// returns 0 on success.
int sepia(cv::Mat &src, cv::Mat &dst);


// Adds vignetting to a sepia image. This means image is darker along the edges. I have chosen to darken the image from 0.6(of total distance) distance from center of the image. The image gets progressively darker up to0.7 times more than the original pixel values. I realize that to make a pixel darker, we just need to reduce the value of all three channels and to make something brighter, we simply increase the values of all three channels.
// Input params - src and dst cv::Mat. Both matrices are passed by reference. threshold defines the distance from the center that will not be affected. So 0.6 of the image from the center will remain as it is. I will progressively start darkening the pixel values towards the edge up to a max increase of 0.7 of original value.
// returns 0 on success.
int vignetting(cv::Mat &src, cv::Mat &dst, float threshold=0.6, float strength=0.7);


// This is a simple 5 by 5 gaussian blur function. It multiplies each value of the 2d matrix with the corresponding pixel. and then it adds up these values. So this results in 25 multiplications, 24 additions and 1 division. So that's 50 operations perpixel per channel.
// Input params - src and dst are cv::Matrices that are passed by reference.
// returns 0 on success.
int blur5x5_1(cv::Mat &src, cv::Mat &dst);


// This is a optimization over the previous blur function. The gaussian blur filter is implemented as a separable filter. First I perform convolution with a 1x5 separable matrix. So this leads to 5 multiplications, 4 additions and 1 division. This gives us 5 values, one value for each from the five rows. Then i convolute the separable filter with these 5 values. this again leads to 10 values. So this is a total of 20 operations. This is what makes it an improvement over the previous function. 
// Moreover I have written the logic for convolution inline instead of making it a function call.
//INput params - src and dst are cv::Mat passed by ref. src contains the unblurred image and dst is where the result is written.
//return 0 on success
int blur5x5_2(cv::Mat &src, cv::Mat &dst);


// Have implemented the sobel X filter as a separable filter. The sobel identifies vertical edges in the image. The sobel becomes positive from left to right.Outputs can be positive or negative. After calculating gradient, I have clamped the values to 255 and -255.
//INput params - src and dst are cv::Mat passed by ref. src contains the unblurred image and dst is where the result is written.
//return 0 on success
int sobelX3x3(cv::Mat &src, cv::Mat &dst);


// Have implemented the sobel Y filter as a separable filter. The sobel identifies horizontal edges in the image. The sobel becomes positive from bottom to top. Outputs can be positive or negative. After calculating gradient, I have clamped the values to 255 and -255.
//INput params - src and dst are cv::Mat passed by ref. src contains the unblurred image and dst is where the result is written. Since gradient can be positive or negative, the output is of the type signed short (CV_16SC3). Conversion to a displayable format is handled by the caller.
//return 0 on success
int sobelY3x3(cv::Mat &src, cv::Mat &dst);


// Combines the output of SobelX and SobelY. At each point in the image we calculate the magnitude of the gradient as mag = sqrt(sx*sx + sy*sy). After the sqrt, we need to normalize. I have clamped the resultant. This means any value greater than 255 is set to 255.
//INput params - sx and sy are outputs from the sobel filters and they are of the type signed short. sx, sy and dst are cv::Mat passed by ref. dst is of type uchar.
//return 0 on success
int magnitude(cv::Mat &sx, cv::Mat &sy, cv::Mat &dst);


// First we blur the input src frame using the the optimal blur function. Then we map the pixel values into buckets. Another way to think of it is that we have 255 buckets each with one possible value. So by reducing the number of buckets to 10, we are grouping 25 pixel values into a group and then assigning them a single value. In this function i using the lowest value in the bucket as the value to represent the bucket.
// src, blur and dst are cv::Mat passed by reference. All three are of type uchar. levels is an optional parameter with the default parameter as 10.
// returns 0 on success.
int blurQuantize(cv::Mat &src, cv::Mat &blur, cv::Mat &dst, int levels=10);


// Uses the output from the DAv2 to fog regions of the image further away from the camera.
// input params. depth is the greyscale output of the DAv2 net.
// returns 0 on success.
int depth_fog(cv::Mat &src, cv::Mat &depth, cv::Mat &dst);


// Uses the output from the DAv2 to blug regions of the image further away from the camera.
// input params. depth is the greyscale output of the DAv2 net.
// returns 0 on success.
int portrait_mode(cv::Mat &src, cv::Mat &depth, cv::Mat &dst);


// Creates an emboss effect. Uses the sobelx and sobely functions underneath. Then multiplies the sobel values with a direction of sunlight to make some edges light and some darker. The plain regions are given a plain grey colour to make it look like a metal.
// input params. src is the original image. dst is where the output is to be written. x and y together determine the direction of incoming light. By default it is set to the top left corner of the image.
// returns 0 on success.
int emboss(cv::Mat &src, cv::Mat &dst, float x=0.707, float y=0.707);


// Uses the identify code to retain the face in colour whereas makes the rest black and white.
// input params - src and dst are the same. faces is vector of rectangles identified in the image.
// returns 0 on success.
int colourful_face(cv::Mat &src, cv::Mat &dst, std::vector<cv::Rect> &faces);


// adjusts the brightness, contrast and makes the image a negative of itself.
// input params - src and dst define the input and output frames. br - brightness, con - contrast, neg - negative true or false
int adjustments(cv::Mat &src, cv::Mat &dst, int br, int con, bool neg);

#endif

