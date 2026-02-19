// Gautam Ajey Khanapuri
// 19 January 2026
// Program to load, and open a video. Display images continuously. Key strokes have different functions.


#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include <iostream>
#include <stdio.h>
#include <chrono>
#include <string>
#include <cmath>
#include <cstdio>
#include <cstdlib>


class Extensions {

	private:
	cv::VideoCapture *vd_cap;
	cv::Mat frame;
	cv::Mat ascii_frame;
	int device_id;
	int api_id;
	cv::Size refS;
	char mode;

	public:
	Extensions();
	~Extensions();
	int begin_loop();

};


Extensions::Extensions() {

	std::cout << "Starting up Video Display" << std::endl;
	this->device_id = 0;
	this->api_id = CAP_AVFOUNDATION;
	this->vd_cap = new VideoCapture(device_id, api_id);
	this->mode = 'c';

	if (!this->vd_cap->isOpened()) {
		std::cout << "Unable to open video device. Terminating program!" << std::endl;
		exit(-1);
	}

	// When Video Capture has successfully been opened
	this->refS = Size((int) this->vd_cap->get(CAP_PROP_FRAME_WIDTH), (int) this->vd_cap->get(CAP_PROP_FRAME_HEIGHT)); 
	std::cout << "Video Display Initialized.\nWidth: " << refS.width << "\nHeight: " << refS.height << std::endl;
}


// This function runs an infinite loop of capturing frames from the camera and then displaying them in the window.
int VideoDisplay::begin_loop() {
	std::cout << "Running frame display in loop." << std::endl;
	cv::namedWindow("Video Display", 1);

	for(;;) {  // Running an infinite loop
		*vd_cap >> frame;  // Using the VideoCapture as a stream. Get a frame from it.
		if (frame.empty()) {
			printf("Frame is empty.");
			break;
		}

		if (this->mode == 'c') {  // c - Normal mode
			imshow("Video Display", frame);
		}
		else if (this-> mode == 'a') {
			ascii_video(frame, ascii_frame);
			imshow("Video Display", ascii_frame);
		}


}
