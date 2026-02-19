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

#include "filter.h"
#include "faceDetect.h"
#include "DA2Network.hpp"


using namespace cv;
using namespace std;


// This class is reponsible for handling all the variables for displaying video, applying filters, saving images and closing the video stream.
// I have created a frame for each kind of filter to enable us to save the different versions of the frame (before and after applying the filter and any intermediate frames also).
class VideoDisplay {
	private:
	cv::VideoCapture *vd_cap;
	cv::Mat frame;
	cv::Mat grayscale_frame;
	cv::Mat my_grayscale_frame;
	cv::Mat sepia_frame;
	cv::Mat vignetting_frame;
	cv::Mat blur_frame;
	cv::Mat sobelx_frame;
	cv::Mat sobely_frame;
	cv::Mat mag_frame;
	cv::Mat bq_frame;
	cv::Mat tmp;
	cv::Mat fog;
	cv::Mat emboss_frame;
	cv::Mat adjust;
	cv::Mat cface_frame;
	int device_id;
	int api_id;
	cv::Size refS;
	char mode;
	int br;
	int con;
	bool neg;

	// For Task 10: Detect faces in an image.
	Mat grey;
	vector<Rect> faces;
	Rect last;
	// Task 10

	// For Task 11: Depth anything network
	cv::Mat da2_src;
	cv::Mat da2_dst;
	cv::Mat da2_dst_vis;
	const float reduction = 0.5; // In order to prevent severe lag during depth detection.
	DA2Network *da_net;
	float scale_factor;
	// Task 11

	public:
	VideoDisplay();
	~VideoDisplay();
	int start_loop();
};

// Contructor of the class. Initializes all the variables.
VideoDisplay::VideoDisplay() {
	std::cout << "Starting up Video Display" << std::endl;
	this->device_id = 0;
	this->api_id = CAP_AVFOUNDATION;
	this->last = Rect(0, 0, 0, 0);
	this->mode = 'c';
	this->br = 10;
	this->con = 10; 
	this->neg = false;

	this->vd_cap = new VideoCapture(device_id, api_id);
	this->da_net = new DA2Network("model_fp16.onnx");

	// Check if Video Capture has been successful or not
	if (!this->vd_cap->isOpened()) {
		std::cout << "Unable to open video device. Terminating program!" << std::endl;
		exit(-1);
	}

	// When Video Capture has successfully been opened
	this->refS = Size((int) this->vd_cap->get(CAP_PROP_FRAME_WIDTH), (int) this->vd_cap->get(CAP_PROP_FRAME_HEIGHT)); 
	std::cout << "Video Display Initialized.\nWidth: " << refS.width << "\nHeight: " << refS.height << std::endl;
	this->scale_factor = 256.0 / (refS.height * reduction);
	std::cout << "Reduction factor: " << this->scale_factor << std::endl; 
}

// This function runs an infinite loop of capturing frames from the camera and then displaying them in the window.
int VideoDisplay::start_loop() {
	std::cout << "Running frame display in loop." << std::endl;
	cv::namedWindow("Video Display", 1);

	for(;;) {  // Running an infinite loop
		*vd_cap >> frame;  // Using the VideoCapture as a stream. Get a frame from it.
		if (frame.empty()) {
			printf("Frame is empty.");
			break;
		}

		// Checking for different modes
		if (this->mode == 'c') {  // c - Normal mode
			if (neg || br != 10 || con != 10) {
				adjustments(frame, adjust, br, con, neg);
				imshow("Video Display", adjust);
			} else {
				imshow("Video Display", frame);
			}
		}
		else if (mode == 'p') {  // Colourful face mode
			cvtColor(frame, grey, COLOR_BGR2GRAY, 0);  // Converting image to greyscale.
			detectFaces(grey, faces);  // Detect faces.
			colourful_face(frame, cface_frame, faces);
			imshow("Video Display", cface_frame);
		}
		else if (mode == 'g') {   // In built opencv's greyscale mode
			cvtColor(frame, grayscale_frame, COLOR_RGB2GRAY);
			imshow("Video Display", grayscale_frame);
		}
		else if (mode == 'h') {  // My custom greyscale mode
			greyscale(frame, my_grayscale_frame);
			imshow("Video Display", my_grayscale_frame);
		}
		else if (mode == 'v') {  // sepia mode
			sepia(frame, sepia_frame);
			imshow("Video Display", sepia_frame);
		}
		else if (mode == 'V') {  // Vignetting mode
			sepia(frame, sepia_frame);
			vignetting(sepia_frame, vignetting_frame);
			imshow("Video Display", vignetting_frame);
		}
		else if (mode == 'b') {  // My custom 5 by 5 gaussian blur mode
			blur5x5_2(frame, blur_frame);
			imshow("Video Display", blur_frame);
		}
		else if (mode == 'x') {  // SobelX mode
			sobelX3x3(frame, sobelx_frame);
			convertScaleAbs(sobelx_frame, sobelx_frame);
			imshow("Video Display", sobelx_frame);
		}
		else if (mode == 'y') {  // SobelY mode
			sobelY3x3(frame, sobely_frame);
			convertScaleAbs(sobely_frame, sobely_frame);
			imshow("Video Display", sobely_frame);
		}
		else if (mode == 'm') {  // Magnitude mode
			sobelX3x3(frame, sobelx_frame);
			sobelY3x3(frame, sobely_frame);
			magnitude(sobelx_frame, sobely_frame, mag_frame);
			imshow("Video Display", mag_frame);
		}
		else if (mode == 'l') {  // Blur-Quantize mode
			blurQuantize(frame, tmp, bq_frame);
			imshow("Video Display", bq_frame);
		}
		else if (mode == 'f') {  // Face detection mode
			cvtColor(frame, grey, COLOR_BGR2GRAY, 0);  // Converting image to greyscale.
			detectFaces(grey, faces);  // Detect faces.
			drawBoxes(frame, faces);  // Draw boxes around the faces
			if (faces.size() > 0) {
				last.x = (faces[0].x + last.x)/2;	
				last.y = (faces[0].y + last.y)/2;	
				last.width = (faces[0].width + last.width)/2;	
				last.height = (faces[0].height + last.height)/2;	
			}
			imshow("Video Display", frame);
		}
		else if (mode == 'w') {  // Monocular depth estimation mode
			resize(frame, da2_src, Size(), reduction, reduction);  // for speed, reduce the size of the input by half
			da_net->set_input(da2_src, scale_factor);  // set the network input
			da_net->run_network(da2_dst, da2_src.size());  // run the network
			applyColorMap(da2_dst, da2_dst_vis, COLORMAP_INFERNO);
			imshow("Video Display", da2_dst_vis);
		}
		else if (mode == 'e') {  // Depth Fog mode
			resize(frame, da2_src, Size(), reduction, reduction);  // for speed, reduce the size of the input by half
			da_net->set_input(da2_src, scale_factor);  // set the network input
			da_net->run_network(da2_dst, da2_src.size());  // run the network
			depth_fog(da2_src, da2_dst, fog);
			imshow("Video Display", fog);
		}
		else if (mode == 'r') {  // Depth blur mode
			resize(frame, da2_src, Size(), reduction, reduction);  // for speed, reduce the size of the input by half
			da_net->set_input(da2_src, scale_factor);  // set the network input
			da_net->run_network(da2_dst, da2_src.size());  // run the network
			portrait_mode(da2_src, da2_dst, fog);
			imshow("Video Display", fog);
		}
		else if (mode == 't') {  // Emboss mode
			emboss(frame, emboss_frame);
			imshow("Video Display", emboss_frame);
		}
		else {  // Any other character will not cause any filter to apply
			imshow("Video Display", frame);
		}

		char key = waitKey(10);
		if (key == 's' || key == 'S') {
			auto frame_instant = chrono::system_clock::now().time_since_epoch();
			auto frame_instant_str = chrono::duration_cast<chrono::seconds>(frame_instant).count();
			string saved_filename = "image_capture_" + to_string(frame_instant_str) + ".png";
			if (key == 's') {
				if (mode == 'g') {
					string grs_filename = "grs_" + saved_filename;
					imwrite(grs_filename, grayscale_frame);
					printf("Saved GRAYSCALE frame with filename: %s \n", grs_filename.c_str());
				}
				else if (mode == 'h') {
					string my_grs_filename = "custom_grs_" + saved_filename;
					imwrite(my_grs_filename, my_grayscale_frame);
					printf("Saved CUSTOM GRAYSCALE frame with filename: %s \n", my_grs_filename.c_str());
				}
				else if (mode == 'v') {
					string sepia_filename = "sepia_" + saved_filename;
					imwrite(sepia_filename, sepia_frame);
					printf("Saved SEPIA frame with filename: %s \n", sepia_filename.c_str());
				}
				else if (mode == 'V') {
					string vignet_filename = "vignet_" + saved_filename;
					imwrite(vignet_filename, vignetting_frame);
					printf("Saved VIGNET frame with filename: %s \n", vignet_filename.c_str());
				}
				else if (mode == 'b') {
					string blur_filename = "blur_" + saved_filename;
					imwrite(blur_filename, blur_frame);
					printf("Saved BLUR (created with separable 1x5 filter) frame with filename: %s \n", blur_filename.c_str());
				}
				else if (mode == 'x') {
					string sobelx_filename = "sobelx_" + saved_filename;
					imwrite(sobelx_filename, sobelx_frame);
					printf("Saved SOBELX frame with filename: %s \n", sobelx_filename.c_str());
				}
				else if (mode == 'y') {
					string sobely_filename = "sobely_" + saved_filename;
					imwrite(sobely_filename, sobely_frame);
					printf("Saved SOBELY frame with filename: %s \n", sobely_filename.c_str());
				}
				else if (mode == 'm') {
					string mag_filename = "mag_" + saved_filename;
					imwrite(mag_filename, mag_frame);
					printf("Saved MAGNITUDE frame with filename: %s \n", mag_filename.c_str());
				}
				else if (mode == 'l') {
					string bq_filename = "bq_" + saved_filename;
					imwrite(bq_filename, bq_frame);
					printf("Saved BLUR QUANTIZE frame with filename: %s \n", bq_filename.c_str());
				}
				else if (mode == 'f') {
					string faces_filename = "faces_" + saved_filename;
					imwrite(faces_filename, frame);
					printf("Saved RGB frame with boxes drawn around faces with filename: %s \n", faces_filename.c_str());
				}
				else if (mode == 'w') {
					string da2_vis_filename = "da2_vis_" + saved_filename;
					imwrite(da2_vis_filename, da2_dst_vis);
					printf("Saved DAv2's visualization frame with filename: %s \n", da2_vis_filename.c_str());
				}
				else if (mode == 'e') {
					string da2_fog_filename = "da2_fog_" + saved_filename;
					imwrite(da2_fog_filename, fog);
					printf("Saved DEPTH FOG frame with filename: %s \n", da2_fog_filename.c_str());
				}
				else if (mode == 'r') {
					string da2_depth_blur_filename = "da2_depth_blur_" + saved_filename;
					imwrite(da2_depth_blur_filename, fog);
					printf("Saved DEPTH BLUR frame with filename: %s \n", da2_depth_blur_filename.c_str());
				} 
				else if (mode == 't') {
					string emboss_filename = "emboss_" + saved_filename;
					imwrite(emboss_filename, emboss_frame);
					printf("Saved EMBOSS EFFECT frame with filename: %s \n", emboss_filename.c_str());
				}
				else if (mode == 'p') {
					string cface_filename = "cface_" + saved_filename;
					imwrite(cface_filename, cface_frame);
					printf("Saved COLOUR FACE frame with filename: %s \n", cface_filename.c_str());
				}
				else {
					if (neg || br != 10 || con != 10) {
					imwrite(saved_filename, adjust);
					printf("Saved frame with filename: %s \n", saved_filename.c_str());
					} else {
					imwrite(saved_filename, frame);
					printf("Saved frame with filename: %s \n", saved_filename.c_str());
					}
				}
			}
			else {  // Inputting capital 'S' will lead to all frames involved in creating the final display frame will be saved.
				if (mode == 'g') {
					string grs_filename = "grs_" + saved_filename;
					imwrite(grs_filename, grayscale_frame);
					printf("Saved GRAYSCALE frame with filename: %s \n", grs_filename.c_str());

					string rgb_filename = "rgb_" + saved_filename;
					imwrite(rgb_filename, frame);
					printf("Saved RGB frame with filename: %s \n", rgb_filename.c_str());
				}
				else if (mode == 'h') {
					string my_grs_filename = "custom_grs_" + saved_filename;
					imwrite(my_grs_filename, my_grayscale_frame);
					printf("Saved CUSTOM GRAYSCALE frame with filename: %s \n", my_grs_filename.c_str());

					string grs_filename = "grs_" + saved_filename;
					cvtColor(frame, grayscale_frame, COLOR_RGB2GRAY);
					imwrite(grs_filename, grayscale_frame);
					printf("Saved GRAYSCALE frame with filename: %s \n", grs_filename.c_str());

					string rgb_filename = "rgb_" + saved_filename;
					imwrite(rgb_filename, frame);
					printf("Saved RGB frame with filename: %s \n", rgb_filename.c_str());
				}
				else if (mode == 'v') {
					string sepia_filename = "sepia_" + saved_filename;
					imwrite(sepia_filename, sepia_frame);
					printf("Saved SEPIA frame with filename: %s \n", sepia_filename.c_str());

					string rgb_filename = "rgb_" + saved_filename;
					imwrite(rgb_filename, frame);
					printf("Saved RGB frame with filename: %s \n", rgb_filename.c_str());
				}
				else if (mode == 'V') {
					string sepia_filename = "sepia_" + saved_filename;
					imwrite(sepia_filename, sepia_frame);
					printf("Saved SEPIA frame with filename: %s \n", sepia_filename.c_str());

					string vignet_filename = "vignet_" + saved_filename;
					imwrite(vignet_filename, vignetting_frame);
					printf("Saved VIGNET frame with filename: %s \n", vignet_filename.c_str());

					string rgb_filename = "rgb_" + saved_filename;
					imwrite(rgb_filename, frame);
					printf("Saved RGB frame with filename: %s \n", rgb_filename.c_str());
				}
				else if (mode == 'b') {
					string blur_filename = "blur_" + saved_filename;
					imwrite(blur_filename, blur_frame);
					printf("Saved BLUR (created with separable 1x5 filter) frame with filename: %s \n", blur_filename.c_str());

					string rgb_filename = "rgb_" + saved_filename;
					imwrite(rgb_filename, frame);
					printf("Saved RGB frame with filename: %s \n", rgb_filename.c_str());
				}
				else if (mode == 'x') {
					string sobelx_filename = "sobelx_" + saved_filename;
					imwrite(sobelx_filename, sobelx_frame);
					printf("Saved SOBELX frame with filename: %s \n", sobelx_filename.c_str());

					string rgb_filename = "rgb_" + saved_filename;
					imwrite(rgb_filename, frame);
					printf("Saved RGB frame with filename: %s \n", rgb_filename.c_str());
				}
				else if (mode == 'y') {
					string sobely_filename = "sobely_" + saved_filename;
					imwrite(sobely_filename, sobely_frame);
					printf("Saved SOBELY frame with filename: %s \n", sobely_filename.c_str());

					string rgb_filename = "rgb_" + saved_filename;
					imwrite(rgb_filename, frame);
					printf("Saved RGB frame with filename: %s \n", rgb_filename.c_str());
				}
				else if (mode == 'm') {
					string sobelx_filename = "sobelx_" + saved_filename;
					imwrite(sobelx_filename, sobelx_frame);
					printf("Saved SOBELX frame with filename: %s \n", sobelx_filename.c_str());

					string sobely_filename = "sobely_" + saved_filename;
					imwrite(sobely_filename, sobely_frame);
					printf("Saved SOBELY frame with filename: %s \n", sobely_filename.c_str());

					string mag_filename = "mag_" + saved_filename;
					imwrite(mag_filename, mag_frame);
					printf("Saved MAGNITUDE frame with filename: %s \n", mag_filename.c_str());

					string rgb_filename = "rgb_" + saved_filename;
					imwrite(rgb_filename, frame);
					printf("Saved RGB frame with filename: %s \n", rgb_filename.c_str());
				}
				else if (mode == 'l') {
					string bq_filename = "bq_" + saved_filename;
					imwrite(bq_filename, bq_frame);
					printf("Saved BLUR QUANTIZE frame with filename: %s \n", bq_filename.c_str());

					string rgb_filename = "rgb_" + saved_filename;
					imwrite(rgb_filename, frame);
					printf("Saved RGB frame with filename: %s \n", rgb_filename.c_str());
				}
				else if (mode == 'f') {
					string faces_filename = "faces_" + saved_filename;
					imwrite(faces_filename, frame);
					printf("Saved RGB frame with boxes drawn around faces with filename: %s \n", faces_filename.c_str());
				}
				else if (mode == 'w') {
					string da2_output_filename = "da2_output_" + saved_filename;
					imwrite(da2_output_filename, da2_dst);
					printf("Saved DAv2's output frame with filename: %s \n", da2_output_filename.c_str());

					string da2_vis_filename = "da2_vis_" + saved_filename;
					imwrite(da2_vis_filename, da2_dst_vis);
					printf("Saved DAv2's visualization frame with filename: %s \n", da2_vis_filename.c_str());

					string rgb_filename = "rgb_" + saved_filename;
					imwrite(rgb_filename, frame);
					printf("Saved RGB frame with filename: %s \n", rgb_filename.c_str());
				}
				else if (mode == 'e') {
					string da2_output_filename = "da2_output_" + saved_filename;
					imwrite(da2_output_filename, da2_dst);
					printf("Saved DAv2's output frame with filename: %s \n", da2_output_filename.c_str());

					string da2_fog_filename = "da2_fog_" + saved_filename;
					imwrite(da2_fog_filename, fog);
					printf("Saved DEPTH FOG frame with filename: %s \n", da2_fog_filename.c_str());

					string rgb_filename = "rgb_" + saved_filename;
					imwrite(rgb_filename, frame);
					printf("Saved RGB frame with filename: %s \n", rgb_filename.c_str());
				}
				else if (mode == 'r') {
					string da2_output_filename = "da2_output_" + saved_filename;
					imwrite(da2_output_filename, da2_dst);
					printf("Saved DAv2's output frame with filename: %s \n", da2_output_filename.c_str());

					string da2_depth_blur_filename = "da2_depth_blur_" + saved_filename;
					imwrite(da2_depth_blur_filename, fog);
					printf("Saved DEPTH BLUR frame with filename: %s \n", da2_depth_blur_filename.c_str());

					string rgb_filename = "rgb_" + saved_filename;
					imwrite(rgb_filename, frame);
					printf("Saved RGB frame with filename: %s \n", rgb_filename.c_str());
				}
				else if (mode == 't') {
					string emboss_filename = "emboss_" + saved_filename;
					imwrite(emboss_filename, emboss_frame);
					printf("Saved EMBOSS EFFECT frame with filename: %s \n", emboss_filename.c_str());

					string rgb_filename = "rgb_" + saved_filename;
					imwrite(rgb_filename, frame);
					printf("Saved RGB frame with filename: %s \n", rgb_filename.c_str());
				}
				else {
					if (neg || br != 10 || con != 10) {
					imwrite(saved_filename, adjust);
					printf("Saved frame with filename: %s \n", saved_filename.c_str());
					} else {
					imwrite(saved_filename, frame);
					printf("Saved frame with filename: %s \n", saved_filename.c_str());
					}
				}
			}
		}
		else if (key == 'g' || key == 'G') {
			mode = 'g';	
			std::cout << "OpenCV grayscale mode selected" << std::endl;
		}
		else if (key == 'q' || key == 'Q') {
			std::cout << "Exiting video display loop." << std::endl;
			break;
		}
		else if (key == 'h' || key == 'H') {
			mode = 'h';
			std::cout << "Custom implementation of grayscale mode selected." << std::endl;
		}
		else if (key == 'v') {
			mode = 'v';
			std::cout << "Sepia mode selected." << std::endl;
		}
		else if (key == 'V') {
			mode = 'V';
			std::cout << "Vignetting on Sepia mode selected." << std::endl;
		}
		else if (key == 'b' || key == 'B') {
			mode = 'b';
			std::cout << "Blur (Separable 5x5) mode selected." << std::endl;
		}
		else if (key == 'x' || key == 'X') {
			mode = 'x';
			std::cout << "Sobel X mode selected." << std::endl;
		}
		else if (key == 'y' || key == 'Y') {
			mode = 'y';
			std::cout << "Sobel Y mode selected." << std::endl;
		}
		else if (key == 'm' || key == 'M') {
			mode = 'm';
			std::cout << "Magnitude mode selected." << std::endl;
		}
		else if (key == 'l' || key == 'L') {
			mode = 'l';
			std::cout << "Blur Quantize mode selected." << std::endl;
		}
		else if (key == 'f' || key == 'F') {
			mode = 'f';
			std::cout << "Face detection mode selected." << std::endl;
		}
		else if (key == 'w' || key == 'W') {
			mode = 'w';
			std::cout << "Monocular depth analysis mode selected." << std::endl;
		}
		else if (key == 'e' || key == 'E') {
			mode = 'e';
			std::cout << "Depth fog mode selected." << std::endl;
		}
		else if (key == 'r' || key == 'R') {
			mode = 'r';
			std::cout << "Portrait mode selected." << std::endl;
		}
		else if (key == 't' || key == 'T') {
			mode = 't';
			std::cout << "Emboss mode selected." << std::endl;
		}
		else if (key == 'p' || key == 'P') {
			mode = 'p';
			std::cout << "Colourful face mode selected." << std::endl;
		}
		else if (key == 'c' || key == 'C') {
			mode = 'c';
			std::cout << "Normal mode selected." << std::endl;
		}
		else if (key == 'u' || key == 'U') {  // u - Increases br, U - reduces br. Max br level is 20 and min is 1
			if (key == 'u') {
				if (br == 20) {
					std::cout << "Already reached max brightness" << std::endl; 
				} else {
					br += 1;
				}
			}
			else {
				if (br == 1) {
					std::cout << "Already reached min brightness" << std::endl;
				} else {
					br -= 1;
				}
			}
			std::cout << "Brightness = "<< br << std::endl;
		}
		else if (key == 'i' || key == 'I') {  // i - Increases con, I - reduces con. Max con level is 15 and min is 5
			if (key == 'i') {
				if (con == 15) {
					std::cout << "Already reached max contrast" << std::endl; 
				} else {
					con += 1;
				}
			}
			else {
				if (con == 5) {
					std::cout << "Already reached min contrast" << std::endl;
				} else {
					con -= 1;
				}
			}
			std::cout << "Contrast = "<< con << std::endl;
		}
		else if (key == 'o' || key == 'O') {  // Toggle between negative mode
			neg = !neg;
			std::cout << "Negative mode switched on = "<< neg << std::endl;
		}

	}
	cv::destroyWindow("Video Display");
	return 0;
}

// Destructor of the VideoDisplay class. deletes both its pointers
VideoDisplay::~VideoDisplay() {
	delete this->vd_cap;
	delete this->da_net;
}

// Main function
int main(int argc, char *argv[]) {
	VideoDisplay vid_display;
	vid_display.start_loop();

	return 0;
	// VideoCapture *vd_cap;
	// Mat frame;
	// Mat grayscale_frame;

	// // Opening the video device
	// int device_id = 0;
	// // int api_id = CAP_ANY;
	// int api_id = CAP_AVFOUNDATION;
	// vd_cap = new VideoCapture(device_id, api_id);
	// if (!vd_cap->isOpened()) {
	// 	printf("Unable to open video device\n");
	// 	return -1;
	// }

	// // Video device has successfully been opened.
	// Size refS( (int) vd_cap->get(CAP_PROP_FRAME_WIDTH),
	// 		   (int) vd_cap->get(CAP_PROP_FRAME_HEIGHT) );

	// printf("Expected size: %d %d \n", refS.width, refS.height);

	// namedWindow("Video", 1);

	// // For Task 10: Detect faces in an image.
	// Mat grey;
	// vector<Rect> faces;
	// Rect last(0, 0, 0, 0);
	// // Task 10

	// // For Task 11: Depth anything network
	// Mat da2_src;
	// Mat da2_dst;
	// Mat da2_dst_vis;
	// const float reduction = 0.5;
	// DA2Network da_net("model_fp16.onnx");
	// float scale_factor = 256.0 / (refS.height * reduction);
	// printf("Will use scale factor %.2f \n", scale_factor);
	// // Task 11

	// bool grayscale_selected = false;
	// bool custom_selected = false;
	// char key_pressed = 'c';
	// 
	// for(;;) {  // Running an infinite loop
	// 	*vd_cap >> frame;  // Using the VideoCapture as a stream. Get a frame from it.
	// 	if (frame.empty()) {
	// 		printf("Frame is empty.");
	// 		break;
	// 	}

	// 	// RGB[A] to Gray: Y <- 0.299xR + 0.587xG + 0.114xB
	// 	// Gray to RGB[A]: R <- Y, G <- Y, B <- Y, A <- max(Channel range)
	// 	if (key_pressed == 'f') {
	// 		cvtColor(frame, grey, COLOR_BGR2GRAY, 0);  // Converting image to greyscale.
	// 		detectFaces(grey, faces);  // Detect faces.
	// 		drawBoxes(frame, faces);  // Draw boxes around the faces
	// 		if (faces.size() > 0) {
	// 			last.x = (faces[0].x + last.x)/2;	
	// 			last.y = (faces[0].y + last.y)/2;	
	// 			last.width = (faces[0].width + last.width)/2;	
	// 			last.height = (faces[0].height + last.height)/2;	
	// 		}
	// 		imshow("Video", frame);
	// 	}
	// 	else if (key_pressed == 'w') {
	// 		resize(frame, da2_src, Size(), reduction, reduction);  // for speed, reduce the size of the input by half
	// 		da_net.set_input(da2_src, scale_factor);  // set the network input
	// 		da_net.run_network(da2_dst, da2_src.size());  // run the network
	// 		applyColorMap(da2_dst, da2_dst_vis, COLORMAP_INFERNO);
	// 		imshow("Video", da2_dst_vis);
	// 	}
	// 	else if (key_pressed == 'e') {
	// 		resize(frame, da2_src, Size(), reduction, reduction);  // for speed, reduce the size of the input by half
	// 		da_net.set_input(da2_src, scale_factor);  // set the network input
	// 		da_net.run_network(da2_dst, da2_src.size());  // run the network
	// 		// applyColorMap(da2_dst, da2_dst_vis, COLORMAP_INFERNO);
	// 		Mat fog;
	// 		depth_fog(da2_src, da2_dst, fog);
	// 		imshow("Video", fog);
	// 	}
	// 	else if (key_pressed == 'r') {
	// 		resize(frame, da2_src, Size(), reduction, reduction);  // for speed, reduce the size of the input by half
	// 		da_net.set_input(da2_src, scale_factor);  // set the network input
	// 		da_net.run_network(da2_dst, da2_src.size());  // run the network
	// 		// applyColorMap(da2_dst, da2_dst_vis, COLORMAP_INFERNO);
	// 		Mat fog;
	// 		portrait_mode(da2_src, da2_dst, fog);
	// 		imshow("Video", fog);
	// 	}
	// 	else if (grayscale_selected) {
	// 		cvtColor(frame, grayscale_frame, COLOR_RGB2GRAY);
	// 		imshow("Video", grayscale_frame);
	// 	}
	// 	else {
	// 		imshow("Video", frame);
	// 	}

	// 	char key = waitKey(10);
	// 	if (key == 's' || key == 'S') {
	// 		auto frame_instant = chrono::system_clock::now().time_since_epoch();
	// 		auto frame_instant_str = chrono::duration_cast<chrono::seconds>(frame_instant).count();
	// 		string saved_filename = "image_capture_" + to_string(frame_instant_str) + ".png";
	// 		if (key == 's') {
	// 			imwrite(saved_filename, frame);
	// 			printf("Saved frame with filename: %s \n", saved_filename.c_str());
	// 		}
	// 		else {
	// 			string grs_filename = "grs_" + saved_filename;
	// 			imwrite(grs_filename, grayscale_frame);
	// 			printf("Saved GRAYSCALE frame with filename: %s \n", grs_filename.c_str());
	// 			string rgb_filename = "rgb_" + saved_filename;
	// 			imwrite(rgb_filename, frame);
	// 			printf("Saved RGB frame with filename: %s \n", rgb_filename.c_str());
	// 		}
	// 	}
	// 	else if (key == 'g' || key == 'G') {
	// 		grayscale_selected = !grayscale_selected;	
	// 	}
	// 	else if (key == 'q' || key == 'Q') {
	// 		break;
	// 	}
	// 	else if (key == 'f' || key == 'F') {
	// 		key_pressed = 'f';
	// 	}
	// 	else if (key == 'w') {
	// 		key_pressed = 'w';
	// 	}
	// }
	// delete vd_cap;
	// return 0;
}
