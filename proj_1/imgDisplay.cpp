// Gautam Ajey Khanapuri
// 19 January 2026
// File for opening and displaying images in a loop till 'q' is pressed.

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>
#include <string>

using namespace cv;

// Function runs an infinite while loop which takes a image file path(path cannot contain whitespaces. Spaces have to be escaped) from the command line and displays it in a window. Waits till a keystroke is entered. If the entered key is 'q' or 'Q'. The loop ends and function returns.
// Params - None
// Returns - int. 0 incase of no error. 1 if any error.
int main() {
	std::cout << "Started!" << std::endl;
	std::string file_path;
	while (true) {
		std::cout << "Enter file path of the image you want to view. Press q to exit image display program: ";
		std::cin >> file_path;

		if(file_path == "q" || file_path == "Q") {
			return 0;
		}

		Mat image = imread(file_path, IMREAD_COLOR);
		if (image.empty()) {
			std::cout << "Could not read the image: '" << file_path << "'. Try again or press 'q'." << std::endl;
			continue;
		}

		imshow("Display Image", image);
		int n = waitKey(0); // Wait for the user to press a key.

		destroyWindow("Display Image");
		// destroyAllWindows();
		if (n == 'q' || n == 'Q') {
			return 0;
		}
	}
	return 0;
}
