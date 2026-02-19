// Gautam Ajey Khanapuri
// 20 January 2026
// Helper file which contains methods that implement different kinds of filters.


#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include <iostream>
#include <string>
#include <cmath>

#include "DA2Network.hpp"

using namespace cv;


const int GAUSSIAN_5x5[25] = {
1, 2, 4, 2, 1, 
2, 4, 8, 4, 2,
4, 8, 16, 8, 4,
2, 4, 8, 4, 2,
1, 2, 4, 2, 1
};

const int SOBEL_g[3] = {1, 2, 1};
const int SOBEL_h[3] = {-1, 0, 1};
const int SOBEL_r[3] = {1, 0, -1};

const int GAUSS_SEP[5] = {1, 2, 4, 2, 1};

// **The documentation of all methods in this class are written in the heder file.**



// int greyscale(Mat &src, Mat &dst, int option=0);  // Function declaration with default parameter.
// int blur5x5_1( Mat &src, Mat &dst);
// int blur5x5_2( Mat &src, Mat &dst);
// int sobelY3x3(Mat &src, Mat &dst);
// int sobelY3x3(Mat &src, Mat &dst);
// int blurQuantize(Mat &src, Mat &dst, int levels=10);
Vec3b gauss_blur_pix(Mat &src, int i, int j);
Vec3b gauss_separable_pix(Mat &src, int i, int j);
Vec3s apply_sobelX(Mat &src, int i, int j);
Vec3s apply_sobelY(Mat &src, int i, int j);
// int quantize(int num, int b);


int greyscale(Mat &src, Mat &dst, int option) {
	dst.create(src.rows, src.cols, CV_8UC3);
	for (int i=0; i<src.rows; i++) {
		Vec3b *ptr = src.ptr<Vec3b>(i);  // Getting a pointer to ith row of the source
		Vec3b *dst_ptr = dst.ptr<Vec3b>(i);  // Getting a pointer to ith row of the destination
		for (int j=0; j<src.cols; j++) {
			uchar b = ptr[j][0];
			uchar g = ptr[j][1];
			uchar r = ptr[j][2];

			uchar y;
			if (option == 0) {
				y = (uchar)((b + g + r) / 3);
			} else {
				y = (uchar)(255 - r);  // TODO: Can add different methods of finding y.
			}
			dst_ptr[j] = Vec3b(y, y, y);
		}
	}
	return 0;
}

// int greyscale_option0(Mat &src, Mat &dst) {
// 	dst.create(src.rows, src.cols, CV_8UC1);
// 	for (int i=0; i<src.rows; i++) {
// 		for (int j=0; j<src.cols; j++) {
// 			uchar b = src.at<Vec3b>(i, j)[0];
// 			uchar g = src.at<Vec3b>(i, j)[1];
// 			uchar r = src.at<Vec3b>(i, j)[2];
// 
// 			uchar y = (b + g + r)/3;
// 			dst.at<uchar>(i, j) = y;
// 		}
// 	}
// 	return 0;
// }

// int greyscale_option1(Mat &src, Mat &dst) {
// 	dst.create(src.rows, src.cols, CV_8UC1);
// 	for (int i=0; i<src.rows; i++) {
// 		Vec3b *ptr = src.ptr<Vec3b>(i);  // Getting a pointer to ith row of the source
// 		uchar *dst_ptr = dst.ptr<uchar>(i);  // Getting a pointer to ith row of the destination
// 		for (int j=0; j<src.cols; j++) {
// 			uchar b = ptr[j][0];
// 			uchar g = ptr[j][1];
// 			uchar r = ptr[j][2];
// 
// 			uchar y = (b + g + r)/3;
// 			dst_ptr[j] = y;
// 		}
// 	}
// 	return 0;
// }

int sepia(Mat &src, Mat &dst) {
	dst.create(src.rows, src.cols, CV_8UC3);

	for (int i=0; i<src.rows; i++) {
		const Vec3b *src_ptr = src.ptr<Vec3b>(i);
		Vec3b *dst_ptr = dst.ptr<Vec3b>(i);
		for (int j=0; j<src.cols; j++) {
			uchar b = src_ptr[j][0];
			uchar g = src_ptr[j][1];
			uchar r = src_ptr[j][2];

			int dst_b = (0.272 * r) + (0.534 * g) + (0.131 * b);
			int dst_g = (0.349 * r) + (0.686 * g) + (0.168 * b);
			int dst_r = (0.393 * r) + (0.769 * g) + (0.189 * b);

			dst_b = dst_b > 255 ? 255 : dst_b;
			dst_g = dst_g > 255 ? 255 : dst_g;
			dst_r = dst_r > 255 ? 255 : dst_r;

			dst_ptr[j][0] = (uchar) dst_b;
			dst_ptr[j][1] = (uchar) dst_g;
			dst_ptr[j][2] = (uchar) dst_r;
		}
	}
	return 0;  //TODO: Add vignetting (image getting darker towards the feature).
}

int vignetting(Mat &src, Mat &dst, float threshold, float strength) {
	dst.create(src.rows, src.cols, CV_8UC3);
	
	for (int i=0; i < src.rows; i++) {
		Vec3b *src_ptr = src.ptr<Vec3b>(i);
		Vec3b *dst_ptr = dst.ptr<Vec3b>(i);
		for (int j=0; j < src.cols; j++) {
			float factor = 1.0;
			float top = i / float (src.rows);
			float bottom = (src.rows - i) / float (src.rows);
			float left = j / float (src.cols);
			float right = (src.cols - j) / float (src.cols);

			float vert = std::min(top, bottom);
			float hori = std::min(left, right);
			float min_dist = std::min(vert, hori);

			if (min_dist <= (1.0f - threshold)) {
				float t = min_dist / (1.0f - threshold);
				factor = t + (1.0f - t) * (1.0f - strength);
			}
			dst_ptr[j][0] = src_ptr[j][0] * factor;
			dst_ptr[j][1] = src_ptr[j][1] * factor;
			dst_ptr[j][2] = src_ptr[j][2] * factor;
		}
	}
	return 0;
}

// Naive 5x5 blur filter.
// Each color stream is blurred separately.
// Paramters: src - original frame passed by reference, dst - the destination frame where the blurred should be placed.
// Returns - int - 0 in case the blurring process takes place without any error.
int blur5x5_1( Mat &src, Mat &dst ) {
//	dst.create(src.rows, src.cols, CV_8UC3);
//	for (int i=0; i<src.rows; i++) {
//		if (i < 2 || i > src.rows - 3) {
//			src.row(i).copyTo(dst.row(i));
//		}
//	}
//	for (int j=0; j<src.cols; j++) {
//		if (j < 2 || j > src.cols - 3) {
//			src.col(j).copyTo(dst.col(j));
//		}
//	}
	src.copyTo(dst);
	for (int i=2; i < src.rows - 2; i++) {
		for (int j=2; j < src.cols - 2; j++) {
//			Vec3b src_pixel_ij = src.at<Vec3b>(i, j);
			Vec3b dst_pixel_ij = gauss_blur_pix(src, i, j);
			dst.at<Vec3b>(i, j) = dst_pixel_ij;
		}
	}
	return 0;
}

int blur5x5_2( Mat &src, Mat &dst ) {
	dst.create(src.rows, src.cols, CV_8UC3);
//	for (int i=0; i<src.rows; i++) {
//		if (i < 2 || i > src.rows - 3) {
//			src.row(i).copyTo(dst.row(i));
//		}
//	}
//	for (int j=0; j<src.cols; j++) {
//		if (j < 2 || j > src.cols - 3) {
//			src.col(j).copyTo(dst.col(j));
//		}
//	}

	for (int i=2; i < src.rows - 2; i++) {
		// Vec3b *ptr = src.ptr<Vec3b>(i);  // Getting a pointer to ith row of the source
		Vec3b *dst_ptr = dst.ptr<Vec3b>(i);
		for (int j=2; j < src.cols - 2; j++) {
			// Vec3b dst_pixel_ij = gauss_separable_pix(src, i, j);
			Vec3b horizontal[5];
			// Horizontal pass
            for (int k=i-2; k <= i+2; k++) {
                float b_h = 0.0;
                float g_h = 0.0;
                float r_h = 0.0;
                Vec3b *row_ptr = src.ptr<Vec3b>(k);
                for (int y=j-2; y<=j+2; y++) {
                    b_h += row_ptr[y][0] * GAUSS_SEP[y-j+2];
                    g_h += row_ptr[y][1] * GAUSS_SEP[y-j+2];
                    r_h += row_ptr[y][2] * GAUSS_SEP[y-j+2];
                }
                b_h = b_h / 10.0;
                g_h = g_h / 10.0;
                r_h = r_h / 10.0;
                horizontal[k-i+2] = Vec3b((uchar)b_h, (uchar)g_h, (uchar)r_h);
            }

            // Vertical pass
            float b_tmp = 0.0;
            float g_tmp = 0.0;
            float r_tmp = 0.0;

            for (int a=0; a<5; a++) {
                b_tmp += horizontal[a][0] * GAUSS_SEP[a];
                g_tmp += horizontal[a][1] * GAUSS_SEP[a];
                r_tmp += horizontal[a][2] * GAUSS_SEP[a];
            }

            uchar b = (uchar)(b_tmp/10);
            uchar g = (uchar)(g_tmp/10);
            uchar r = (uchar)(r_tmp/10);
			dst_ptr[j] = Vec3b(b, g, r);
		}
	}
	return 0;
}

Vec3b gauss_separable_pix(Mat &src, int i, int j) {
	Vec3b horizontal[5];

	for (int k=i-2; k <= i+2; k++) {
		float b_h = 0.0;
		float g_h = 0.0;
		float r_h = 0.0;
		Vec3b *row_ptr = src.ptr<Vec3b>(k);
		for (int y=j-2;y<=j+2;y++) {
			b_h += row_ptr[y][0] * GAUSS_SEP[y-j+2];
			g_h += row_ptr[y][1] * GAUSS_SEP[y-j+2];
			r_h += row_ptr[y][2] * GAUSS_SEP[y-j+2];
		}
		b_h = b_h / 10.0;
		g_h = g_h / 10.0;
		r_h = r_h / 10.0;
		horizontal[k-i+2] = Vec3b((uchar)b_h, (uchar)g_h, (uchar)r_h);
	}

	float b_tmp = 0.0;
	float g_tmp = 0.0;
	float r_tmp = 0.0;

	for (int a=0; a<5; a++) {
		b_tmp += horizontal[a][0] * GAUSS_SEP[a];
		g_tmp += horizontal[a][1] * GAUSS_SEP[a];
		r_tmp += horizontal[a][2] * GAUSS_SEP[a];
	}

	uchar b = (uchar)(b_tmp/10);
	uchar g = (uchar)(g_tmp/10);
	uchar r = (uchar)(r_tmp/10);

	Vec3b resulting_pix(b, g, r);
	return resulting_pix;
}

Vec3b gauss_blur_pix(Mat &src, int i, int j) {
	Vec3b five_by_five_src[25];
	int c = 0;
	for (int x=i-2; x<=i+2; x++) {
		for (int y=j-2; y<=j+2; y++) {
			five_by_five_src[c] = src.at<Vec3b>(x, y); 
			c++;
		}
	}

	float b_tmp = 0.0;
	float g_tmp = 0.0;
	float r_tmp = 0.0;

	for (int k=0; k<25; k++) {
		b_tmp += five_by_five_src[k][0] * GAUSSIAN_5x5[k];
		g_tmp += five_by_five_src[k][1] * GAUSSIAN_5x5[k];
		r_tmp += five_by_five_src[k][2] * GAUSSIAN_5x5[k];
	}

	uchar b = (uchar)(b_tmp/100);
	uchar g = (uchar)(g_tmp/100);
	uchar r = (uchar)(r_tmp/100);

	Vec3b resulting_pix(b, g, r);
	return resulting_pix;
	//Vec3b pix1 = src.at<Vec3b>(i-2, j-2);
	//Vec3b pix2 = src.at<Vec3b>(i-2, j-1);
	//Vec3b pix3 = src.at<Vec3b>(i-2, j);
	//Vec3b pix4 = src.at<Vec3b>(i-2, j+1);
	//Vec3b pix5 = src.at<Vec3b>(i-2, j+2);
	//Vec3b pix6 = src.at<Vec3b>(i-1, j-2);
	//Vec3b pix7 = src.at<Vec3b>(i-1, j-1);
	//Vec3b pix8 = src.at<Vec3b>(i-1, j);
	//Vec3b pix9 = src.at<Vec3b>(i-1, j+1);
	//Vec3b pix10 = src.at<Vec3b>(i-1, j+2);
	//Vec3b pix11 = src.at<Vec3b>(i, j-2);
	//Vec3b pix12 = src.at<Vec3b>(i, j-1);
	//Vec3b pix13 = src.at<Vec3b>(i, j);
	//Vec3b pix14 = src.at<Vec3b>(i, j+1);
	//Vec3b pix15 = src.at<Vec3b>(i, j+2);
	//Vec3b pix16 = src.at<Vec3b>(i+1, j-2);
	//Vec3b pix17 = src.at<Vec3b>(i+1, j-1);
	//Vec3b pix18 = src.at<Vec3b>(i+1, j);
	//Vec3b pix19 = src.at<Vec3b>(i+1, j+1);
	//Vec3b pix20 = src.at<Vec3b>(i+1, j+2);
	//Vec3b pix21 = src.at<Vec3b>(i+2, j-2);
	//Vec3b pix22 = src.at<Vec3b>(i+2, j-1);
	//Vec3b pix23 = src.at<Vec3b>(i+2, j);
	//Vec3b pix24 = src.at<Vec3b>(i+2, j+1);
	//Vec3b pix25 = src.at<Vec3b>(i+2, j+2);

	//Vec3b five_by_five_src[25] = {pix1, pix2, pix3, pix4, pix5, pix6, pix7, pix8, pix9, pix10, pix11, pix12, pix13, pix14, pix15, pix16, pix17, pix18, pix19, pix20, pix21, pix22, pix23, pix24, pix25};
}

int sobelX3x3(Mat &src, Mat &dst) {
	dst.create(src.rows, src.cols, CV_16SC3);
	int r = src.rows;
	int c = src.cols;
	for (int i=0;i<src.rows;i++) {
		Vec3s *dst_ptr = dst.ptr<Vec3s>(i);
		Vec3b *top = nullptr;
		Vec3b *bot = nullptr;
		Vec3b *mid = src.ptr<Vec3b>(i);
		if (i == 0) {
			top = src.ptr<Vec3b>(i);
			bot = src.ptr<Vec3b>(i+1);
		} else if (i == r - 1) {
			top = src.ptr<Vec3b>(i-1);
			bot = src.ptr<Vec3b>(i);
		} else {
			top = src.ptr<Vec3b>(i-1);
			bot = src.ptr<Vec3b>(i+1);
		}
		Vec3b* rows[3] = {top, mid, bot};
		for (int j=0;j<src.cols;j++) {
			// Vec3s pixel = apply_sobelX(src, i, j);

			int left = -1;
			int right = -1;
			int middle = j;
			if (j==0) {
				left = j;
				right = j+1;
			} else if (j==c-1) {
				left = j-1;
				right = j;
			} else {
				left = j-1;
				right = j+1;
			}

			Vec3s vertical_result[3];

			int column_indexes[3] = {left, middle, right};
			for (int k=0; k<=2; k++) {
				int column = column_indexes[k];
				float b_t = 0;
				float g_t = 0;
				float r_t = 0;
				for (int x=0; x<=2; x++) {
					b_t += rows[x][column][0] * SOBEL_g[x];
					g_t += rows[x][column][1] * SOBEL_g[x];
					r_t += rows[x][column][2] * SOBEL_g[x];
				}
				signed short b = (signed short) (b_t/4);
				signed short g = (signed short) (g_t/4);
				signed short r = (signed short) (r_t/4);

				Vec3s tmp(b, g, r);
				vertical_result[k] = tmp;
			}

			signed short b_final = 0;
			signed short g_final = 0;
			signed short r_final = 0;
			for (int a=0; a<=2; a++) {
				b_final += vertical_result[a][0] * SOBEL_h[a];
				g_final += vertical_result[a][1] * SOBEL_h[a];
				r_final += vertical_result[a][2] * SOBEL_h[a];
			}

			Vec3s return_pixel(b_final, g_final, r_final);
			dst_ptr[j] = return_pixel;
		}
	}
	// convertScaleAbs(dst, dst);
	return 0;
}

Vec3s apply_sobelX(Mat &src, int i, int j) {
	int r = src.rows;
	int c = src.cols;
	
	Vec3b *top = nullptr;
	Vec3b *bot = nullptr;
	Vec3b *mid = src.ptr<Vec3b>(i);
	if (i == 0) {
		top = src.ptr<Vec3b>(i);
		bot = src.ptr<Vec3b>(i+1);
	} else if (i == r - 1) {
		top = src.ptr<Vec3b>(i-1);
		bot = src.ptr<Vec3b>(i);
	} else {
		top = src.ptr<Vec3b>(i-1);
		bot = src.ptr<Vec3b>(i+1);
	}

	int left = -1;
	int right = -1;
	int middle = j;
	if (j==0) {
		left = j;
		right = j+1;
	} else if (j==c-1) {
		left = j-1;
		right = j;
	} else {
		left = j-1;
		right = j+1;
	}

	Vec3s vertical_result[3];
	Vec3b* rows[3] = {top, mid, bot};

	int column_indexes[3] = {left, middle, right};
	for (int k=0; k<=2; k++) {
		int column = column_indexes[k];
		float b_t = 0;
		float g_t = 0;
		float r_t = 0;
		for (int x=0; x<=2; x++) {
			b_t += rows[x][column][0] * SOBEL_g[x];
			g_t += rows[x][column][1] * SOBEL_g[x];
			r_t += rows[x][column][2] * SOBEL_g[x];
		}
		signed short b = (signed short) (b_t/4);
		signed short g = (signed short) (g_t/4);
		signed short r = (signed short) (r_t/4);

		Vec3s tmp(b, g, r);
		vertical_result[k] = tmp;
	}

	signed short b_final = 0;
	signed short g_final = 0;
	signed short r_final = 0;
	for (int a=0; a<=2; a++) {
		b_final += vertical_result[a][0] * SOBEL_h[a];
		g_final += vertical_result[a][1] * SOBEL_h[a];
		r_final += vertical_result[a][2] * SOBEL_h[a];
	}

	Vec3s return_pixel(b_final, g_final, r_final);
	return return_pixel;
}

int sobelY3x3(Mat &src, Mat &dst) {
	dst.create(src.rows, src.cols, CV_16SC3);
	int r = src.rows;
	int c = src.cols;
	for (int i=0;i<src.rows;i++) {
		Vec3s *dst_ptr = dst.ptr<Vec3s>(i);
		Vec3b *top = nullptr;
		Vec3b *bot = nullptr;
		Vec3b *mid = src.ptr<Vec3b>(i);
		if (i == 0) {
			top = src.ptr<Vec3b>(i);
			bot = src.ptr<Vec3b>(i+1);
		} else if (i == r - 1) {
			top = src.ptr<Vec3b>(i-1);
			bot = src.ptr<Vec3b>(i);
		} else {
			top = src.ptr<Vec3b>(i-1);
			bot = src.ptr<Vec3b>(i+1);
		}
		for (int j=0;j<src.cols;j++) {
			// Vec3s pixel = apply_sobelY(src, i, j);
			int left = j;
			int right = j;
			int middle = j;
			if (j==0) {
				left = j;
				right = j+1;
			} else if (j==c-1) {
				left = j-1;
				right = j;
			} else {
				left = j-1;
				right = j+1;
			}

			Vec3s vertical_result[3];
			Vec3b* rows[3] = {top, mid, bot};

			int column_indexes[3] = {left, middle, right};
			for (int k=0; k<=2; k++) {
				int column = column_indexes[k];
				int b_t = 0;
				int g_t = 0;
				int r_t = 0;
				for (int x=0; x<=2; x++) {
					b_t += rows[x][column][0] * SOBEL_r[x];
					g_t += rows[x][column][1] * SOBEL_r[x];
					r_t += rows[x][column][2] * SOBEL_r[x];
				}
				// signed short b = (signed short) (b_t/4);
				// signed short g = (signed short) (g_t/4);
				// signed short r = (signed short) (r_t/4);

				// Vec3b tmp(b, g, r);
				Vec3s tmp(b_t, g_t, r_t);
				vertical_result[k] = tmp;
			}

			float b_f = 0.0;
			float g_f = 0.0;
			float r_f = 0.0;
			for (int a=0; a<=2; a++) {
				b_f += vertical_result[a][0] * SOBEL_g[a];
				g_f += vertical_result[a][1] * SOBEL_g[a];
				r_f += vertical_result[a][2] * SOBEL_g[a];
			}

			signed short b_final = (signed short) (b_f/4);
			signed short g_final = (signed short) (g_f/4);
			signed short r_final = (signed short) (r_f/4);

			Vec3s return_pixel(b_final, g_final, r_final);
			dst_ptr[j] = return_pixel;
		}
	}
	// convertScaleAbs(dst, dst);
	return 0;
}

Vec3s apply_sobelY(Mat &src, int i, int j) {
	int r = src.rows;
	int c = src.cols;
	
	Vec3b *top = nullptr;
	Vec3b *bot = nullptr;
	Vec3b *mid = src.ptr<Vec3b>(i);
	if (i == 0) {
		top = src.ptr<Vec3b>(i);
		bot = src.ptr<Vec3b>(i+1);
	} else if (i == r - 1) {
		top = src.ptr<Vec3b>(i-1);
		bot = src.ptr<Vec3b>(i);
	} else {
		top = src.ptr<Vec3b>(i-1);
		bot = src.ptr<Vec3b>(i+1);
	}

	int left = j;
	int right = j;
	int middle = j;
	if (j==0) {
		left = j;
		right = j+1;
	} else if (j==c-1) {
		left = j-1;
		right = j;
	} else {
		left = j-1;
		right = j+1;
	}

	Vec3s vertical_result[3];
	Vec3b* rows[3] = {top, mid, bot};

	int column_indexes[3] = {left, middle, right};
	for (int k=0; k<=2; k++) {
		int column = column_indexes[k];
		int b_t = 0;
		int g_t = 0;
		int r_t = 0;
		for (int x=0; x<=2; x++) {
			b_t += rows[x][column][0] * SOBEL_r[x];
			g_t += rows[x][column][1] * SOBEL_r[x];
			r_t += rows[x][column][2] * SOBEL_r[x];
		}
		// signed short b = (signed short) (b_t/4);
		// signed short g = (signed short) (g_t/4);
		// signed short r = (signed short) (r_t/4);

		// Vec3b tmp(b, g, r);
		Vec3s tmp(b_t, g_t, r_t);
		vertical_result[k] = tmp;
	}

	float b_f = 0.0;
	float g_f = 0.0;
	float r_f = 0.0;
	for (int a=0; a<=2; a++) {
		b_f += vertical_result[a][0] * SOBEL_g[a];
		g_f += vertical_result[a][1] * SOBEL_g[a];
		r_f += vertical_result[a][2] * SOBEL_g[a];
	}

	signed short b_final = (signed short) (b_f/4);
	signed short g_final = (signed short) (g_f/4);
	signed short r_final = (signed short) (r_f/4);

	Vec3s return_pixel(b_final, g_final, r_final);
	return return_pixel;
}

int magnitude(Mat &sx, Mat &sy, Mat &dst) {
	int r = sx.rows;
	int c = sx.cols;
	dst.create(sx.rows, sx.cols, CV_8UC3);

	for (int i=0; i<r; i++) {
	Vec3s *x_ptr = sx.ptr<Vec3s>(i);
	Vec3s *y_ptr = sy.ptr<Vec3s>(i);
	Vec3b *dst_ptr = dst.ptr<Vec3b>(i);
		for (int j=0; j<c; j++) {
			Vec3s x_val = x_ptr[j];
			Vec3s y_val = y_ptr[j];
			
			float b_t = std::sqrt((x_val[0] * x_val[0]) + (y_val[0] * y_val[0])); 
			float g_t = std::sqrt((x_val[1] * x_val[1]) + (y_val[1] * y_val[1])); 
			float r_t = std::sqrt((x_val[2] * x_val[2]) + (y_val[2] * y_val[2])); 

			uchar b = (uchar) (b_t<256 ? b_t : 255);
			uchar g = (uchar) (g_t<256 ? g_t : 255);
			uchar r = (uchar) (r_t<256 ? r_t : 255);

			dst_ptr[j] = Vec3b(b, g, r); 
		}
	}
	return 0;
}

int blurQuantize(Mat &src, Mat &blur, Mat &dst, int levels) {
	blur.create(src.rows, src.cols, CV_8UC3);
	blur5x5_2(src, blur);
	int bucket = 255/levels;

	dst.create(src.rows, src.cols, CV_8UC3);
	for (int a=0; a<blur.rows;a++) {
		Vec3b *ptr = blur.ptr<Vec3b>(a);
		Vec3b *dst_ptr = dst.ptr<Vec3b>(a);
		for (int c=0; c<blur.cols;c++) {
			int bt = ptr[c][0] / bucket;
			int bh = bt * bucket;
			uchar b = (uchar) bh;

			int gt = ptr[c][1] / bucket;
			int gh = gt * bucket;
			uchar g = (uchar) gh;
			 
			int rt = ptr[c][2] / bucket;
			int rh = rt * bucket;
			uchar r = (uchar) rh;

			dst_ptr[c] = Vec3b(b, g, r);
		}
	}
	return 0;
}

int depth_fog(Mat &src, Mat &depth, Mat &dst) {
	Vec3b fog_colour(255, 255, 255);
	dst.create(src.rows, src.cols, CV_8UC3);

	for (int i=0; i<src.rows; i++) {
		Vec3b *src_ptr = src.ptr<Vec3b>(i);
		uchar *depth_ptr = depth.ptr<uchar>(i);
		Vec3b *dst_ptr = dst.ptr<Vec3b>(i);

		for (int j=0; j<src.cols; j++) {
			float fog_amount = depth_ptr[j] / 255.0f;
			fog_amount = fog_amount * fog_amount;

			// dst_ptr[j][0] = src_ptr[j][0] * (1 - fog_amount) + fog_colour[0] * fog_amount;  // Linear interpolation formula. original * (1- effect portion) + effect_frame * effect_portion.
			// dst_ptr[j][1] = src_ptr[j][1] * (1 - fog_amount) + fog_colour[1] * fog_amount;
			// dst_ptr[j][2] = src_ptr[j][2] * (1 - fog_amount) + fog_colour[2] * fog_amount;
			dst_ptr[j][0] = src_ptr[j][0] * fog_amount + fog_colour[0] * (1 - fog_amount);  // Linear interpolation formula. original * (1- effect portion) + effect_frame * effect_portion.
			dst_ptr[j][1] = src_ptr[j][1] * fog_amount + fog_colour[1] * (1 - fog_amount);
			dst_ptr[j][2] = src_ptr[j][2] * fog_amount + fog_colour[2] * (1 - fog_amount);
		}
	}
	return 0;
}

int portrait_mode(Mat &src, Mat &depth, Mat &dst) {
	Mat blurred;
	// blur5x5_2(src, blurred);  // Using my own 5x5 seprable gaussian blur filter. If it is not goog, use opencv's functions
	
	GaussianBlur(src, blurred, Size(21, 21), 0);  // USING OPENCV FILTER BECAUSE my own 5x5 filter does not blur out the first two cols and rows.
	dst.create(src.rows, src.cols, CV_8UC3);

	for (int i=0; i<src.rows; i++) {
		Vec3b *src_ptr = src.ptr<Vec3b>(i);
		Vec3b *blur_ptr = blurred.ptr<Vec3b>(i);
		uchar *depth_ptr = depth.ptr<uchar>(i);
		Vec3b *dst_ptr = dst.ptr<Vec3b>(i);

		for (int j=0; j<src.cols; j++) {
			float blur_amount = depth_ptr[j] / 255.0f;
			blur_amount = blur_amount * sqrt(blur_amount);

			// dst_ptr[j][0] = src_ptr[j][0] * (1 - blur_amount) + blur_ptr[j][0] * blur_amount;  //Linear interpolation
			// dst_ptr[j][1] = src_ptr[j][1] * (1 - blur_amount) + blur_ptr[j][1] * blur_amount;  //Linear interpolation
			// dst_ptr[j][2] = src_ptr[j][2] * (1 - blur_amount) + blur_ptr[j][2] * blur_amount;  //Linear interpolation
			dst_ptr[j][0] = src_ptr[j][0] * blur_amount + blur_ptr[j][0] * (1 - blur_amount);  //Linear interpolation
			dst_ptr[j][1] = src_ptr[j][1] * blur_amount + blur_ptr[j][1] * (1 - blur_amount);  //Linear interpolation
			dst_ptr[j][2] = src_ptr[j][2] * blur_amount + blur_ptr[j][2] * (1 - blur_amount);  //Linear interpolation
		}
	}
	return 0;
}

int emboss(cv::Mat &src, cv::Mat &dst, float x, float y) {
	cv:: Mat sx;
	cv:: Mat sy;
	sobelX3x3(src, sx);
	sobelY3x3(src, sy);

	dst.create(src.rows, src.cols, CV_8UC3);
	for (int i=0; i < src.rows; i++) {
		Vec3s *sx_ptr = sx.ptr<Vec3s>(i);
		Vec3s *sy_ptr = sy.ptr<Vec3s>(i);
		Vec3b *dst_ptr = dst.ptr<Vec3b>(i);

		for (int j=0; j < src.cols; j++) {
			float b = sx_ptr[j][0] * x + sy_ptr[j][0] * y;
			float g = sx_ptr[j][1] * x + sy_ptr[j][1] * y;
			float r = sx_ptr[j][2] * x + sy_ptr[j][2] * y;

			b = b + 128;
			g = g + 128;
			r = r + 128;

			// b = b < 0 ? -b : b;
			// g = g < 0 ? -g : g;
			// r = r < 0 ? -r : r;

			// b = b > 255 ? 255 : b;
			// g = g > 255 ? 255 : g;
			// r = r > 255 ? 255 : r;
            b = b < 0 ? 0 : (b > 255 ? 255 : b);
            g = g < 0 ? 0 : (g > 255 ? 255 : g);
            r = r < 0 ? 0 : (r > 255 ? 255 : r);
			dst_ptr[j] = Vec3b((uchar) b, (uchar) g, (uchar) r);
		}
	}
	return 0;
}

int colourful_face(cv::Mat &src, cv::Mat &dst, std::vector<Rect> &faces) {
	greyscale(src, dst, 0);
	for (const Rect &face: faces) {
		int padding = face.width/8;
		// Rect padded_face(
		// 	max(0, face.x - padding),
		// 	max(0, face.y - padding),
		// 	min(src.cols - face.x + padding, face.width + 2 * padding),
		// 	min(src.rows - face.y + padding, face.height + 2 * padding));
		// src(padded_face).copyTo(dst(padded_face));
		int x1 = max(0, face.x - padding);
        int y1 = max(0, face.y - padding);
        int x2 = min(src.cols, face.x + face.width + padding);
        int y2 = min(src.rows, face.y + face.height + padding);
        
        Rect padded_face(x1, y1, x2 - x1, y2 - y1);
        
        if (padded_face.width > 0 && padded_face.height > 0) {
            src(padded_face).copyTo(dst(padded_face));
        }
	}
	return 0;
}

int adjustments(cv::Mat &src, cv::Mat &dst, int br, int con, bool neg) {
	src.copyTo(dst);

	if (neg) {
		dst = cv::Scalar(255, 255, 255) - dst;
	}

	if (br != 10) {
		float bright = br / 10.0f;
		// dst = src * bright;
		dst.convertTo(dst, -1, bright, 0);
	}

	if (con != 10) {
	float contrast = con / 10.0f;
		dst.convertTo(dst, -1, contrast, 128 * (1.0f - contrast));
	}
	return 0;
}
























