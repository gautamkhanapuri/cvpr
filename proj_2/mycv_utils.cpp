//
// Created by Gautam Ajey Khanapuri
// 07 Feb 2026
// Contains all common opencv functionalities.
//

#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <cstdlib>
#include <iostream>
#include <filesystem>

#include "mycv_utils.h"

namespace fs = std::filesystem;

int parse_config(std::string spec_str, FeatureConfig& config) {

    if (spec_str.empty()) {
      std::cout << "Empty specification string" << std::endl;
      std::exit(-1);
    }

    if (spec_str.size() % 2 > 0) {
      std::cout << "Invalid specification string" << std::endl;
      std::cout << "Specification string must be even. It must of this form: part_of_image|type_of_histogram" << std::endl;
      std::exit(-1);
    }

    for (int i = 0; i < spec_str.size(); i += 2) {
      char c1 = spec_str[i];
      char c2 = spec_str[i + 1];
      config.parts.push_back(parse_part_name(c1));
      config.hists.push_back(parse_hist_type(c2));
      // parse_part_name(c1, config.parts);
      // parse_hist_type(c2, config.hists);
    }
    return 0;
}


std::string parse_part_name(char c) {
  switch (c) {
    case 'W':
      return "basic_central_box";

    case 'w':
      return "whole";

    case 't':
      return "top";

    case 'T':
      return "bottom";

    case 'l':
      return "left";

    case 'L':
      return"right";

    case 'a':
      return "q1";

    case 'A':
      return "q2";

    case 'b':
      return"q3";

    case 'B':
      return"q4";

    case 'c':
      return "center";

    case 'C':
      return "edge";

    default:
      std::cout << "Invalid character '" << c << "'" << std::endl;
      std::exit(-1);
  }
}


cv::Rect parse_rect_size(std::string ch, int r, int c) {
  char first_ch = ch[0];
  switch (first_ch) {
    case 'e':  // edge
    case 'w':  // whole
      return cv::Rect(0, 0, c, r);

    case 'W':  // Basic box
      return cv::Rect(((c/2 ) - 3), ((r/2) - 3), basic_box_size, basic_box_size);

    case 't':  // top
      return cv::Rect(0, 0, c, r/2);

    case 'b':  // bottom
      return cv::Rect(0, r/2, c, r/2);

    case 'l':  // left
      return cv::Rect(0, 0, c/2, r);

    case 'r':  // right
      return cv::Rect(c/2, 0, c/2, r);

    case 'c':
      return cv::Rect(c/4, r/4, c/2, r/2);

    case 'q':  // q1, q2, q3, q4
      if (ch[1] == '1') {
        return cv::Rect(c/2, 0, c/2, r/2);
      }
      else if (ch[1] == '2') {
        return cv::Rect(0, 0, c/2, r/2);
      }
      else if (ch[1] == '3') {
        return cv::Rect(0, r/2, c/2, r/2);
      }
      else if (ch[1] == '4') {
        return cv::Rect(c/2, r/2, c/2, r/2);
      }
      else {
        std::cout << "Invalid character '" << ch << "'" << std::endl;
        std::exit(-1);
      }

//    case "edge":
//      pt.push_back("edge");
//    return 0;

    default:
      std::cout << "Invalid character '" << c << "'" << std::endl;
      std::exit(-1);
  }
}


HistogramType parse_hist_type(char c) {
  switch (c) {
    case 'B':
      return HistogramType::BASIC_BOX;

    case 'r':
      return HistogramType::RG_CHROMATICITY;

    case 'R':
      return HistogramType::RGB;

    case 'h':
      return HistogramType::HS;

    case 'u':
      return HistogramType::INTENSITY;

    case 's':
      return HistogramType::SOBEL_MAG_1D;

    case 'S':
      return HistogramType::SOBEL_MAGvORN_2D;

    case 'g':
      return HistogramType::GLCM;

    case 'G':
      return HistogramType::LAW;

     default:
       std::cout << "Invalid character for histogram type. '" << c << "'" << std::endl;
       std::exit(-1);
  }
}


int compute_histogram(const fs::path& im_path, std::vector<float>& vec, HistogramType hist_type, std::string& part) {
  auto it = histogram_functions.find(hist_type);
  int ret = 0;
  if (it == histogram_functions.end()) {
    ret = compute_rg_histogram(im_path, vec, part);  // defaults to rg histogram
  } else {
    ret = it->second(im_path, vec, part);
  }
  return ret;
}


int compute_basic_box_vector(const fs::path &img_path, std::vector<float> &vec, std::string &part, int bins) {
  cv::Mat src = cv::imread(img_path.string());
  if (src.empty()) {
    std::cout << "Could not open or find the image" << img_path  << std::endl;
    return -1;
  }
  cv::Mat img = src(parse_rect_size(part, src.rows, src.cols));  // part = "W"
  vec.clear();
  for (int i=0; i < img.rows; i++) {
    // uchar* ptr = img.ptr<uchar>(i);
    cv::Vec3b* ptr = img.ptr<cv::Vec3b>(i);
    for (int j=0; j<img.cols; j++) {
      float B = ptr[j][0];
      float G = ptr[j][1];
      float R = ptr[j][2];
      //
      // B /= 255.0;
      // G /= 255.0;
      // R /= 255.0;
      vec.push_back(B);
      vec.push_back(G);
      vec.push_back(R);
    }
  }
  return 0;
}


int compute_rg_histogram(const fs::path& img_path, std::vector<float>& vec, std::string& part, int bins) {
  cv::Mat hist;
  hist = cv::Mat::zeros(bins, bins, CV_32FC1);
  cv::Mat src = cv::imread(img_path.string());
  if (src.empty()) {
    std::cout << "Could not open or find the image" << img_path  << std::endl;
    return -1;
  }
  cv::Mat img = src(parse_rect_size(part, src.rows, src.cols));
  for (int i=0; i < img.rows; i++) {
    cv::Vec3b* ptr = img.ptr<cv::Vec3b>(i);
    for (int j=0; j<img.cols; j++) {
      float B = ptr[j][0];
      float G = ptr[j][1];
      float R = ptr[j][2];

      float divisor = B + G + R;
      divisor = divisor > 0.0 ? divisor : 1.0;
      float r = R / divisor;
      float g = G/ divisor;

      int rindex = (int) (r * (bins - 1) + 0.5);
      int gindex = (int) (g * (bins - 1) + 0.5);

      hist.at<float>(rindex, gindex)++;

    }
  }
  vec.clear();
  // std::cout << "Size of vec = " << vec.size() << std::endl;
  hist /= (img.rows * img.cols);
  for (int i=0; i < hist.rows; i++) {
    float* h_ptr = hist.ptr<float>(i);
    for (int j=0; j < hist.cols; j++) {
      vec.push_back(h_ptr[j]);
      // std::cout << "Size of vec IN THE LOOP = " << vec.size() << std::endl;
    }
  }
  return 0;
}


int compute_rgb_histogram(const fs::path& img_path, std::vector<float>& vec, std::string& part, int bins) {
  cv::Mat src = cv::imread(img_path.string());
  if (src.empty()) {
    std::cout << "Could not open or find the image" << img_path  << std::endl;
    return -1;
  }
  cv::Mat img = src(parse_rect_size(part, src.rows, src.cols));
  vec.clear();
  vec.resize(bins * bins * bins, 0.0f);
  for (int i=0; i < img.rows; i++) {
    cv::Vec3b* ptr = img.ptr<cv::Vec3b>(i);
    for (int j=0; j<img.cols; j++) {
      uchar B = ptr[j][0];
      uchar G = ptr[j][1];
      uchar R = ptr[j][2];

      int r_bin = (R * bins) / 256;
      int g_bin = (G * bins) / 256;
      int b_bin = (B * bins) / 256;

      if (r_bin >= bins) r_bin = bins - 1;
      if (g_bin >= bins) g_bin = bins - 1;
      if (b_bin >= bins) b_bin = bins - 1;

      // Flatten 3D to 1D: index = r*(bins²) + g*bins + b
      int index = r_bin * (bins * bins) + g_bin * bins + b_bin;
      vec[index]++;
    }
  }
  float total = static_cast<float>(img.rows * img.cols);
  for (float& val : vec) {
    val /= total;
  }
  return 0;
}


int compute_hs_histogram(const fs::path& img_path, std::vector<float>& vec, std::string& part, int bins) {

  cv::Mat s = cv::imread(img_path.string());
  if (s.empty()) {
    std::cout << "Could not open or find the image" << img_path << std::endl;
    return -1;
  }
  cv::Mat src;
  cv::cvtColor(s, src, cv::COLOR_BGR2HSV);
  vec.clear();
  vec.resize(bins * bins, 0.0f);
//  int skipped = 0;
  cv::Mat img = src(parse_rect_size(part, src.rows, src.cols));
  for (int i=0; i < img.rows; i++) {
    cv::Vec3b* ptr = img.ptr<cv::Vec3b>(i);
    for (int j=0; j<img.cols; j++) {
      float H = ptr[j][0];
      float S = ptr[j][1];
//      float V = ptr[j][2];
//
//      if (V < 20) {
//        skipped++;
//        continue;
//      }

      // Map to bins
      int h_bin = (H * bins) / 180;  // Hue is 0-179
      int s_bin = (S * bins) / 256;

      // Clamp
      if (h_bin >= bins) h_bin = bins - 1;
      if (s_bin >= bins) s_bin = bins - 1;

      int index = h_bin * bins + s_bin;
      vec[index]++;
    }
  }
//  float div = img.rows * img.cols - skipped;
  float total = static_cast<float>(img.rows * img.cols);
  for (float& val : vec) {
    val /= total;
  }
  return 0;
}


int compute_intensity_histogram(const fs::path& img_path, std::vector<float>& vec, std::string& part, int bins) {

  cv::Mat src = cv::imread(img_path.string(), cv::IMREAD_GRAYSCALE);
  if (src.empty()) {
    std::cout << "Could not open or find the image" << img_path  << std::endl;
    return -1;
  }
  vec.clear();
  vec.resize(bins, 0.0f);
//  int skipped = 0;
  cv::Mat img = src(parse_rect_size(part, src.rows, src.cols));
  for (int i=0; i < img.rows; i++) {
    const uchar *ptr = img.ptr<uchar>(i);

    for (int j = 0; j < img.cols; j++) {
      uchar intensity = ptr[j];

      // Map to bin
      int bin_index = (intensity * bins) / 256;
      if (bin_index >= bins) bin_index = bins - 1;

      vec[bin_index]++;
    }
  }
  float total = static_cast<float>(img.rows * img.cols);
  for (float& val : vec) {
    val /= total;
  }
  return 0;
}


int sobelX3x3(cv::Mat &src, cv::Mat &dst) {
//    dst.create(src.rows, src.cols, CV_16SC3);
	// src -> 8UC1 -- greyscale
	// dst -> 16SC1 -- single signed short channel
    int r = src.rows;
    int c = src.cols;
    for (int i=0;i<src.rows;i++) {
        short *dst_ptr = dst.ptr<short>(i);
        uchar *top = nullptr;
        uchar *bot = nullptr;
        uchar *mid = src.ptr<uchar>(i);
        if (i == 0) {
            top = src.ptr<uchar>(i);
            bot = src.ptr<uchar>(i+1);
        } else if (i == r - 1) {
            top = src.ptr<uchar>(i-1);
            bot = src.ptr<uchar>(i);
        } else {
            top = src.ptr<uchar>(i-1);
            bot = src.ptr<uchar>(i+1);
        }
        uchar* rows[3] = {top, mid, bot};
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

            short vertical_result[3];

            int column_indexes[3] = {left, middle, right};
            for (int k=0; k<=2; k++) {
                int column = column_indexes[k];
                float b_t = 0;
//                float g_t = 0;
//                float r_t = 0;
                for (int x=0; x<=2; x++) {
                    b_t += rows[x][column] * SOBEL_g[x];
//                    g_t += rows[x][column][1] * SOBEL_g[x];
//                    r_t += rows[x][column][2] * SOBEL_g[x];
                }
                signed short b = (signed short) (b_t/4);
//                signed short g = (signed short) (g_t/4);
//                signed short r = (signed short) (r_t/4);

//                cv::Vec3s tmp(b, g, r);
                short tmp = b;
                vertical_result[k] = tmp;
            }

            signed short b_final = 0;
//            signed short g_final = 0;
//            signed short r_final = 0;
            for (int a=0; a<=2; a++) {
                b_final += vertical_result[a] * SOBEL_h[a];
//                g_final += vertical_result[a][1] * SOBEL_h[a];
//                r_final += vertical_result[a][2] * SOBEL_h[a];
            }

            short return_pixel = b_final;
            dst_ptr[j] = return_pixel;
        }
    }
    return 0;
}


int sobelY3x3(cv::Mat &src, cv::Mat &dst) {
//    dst.create(src.rows, src.cols, CV_16SC3);
	// src -> 8UC1 -- greyscale
	// dst -> 16SC1 -- single signed short channel
    int r = src.rows;
    int c = src.cols;
    for (int i=0;i<src.rows;i++) {
        short *dst_ptr = dst.ptr<short>(i);
        uchar *top = nullptr;
        uchar *bot = nullptr;
        uchar *mid = src.ptr<uchar>(i);
        if (i == 0) {
            top = src.ptr<uchar>(i);
            bot = src.ptr<uchar>(i+1);
        } else if (i == r - 1) {
            top = src.ptr<uchar>(i-1);
            bot = src.ptr<uchar>(i);
        } else {
            top = src.ptr<uchar>(i-1);
            bot = src.ptr<uchar>(i+1);
        }
        for (int j=0;j<src.cols;j++) {
            // short pixel = apply_sobelY(src, i, j);
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

            short vertical_result[3];
            uchar* rows[3] = {top, mid, bot};

            int column_indexes[3] = {left, middle, right};
            for (int k=0; k<=2; k++) {
                int column = column_indexes[k];
                int b_t = 0;
//                int g_t = 0;
//                int r_t = 0;
                for (int x=0; x<=2; x++) {
                    b_t += rows[x][column] * SOBEL_r[x];
//                    g_t += rows[x][column][1] * SOBEL_r[x];
//                    r_t += rows[x][column][2] * SOBEL_r[x];
                }
                // signed short b = (signed short) (b_t/4);
                // signed short g = (signed short) (g_t/4);
                // signed short r = (signed short) (r_t/4);

                // uchar tmp(b, g, r);
                short tmp = b_t;
                vertical_result[k] = tmp;
            }

            float b_f = 0.0;
//            float g_f = 0.0;
//            float r_f = 0.0;
            for (int a=0; a<=2; a++) {
                b_f += vertical_result[a] * SOBEL_g[a];
//                g_f += vertical_result[a][1] * SOBEL_g[a];
//                r_f += vertical_result[a][2] * SOBEL_g[a];
            }

            signed short b_final = (signed short) (b_f/4);
//            signed short g_final = (signed short) (g_f/4);
//            signed short r_final = (signed short) (r_f/4);

//            short return_pixel(b_final, g_final, r_final);
            short return_pixel = b_final;
            dst_ptr[j] = return_pixel;
        }
    }
    return 0;
}


int magnitude(cv::Mat &sx, cv::Mat &sy, cv::Mat &dst) {
    int r = sx.rows;
    int c = sx.cols;
//    dst.create(sx.rows, sx.cols, CV_8UC3);

    // sx - 16SC1
    // sy - 16SC1
    // dst - 8UC1
    for (int i=0; i<r; i++) {
    short *x_ptr = sx.ptr<short>(i);
    short *y_ptr = sy.ptr<short>(i);
    uchar *dst_ptr = dst.ptr<uchar>(i);
        for (int j=0; j<c; j++) {
            short x_val = x_ptr[j];
            short y_val = y_ptr[j];

            float b_t = std::sqrt((x_val * x_val) + (y_val * y_val));
//            float g_t = std::sqrt((x_val[1] * x_val[1]) + (y_val[1] * y_val[1]));
//            float r_t = std::sqrt((x_val[2] * x_val[2]) + (y_val[2] * y_val[2]));

            uchar b = (uchar) (b_t<256 ? b_t : 255);
//            uchar g = (uchar) (g_t<256 ? g_t : 255);
//            uchar rc = (uchar) (r_t<256 ? r_t : 255);

//            dst_ptr[j] = Vec3b(b, g, rc);
            dst_ptr[j] = b;
        }
    }
    return 0;
}


int compute_sobel_mag_1d_histogram(const fs::path& img_path, std::vector<float>& vec, std::string& part, int bins) {
  vec.clear();
  vec.resize(bins, 0.0f);
  cv::Mat src = cv::imread(img_path.string(), cv::IMREAD_GRAYSCALE);  // 8UC1
  if (src.empty()) {
    std::cout << "Could not open or find the image" << img_path  << std::endl;
    return -1;
  }
  cv::Mat region = src(parse_rect_size(part, src.rows, src.cols));  // 8UC1

  cv::Mat sx(cv::Size(region.cols, region.rows), CV_16SC1);
  cv::Mat sy(cv::Size(region.cols, region.rows), CV_16SC1);
  cv::Mat img(cv::Size(region.cols, region.rows), CV_8UC1);
  sobelX3x3(region, sx);
  sobelY3x3(region, sy);
  magnitude(sx, sy, img);

  for (int i=0; i < img.rows; i++) {
    uchar* ptr = img.ptr<uchar>(i);
    for (int j=0; j<img.cols; j++) {
      float B = ptr[j];
//      float G = ptr[j][1];
//      float R = ptr[j][2];

//      float divisor = B + G + R;
//      divisor = divisor > 0.0 ? divisor : 1.0;
//      float r = R / divisor;
//      float g = G/ divisor;
//
//      int rindex = (int) (r * (bins - 1) + 0.5);
//      int gindex = (int) (g * (bins - 1) + 0.5);
//
//      hist.at<float>(rindex, gindex)++;
	  	int bin_index = (int) (B * bins / 256);
        vec[bin_index]++;
    }
  }
  float total = static_cast<float>(img.rows * img.cols);
  for (float& val : vec) {
    val /= total;
  }
  return 0;
}


int compute_sobel_mag_vs_orn_2d_histogram(const fs::path& img_path, std::vector<float>& vec, std::string& part, int bins) {
  cv::Mat hist;
  hist = cv::Mat::zeros(bins, angle_bins, CV_32FC1);
  vec.clear();
  cv::Mat src = cv::imread(img_path.string(), cv::IMREAD_GRAYSCALE);  // 8UC1
  if (src.empty()) {
    std::cout << "Could not open or find the image" << img_path  << std::endl;
    return -1;
  }
  cv::Mat region = src(parse_rect_size(part, src.rows, src.cols));  // 8UC1

  cv::Mat sx(cv::Size(region.cols, region.rows), CV_16SC1);
  cv::Mat sy(cv::Size(region.cols, region.rows), CV_16SC1);
  cv::Mat img(cv::Size(region.cols, region.rows), CV_8UC1);
  sobelX3x3(region, sx);
  sobelY3x3(region, sy);
  magnitude(sx, sy, img);

  for (int i=0; i < img.rows; i++) {
    short* x_ptr = sx.ptr<short>(i);
    short* y_ptr = sy.ptr<short>(i);
    uchar* ptr = img.ptr<uchar>(i);
    for (int j=0; j<img.cols; j++) {
      float B = ptr[j];
      float x_val = x_ptr[j];
      float y_val = y_ptr[j];
//      float G = ptr[j][1];
//      float R = ptr[j][2];

//      float divisor = B + G + R;
//      divisor = divisor > 0.0 ? divisor : 1.0;
//      float r = R / divisor;
//      float g = G/ divisor;

      float angle_rad = std::atan2(y_val, x_val);
      float angle_deg = (angle_rad * 180.0 / CV_PI);
      if (angle_deg < 0) {
        angle_deg += 180;
      }

      int angle_bin = (int) (angle_deg * angle_bins / 180.0);
	  int mag_bin = (int) (B * bins / 256);
//      int rindex = (int) (r * (bins - 1) + 0.5);
//      int gindex = (int) (g * (bins - 1) + 0.5);

      hist.at<float>(mag_bin, angle_bin)++;
    }
  }
  vec.clear();
  hist /= (img.rows * img.cols);
  for (int i=0; i < hist.rows; i++) {
    float* h_ptr = hist.ptr<float>(i);
    for (int j=0; j < hist.cols; j++) {
      vec.push_back(h_ptr[j]);
    }
  }
  return 0;
}


int quantize_img(cv::Mat& src, cv::Mat& dst, int levels) {
  for (int i=0; i < src.rows; i++) {
    uchar* ptr = src.ptr<uchar>(i);
    uchar* dst_ptr = dst.ptr<uchar>(i);
    for (int j=0; j < src.cols; j++) {
      float B = ptr[j];
      int new_val = ((int) (B * levels / 256));
      if (new_val >= levels) {
        new_val = levels - 1;
      }
      dst_ptr[j] = (uchar) new_val;
    }
  }
  return 0;
}


int compute_cooccurrence_matrix(cv::Mat&src, cv::Mat& dst, int dx, int dy) {
  for (int i=0; i < (src.rows - dy); i++) {
    uchar* sr1 = src.ptr<uchar>(i);
    uchar* sr2 = src.ptr<uchar>(i + dy);
//    uchar* dst_ptr = dst.ptr<uchar>(i);
    for (int j=0; j < (src.cols - dx); j++) {
      uchar val1 = sr1[j];
      uchar val2 = sr2[j + dx];

      dst.at<float>(val1, val2)++;
      dst.at<float>(val2, val1)++;
    }
  }

  dst = dst / (quantize_levels * quantize_levels);
  return 0;
}


int calculate_glcm_vectors(cv::Mat& co_mat, float& energy, float& contrast, float& homogeneity, float& entropy, float& max_prob) {
  for (int i=0; i < co_mat.rows; i++) {
    float* ptr = co_mat.ptr<float>(i);
    for (int j=0; j < co_mat.cols; j++) {
      float p = ptr[j];
      if (p > 0) {
        energy += p * p;

        int diff = i - j;
        contrast += diff * diff * p;

        homogeneity += p / (1.0 + std::abs(diff));

        entropy += -p * std::log2(p);

        if (p > max_prob) {
          max_prob = p;
        }
      }
    }
  }
  return 0;
}


int compute_glcm_features(const fs::path& img_path, std::vector<float>& vec, std::string& part, int bins) {
  vec.clear();

  cv::Mat src = cv::imread(img_path.string(), cv::IMREAD_GRAYSCALE);  // 8UC1
  if (src.empty()) {
    std::cout << "Could not open or find the image" << img_path  << std::endl;
    return -1;
  }
  cv::Mat region = src(parse_rect_size(part, src.rows, src.cols));

  quantize_img(region, region, bins);

  for (std::pair<int, int> offset: offsets) {
    cv::Mat co_mat = cv::Mat::zeros(cv::Size(bins, bins), CV_32FC1);
    float energy = 0.0f;
    float contrast = 0.0f;
    float homogeneity = 0.0f;
    float entropy = 0.0f;
    float max_prob = 0.0f;
    compute_cooccurrence_matrix(region, co_mat, offset.first, offset.second);
    calculate_glcm_vectors(co_mat, energy, contrast, homogeneity, entropy, max_prob);
    vec.push_back(energy);
    vec.push_back(contrast);
    vec.push_back(homogeneity);
    vec.push_back(entropy);
    vec.push_back(max_prob);
  }
  return 0;
}


int create_laws_filter(const std::array<float, 5>& a, const std::array<float, 5>& b, cv::Mat& filter) {
	for (int i=0; i < 5; i++) {
          uchar* ptr = filter.ptr<uchar>(i);
          for (int j=0; j < 5; j++) {
            ptr[j] = a[i] * b[j];
          }
	}
    return 0;
}


int calculate_law_response_histogram(cv::Mat& src, std::vector<float>& hist, int bins) {
  hist.clear();
  hist.resize(bins, 0.0f);

  double min_val;
  double max_val;
  minMaxLoc(src, &min_val, &max_val);
  if (max_val - min_val < 0.001) {
    hist[0] = 1.0f;
    return 0;
  }
  for (int i=0; i < bins; i++) {
    float* ptr = src.ptr<float>(i);
    for (int j=0; j < bins; j++) {
      float val = ptr[j];

      float norm = (val - min_val) / (max_val - min_val);
      int bin_index = (int) (norm * bins);

      if (bin_index >= bins) {
      	bin_index = bins - 1;
      }
      hist[bin_index]++;
    }
  }
  float total = static_cast<float>(src.rows * src.cols);
  for (float& val : hist) {
    val /= total;
  }
  return 0;
}


int compute_law_histogram(const fs::path& img_path, std::vector<float>& vec, std::string& part, int bins) {
  cv::Mat src = cv::imread(img_path.string(), cv::IMREAD_GRAYSCALE);  // 8UC1
  if (src.empty()) {
    std::cout << "Could not open or find the image" << img_path  << std::endl;
    return -1;
  }
  cv::Mat region = src(parse_rect_size(part, src.rows, src.cols));

  for (std::pair<std::array<float, 5>, std::array<float, 5>> lf: laws_filters) {
    cv::Mat two_d_filter(cv::Size(5, 5), CV_32FC1);
    create_laws_filter(lf.first, lf.second, two_d_filter);

    cv::Mat response;
    cv::filter2D(region, response, -1, two_d_filter);
    response = cv::abs(response);

    std::vector<float> tmp;
    calculate_law_response_histogram(response, tmp, bins);
    vec.insert(vec.end(), tmp.begin(), tmp.end());
  }
  return 0;
}