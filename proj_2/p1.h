/**
 * Created by Gautam Ajey Khanapuri
 * 07 Feb 2026
 * Header file for Program Part 1. Abbreviated to p1. It declares all the functions that will be used for generating feature
 * vectors of the database of images.
 *
 *
 * USAGE EXAMPLES FOR P1 (Program Part 1)
 *
 * P1 pre-computes feature vectors for all images in a directory and writes them to CSV files.
 * These CSV files are then used by P2 for fast image matching.
 *
 * MODE 1: BASELINE (-b-)
 * Extracts 7x7 center square from each image as baseline feature.
 * Simple but effective for centered objects.
 *
 * Format: ./p1 -b- <image_directory>
 *
 * Example:
 *   ./p1 -b- ~/datasets/olympus
 *
 * Output:
 *   Creates: baseline_ft_vec_<timestamp>.csv
 *   Format: image_path,0.0
 *   Vector size: 1
 *   For Baseline mode, generation and comparison is done by P2.
 *   P1 only takes in directory and writes it to a csv file. This file is passed to P2.
 *
 * MODE 2: MULTIPLE HISTOGRAMS (-m-)
 * Extracts custom histogram features from specified image regions.
 * Creates separate CSV file for each part-histogram combination.
 *
 * Format: ./p1 -m-<spec> <image_directory>
 *
 * Spec format (groups of 2 characters):
 *   [part][histogram][part][histogram]...
 *
 * Part codes:
 *   w=whole image, t=top half, T=bottom half, l=left half, L=right half,
 *   a/A/b/B=quadrants (counter-clockwise), c=center region, C=edge region
 *
 * Histogram codes:
 *   B=baseline 7x7, r=RG chromaticity, R=RGB color, h=Hue-Saturation,
 *   u=intensity (grayscale), s=sobel magnitude (1D), S=sobel magnitude vs orientation (2D),
 *   g=GLCM texture features, G=Laws filter responses
 *
 * Example 1 - Single whole image RG histogram:
 *   ./p1 -m-wr ~/datasets/olympus
 *
 *   Output file: whole_rg_multi_histogram_ft_vec_<timestamp>.csv
 *   Vector size: 1024 floats (32x32 bins)
 *
 * Example 2 - Top and bottom RG histograms:
 *   ./p1 -m-trTr ~/datasets/olympus
 *
 *   Output files:
 *     top_rg_multi_histogram_ft_vec_<timestamp>.csv (1024 floats)
 *     bottom_rg_multi_histogram_ft_vec_<timestamp>.csv (1024 floats)
 *
 *   Use case: Sunset matching (blue sky top, warm colors bottom)
 *
 * FEATURE VECTOR SIZES
 *
 * Default bin counts produce these vector sizes:
 *   - Baseline (B): 49 floats (7x7 square)
 *   - RG chromaticity (r): 1024 floats (32x32 bins)
 *   - RGB (R): 32768 floats (32x32x32 bins, flattened)
 *   - Hue-Saturation (h): 1024 floats (32x32 bins)
 *   - Intensity (u): 32 floats (32 bins)
 *   - Sobel magnitude 1D (s): 32 floats (32 bins)
 *   - Sobel magnitude vs orientation 2D (S): 288 floats (32 mag bins x 9 angle bins)
 *   - GLCM (g): 20 floats (5 features x 4 offsets)
 *   - Laws (G): 144 floats (9 filters x 16 bins per filter)
 *
 * OUTPUT FILES
 *
 * P1 creates CSV files with this naming pattern:
 *   <part>_<histogram>_multi_histogram_ft_vec_<timestamp>.csv
 *
 * CSV format:
 *   image_path,feature_1,feature_2,...,feature_n
 *   /path/to/image1.jpg,0.123,0.456,...
 *   /path/to/image2.jpg,0.234,0.567,...
 *
 * Each CSV contains features for ALL images in the directory.
 *
 * WORKFLOW
 *
 * Step 1: Run P1 to generate feature vectors
 *   ./p1 -m-trTr ~/datasets/olympus
 *
 *   Output: top_rg_*.csv and bottom_rg_*.csv
 *
 * Step 2: Run P2 to find matches
 *   ./p2 query.jpg -m-trI3TrI1 top_rg_*.csv bottom_rg_*.csv
 *
 * Step 3: View results (interactive)
 *   Program displays top 5, prompts for more
 *
 * REMEMBER TO
 *
 * 1. Spec length must be even (groups of 2)
 *    Wrong: ./p1 -m-wra  (3 characters)
 *    Right: ./p1 -m-wr   (2 characters)
 *
 * 2. Directory must exist and contain images
 *    Check: ls ~/datasets/olympus/example.jpg
 *
 * 3. Allowed image formats: jpg, jpeg, png, webp, tiff
 *    Other formats will be skipped
 *
 * 4. Remember CSV filenames for P2
 *    P1 outputs full paths - copy these for P2 usage
 */

#ifndef P1_H
#define P1_H


#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>

#include <iostream>
#include <string>
#include <cstdlib>
#include <filesystem>

#include "utils.h"
#include "csv_util.h"
#include "mycv_utils.h"


namespace fs = std::filesystem;

// COnstants used throughout the part 1 of the program are defined here.
const std::set<char> valid_modes = {'b', 'h', 'm'};
inline const std::set<std::string> &allowed_img_formats = {".jpg", ".jpeg", ".jpe", ".png", ".webp", ".tiff", ".tif"};
const std::string op_file_format = ".csv";
const std::string bsm_op_file_name = "baseline_ft_vec_";
const std::string bhs_op_file_name = "histogram_rg_ft_vec_";
const std::string mhs_op_file_name = "multi_histogram_ft_vec_";


// Class P1 is responsible for parsing the command line inputs and deletaging the task of generating vectors to appropriate methods.
class P1 {
  private:
    // Instance variables of P1. They are used across different functions in this class.
    std::string mode;
    std::string spec;
    fs::path dir;
    std::vector<fs::path> img_paths;
    std::vector<fs::path> op_files;

  public:

    /**
     * Parses the char* recevied from the command line and decides which mode needs to be used. Performs simple char matching.
     * Correct format '-[d/b/m]-...'
     *
     * @param arg The string recevied on the command line is passed to the function by reference.
     * @return 0 in case of no errors. exits otherwise
     */
    int parse_mode(std::string& arg);

    /**
     * Parses the third argument on the command line. The third argument must be the path to directory that contains
     * the images we want to comapre against the target.
     * Escape white spaces using back-slash.
     *
     * @param arg The 3 argument from the command line is passed by reference.
     * @return  0 in case of no errors. exits otherwise
     */
    int parse_dir(std::string& arg);

    /**
     * Iterates over the files in the directory parsed in {@code parse_mode} function and identifies all the images.
     * Performs the some sanity checks on each file and stores their absolute paths in the class.
     *
     * @return 0 on no errors. exits otherwise
     */
    int find_img_paths();

    /**
     * This is a dispatcher function. Based on the mode parsed, the appropriate method is called with all the necessary
     * arguments.
     * After the functions run, it prints out the absolute path of the files that contain the feature vectors.
     *
     * @return 0 on no errors. Exits otherwise.
     */
    int run();

    /**
     * It is called in case the selected mode is basic. For this particular task, the images in the given directory are identified
     * and their paths written to a csv file.
     * No vectors for the central 7x7 square is generated. 7x7 scentral squares for both the target and the images are generated
     * in p2 itself. I did this to reduce the number of file reads and writes.
     *
     * @return 0 on success. exits otherwise.
     */
    int run_bsm();
    // int run_bhs();

    /**
     * This function serves the purpose of both task 2 and 3. Using the correct flags will enable you to do both using this
     * same function. This function generates vectors and write them to csv files. In case you wish to generate vectors for
     * different portions of the image, you can specify the part along with the histogram type that you want to generate.
     * Each spatial portions vectors are written to a separate file.
     * These files are printed to the terminal.
     * While providing these files to Part2, they must be provided in the same order to as the flags (character identfiers) appear.
     * Otherwise, erroneous results may ensue.
     *
     * @return 0 on success. exits othersie
     */
    int run_mhs();
};


#endif //P1_H
