//
// Created by Gautam Khanapuri on 10th March 2026
// Main entry point for camera calibration and augmented reality system.
// Validates command line arguments and initializes the CAAR orchestrator.
//

#include <filesystem>
#include <string>
#include <iostream>
#include <fstream>

#include "caar.h"
#include "utils.h"

/**
 * Program entry point.
 * Validates calibration file path and starts calibration/AR system.
 *
 * @param argc argument count (expects 2)
 * @param argv arguments: [program_name] [calibration_file.csv]
 * @return 0 on successful exit
 */
int main(int argc, char* argv[]) {
    std::cout << "Starting up CALIBRATION and AR..." << std::endl;
    if (argc != 2) {
        std::cout << "ERROR!" << std::endl;
        std::cout << "Expected usage: ./caar <calibration_data_file>" << std::endl;
        std::exit(-1);
    }
    std::string calib_file = argv[1];
    if (!check_file(calib_file, calibration_file_formats)) {
        std::cout << "ERROR!" << std::endl;
        std::cout << "Calibration file does not exist" << std::endl;
        std::cout << "Please create a file with an extension '.csv' (EMPTY file is also acceptable)" << std::endl;
        std::exit(-1);
    }
    std::cout << "Calibration data file found: " << calib_file << std::endl;

    fs::path calibration_file(calib_file);
    CAAR caar(calibration_file);
    caar.run();

    std::cout << "Terminating CALIBRATION and AR..." << std::endl;
    return 0;
}