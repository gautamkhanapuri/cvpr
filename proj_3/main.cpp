//
// Created by Ajey K on 21/02/26.
// Main entry point for real-time object recognition system.
// Validates command line arguments and initializes the recognition system.
//

#include "iostream"
#include "string"
#include "utils.h"
#include "rtor.h"

/**
 * Program entry point.
 * Validates database file paths and starts real-time object recognition system.
 *
 * @param argc argument count (expects 3)
 * @param argv arguments: [program_name] [classic_features_db] [dnn_features_db]
 * @return 0 on successful exit
 */
int main(int argc, char *argv[]) {
    std::cout << "Starting up main func..." << std::endl;
    if (argc != 3) {
        std::cout << "Error! Wrong number of arguments!" << std::endl;
        std::cout << "Usage: ./rtor <classic_features_db_filename> <dnn_features_db_filename>" << std::endl;
        std::cout << "filename = path to the vector database file. Expected format = CSV" << std::endl;
        std::cout << "Make sure whitespaces are escaped with backslash \\ " << std::endl;
        std::exit(-1);
    }

    const std::set<std::string> csv_file_type = {".csv"};
    bool db_file_exists = check_file(argv[1], csv_file_type);
    if (!db_file_exists) {
        std::cout << "Error! Database file " << argv[1] << " does not exist." << std::endl;
        std::cout << "Even if empty, the file must be provided." << std::endl;
        std::exit(-1);
    }
    std::cout << "Database file " << argv[1] << " found!" << std::endl;

    bool resent_db_file_exists = check_file(argv[2], csv_file_type);
    if (!resent_db_file_exists) {
        std::cout << "Error! resnet Database file " << argv[2] << " does not exist." << std::endl;
        std::cout << "Even if empty, the file must be provided." << std::endl;
        std::exit(-1);
    }
    std::cout << "resnet database file " << argv[2] << " found!" << std::endl;

    RTObectRecognizer rt_object_recognizer(argv[1], argv[2]);
    rt_object_recognizer.run();
    return 0;
}
