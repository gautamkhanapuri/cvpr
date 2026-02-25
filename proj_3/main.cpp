//
// Created by Ajey K on 21/02/26.
//


#include "iostream"
#include "string"
#include "utils.h"
#include "rtor.h"


int main(int argc, char* argv[]) {
    std::cout<< "Starting up main func..." <<std::endl;
    if (argc != 2) {
        std::cout << "Error! Wrong number of arguments!" << std::endl;
        std::cout << "Usage: ./rtor <filename>" << std::endl;
        std::cout << "filename = path to the vector database file. Expected format = CSV" << std::endl;
        std::cout << "Make sure whitespaces are escaped with backslash \\ " << std::endl;
        std::exit(-1);
    }


    const std::set<std::string> csv_file_type = {".csv"};
    bool db_file_exists = check_file(argv[1], csv_file_type);
    if (!db_file_exists) {
        std::cout << "Error! Database file " << argv[1] << " does not exist." << std::endl;
        std::cout << "Even if empty, the file must exist." << std::endl;
        std::exit(-1);
    }
    std::cout << "Database file " << argv[1] << " found!" << std::endl;
    RTObectRecognizer rt_object_recognizer(argv[1]);
    rt_object_recognizer.run();
    return 0;
}