//
// Gautam Ajey Khanapuri
// 03 February 2026
// Common utility functions that are required for both programs. Header file lists them out.
//

#ifndef UTILS_H
#define UTILS_H


#include <chrono>
#include <filesystem>
#include <set>


namespace cr = std::chrono;
namespace fs = std::filesystem;


long long get_time_instant();
bool check_file(const fs::path& f, const std::set<std::string>& fmts);


#endif //UTILS_H
