//
// Created by Ajey K on 21/02/26.
// Header file for utility functions.
// Provides helper functions for file validation and timestamp generation.
//

#ifndef UTILS_H
#define UTILS_H

#include <filesystem>
#include <chrono>
#include <string>
#include <set>

namespace fs = std::filesystem;
namespace cr = std::chrono;

/**
 * Gets current timestamp in milliseconds since epoch.
 * Used for generating unique filenames.
 * @return timestamp as long long integer
 */
long long get_time_instant();

/**
 * Checks if file exists and has allowed extension.
 * @param f file path to check
 * @param fmts set of allowed file extensions (e.g., {".csv", ".txt"})
 * @return true if file exists and extension is allowed
 */
bool check_file(const fs::path &f, const std::set<std::string> &fmts);

#endif //UTILS_H
