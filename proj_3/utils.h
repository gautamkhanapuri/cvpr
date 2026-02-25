//
// Created by Ajey K on 21/02/26.
//

#ifndef UTILS_H
#define UTILS_H

#include <filesystem>
#include <chrono>
#include <string>
#include <set>

namespace fs = std::filesystem;
namespace cr = std::chrono;

long long get_time_instant();

bool check_file(const fs::path& f, const std::set<std::string>& fmts);

#endif //UTILS_H
