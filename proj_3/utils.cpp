//
// Created by Ajey K on 21/02/26.
//

#include "utils.h"

long long get_time_instant() {
    cr::system_clock::duration inst = cr::system_clock::now().time_since_epoch();
    cr::seconds sec = cr::duration_cast<cr::seconds>(inst);
    long long ts = sec.count();
    return ts;
}

bool check_file(const fs::path &f, const std::set<std::string> &fmts) {
    bool does_exist = fs::exists(f);
    bool is_regular = fs::is_regular_file(f);
    // bool is_it_empty = fs::is_empty(f);

    std::string ext = f.extension();
    bool ext_is_allowed = fmts.contains(ext);

    return does_exist && is_regular && ext_is_allowed;
}
