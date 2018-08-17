
#ifndef VISUAL_ODEMETRY_MAIN_H
#define VISUAL_ODEMETRY_MAIN_H

#include <stdio.h>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <boost/filesystem/operations.hpp>


static bool validatePath(const char *flagname, const std::string &value) {
    if (value.empty()) {
        printf("Invalid value for --%s: %s\n", flagname, value.c_str());
        return false;
    }

    if (!boost::filesystem::exists(value)) {
        printf("Path not found for --%s: %s\n", flagname, value.c_str());
        return false;
    }

    return true;
}

const float kScale = 1;

DEFINE_string(dir, "", "Data directory");
DEFINE_validator(dir, &validatePath);

#endif //VISUAL_ODEMETRY_MAIN_H
