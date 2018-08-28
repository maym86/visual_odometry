
#ifndef VO_MAIN_H
#define VO_MAIN_H

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

const float kDrawScale = 1;

DEFINE_string(data_dir, "/mnt/3b31043d-473f-40dc-bcc7-faebcc6626fb/kitti/dataset/sequences/", "Data directory");
DEFINE_validator(data_dir, &validatePath);

DEFINE_string(res_dir, "/mnt/3b31043d-473f-40dc-bcc7-faebcc6626fb/kitti/dataset/results", "Results directory");
DEFINE_validator(res_dir, &validatePath);

DEFINE_string(poses, "/mnt/3b31043d-473f-40dc-bcc7-faebcc6626fb/kitti/dataset/poses/", "Ground truth poses dir");
DEFINE_validator(poses, &validatePath);

DEFINE_string(seq, "00", "Sequence number");

DEFINE_string(image_dir, "image_0", "Image directory");

DEFINE_string(calib_file, "calib.txt", "Calibration data file");
DEFINE_int32(calib_line_number, 0, "Calibration data line number");

#endif //VO_MAIN_H
