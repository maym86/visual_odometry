#ifndef VO_KITTI_H
#define VO_KITTI_H

#include <cxcore.h>

#include "evaluate_odometry.h"

cv::Mat loadKittiCalibration(std::string calib_file, int line_number);

Matrix kittiResultMat(cv::Mat pose);

void saveResults(const std::string &gt_poses_path, const std::string &res_dir, const std::string &seq, std::vector<Matrix> &result_poses);

#endif //VO_KITTI_H
