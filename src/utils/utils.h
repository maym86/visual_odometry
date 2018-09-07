

#ifndef VO_UTILS_H
#define VO_UTILS_H

#include <cv.hpp>

cv::Mat eulerAnglesToRotationMatrix(const cv::Mat &theta);

bool isRotationMatrix(cv::Mat &R);

cv::Mat rotationMatrixToEulerAngles(cv::Mat &R);

#endif //VO_UTILS_H
