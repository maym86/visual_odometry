

#ifndef VO_SFM_TRIANGULATION_H
#define VO_SFM_TRIANGULATION_H

#include "src/visual_odometry/vo_frame.h"

#include <opencv2/stitching/detail/matchers.hpp>

void triangulateFrame(const cv::Point2f &pp, const double focal, const VOFrame &frame0, VOFrame *frame1);

std::vector<cv::Point3d> points4dToVec(const cv::Mat &points4d);

std::vector<cv::Point3d> triangulate(const cv::Point2f &pp, const double focal, const std::vector<cv::Point2f> &points0, const std::vector<cv::Point2f> &points1, const cv::Mat &P0, const cv::Mat &P1);

float getScale(const VOFrame &vo0, const VOFrame &vo1, int min_points, int max_points);

#endif //VO_SFM_TRIANGULATION_H
