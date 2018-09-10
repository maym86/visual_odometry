

#ifndef VO_SFM_TRIANGULATION_H
#define VO_SFM_TRIANGULATION_H

#include "src/visual_odometry/vo_frame.h"

#include <opencv2/stitching/detail/matchers.hpp>

std::vector<cv::Point3d> points3dToVec(const cv::Mat &points3d);

std::vector<cv::Point3d> points4dToVec(const cv::Mat &points4d);

std::vector<cv::Point3d>
triangulate(const cv::Point2f &pp, const cv::Point2f &focal, const std::vector<cv::Point2f> &points0,
            const std::vector<cv::Point2f> &points1, const cv::Mat &P0, const cv::Mat &P1);

float getScale(const VOFrame &frame0, const VOFrame &frame1, size_t min_points, size_t max_points, float max_3d_dist);

#endif //VO_SFM_TRIANGULATION_H
