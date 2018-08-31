

#ifndef VO_SFM_TRIANGULATION_H
#define VO_SFM_TRIANGULATION_H

#include "src/visual_odometry/vo_frame.h"

#include <opencv2/stitching/detail/matchers.hpp>

#include "pba/src/pba/pba.h"


void triangulateFrame(VOFrame *vo0, VOFrame *vo1);

std::vector<cv::Point3f> triangulate(const std::vector<cv::Point2f> &points0, const std::vector<cv::Point2f> &points1, const cv::Mat &P0, const cv::Mat &P1);

void triangulatePairwiseMatches(const std::vector<std::vector<cv::Point2f>> &keypoints, const std::vector<cv::detail::MatchesInfo> &pairwise_matches, const std::vector<cv::Mat> &poses,
                                                                 std::vector<Point3D> *pba_point_data, std::vector<Point2D> *pba_measurements, std::vector<int> *pba_camidx, std::vector<int> *pba_ptidx);

float getScale(const VOFrame &vo0, const VOFrame &vo1, int min_points, int max_points);

#endif //VO_SFM_TRIANGULATION_H
