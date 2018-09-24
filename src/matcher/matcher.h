//
// Created by maym86 on 9/23/18.
//

#ifndef VO_MATCHER_H
#define VO_MATCHER_H

#include <vector>
#include <opencv2/stitching/detail/matchers.hpp>

std::vector<cv::detail::MatchesInfo>  matcher(const std::vector<cv::detail::ImageFeatures> &features, const cv::Mat &K);

std::vector<std::vector<int>> createMatchMatrix(const std::vector<cv::detail::MatchesInfo> pairwise_matches, const int pose_count);

std::vector<cv::detail::MatchesInfo> matcher(const std::vector<cv::Mat> &images, const std::vector<cv::detail::ImageFeatures> &features, const cv::Mat &K);

#endif //VO_MATCHER_H
