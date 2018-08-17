
#ifndef VISUALODEMETRY_FEATURES_DETECTION_H
#define VISUALODEMETRY_FEATURES_DETECTION_H

#include <list>
#include <vector>
#include "opencv2/features2d/features2d.hpp"

class FeatureMatcher {

public:

    FeatureMatcher();

    void addFrame(cv::Mat image);
    void getMatches(std::vector<cv::DMatch> *good_matches,
            std::vector<cv::Point2f> *good_points0,
            std::vector<cv::Point2f> *good_points1);

    cv::Mat drawMatches();

private:
    const int kMinDist = 200;

    cv::Ptr<cv::FeatureDetector> detector_;
    cv::Ptr<cv::DescriptorExtractor> extractor_;
    cv::FlannBasedMatcher matcher_;

    std::list<cv::Mat> images_;
    std::list<std::vector<cv::KeyPoint>> keypoints_;
    std::list<cv::Mat> descriptors_;

    std::vector<cv::DMatch> matches_;

};


#endif //VISUALODEMETRY_FEATURES_DETECTION_H
