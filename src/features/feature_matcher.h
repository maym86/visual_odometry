
#ifndef VISUALODEMETRY_FEATURES_DETECTION_H
#define VISUALODEMETRY_FEATURES_DETECTION_H

#include <list>
#include <vector>
#include "opencv2/features2d/features2d.hpp"

class FeatureMatcher {

public:

    FeatureMatcher();

    void addFrame(const cv::Mat &image);
    void getMatches(std::vector<cv::DMatch> *matches,
            std::vector<cv::Point2f> *points0,
            std::vector<cv::Point2f> *points1);

    cv::Mat drawMatches();

private:
    int match_count;


    const int kMinFeatures = 5000;
    const float kMatchRatio = 0.8; // As in Lowe's paper; can be tuned

    cv::Ptr<cv::FeatureDetector> detector_;
    cv::Ptr<cv::DescriptorExtractor> extractor_;
    cv::FlannBasedMatcher matcher_;

    std::list<cv::Mat> images_;
    std::list<std::vector<cv::KeyPoint>> keypoints_;
    std::list<cv::Mat> descriptors_;

    std::vector<cv::DMatch> matches_;

};


#endif //VISUALODEMETRY_FEATURES_DETECTION_H
