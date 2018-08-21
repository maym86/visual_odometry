#include "feature_detector.h"


FeatureDetector::FeatureDetector(){
    detector_ = cv::ORB::create(kMinFeatures);
}

std::vector<cv::Point2f> FeatureDetector::detect(const cv::Mat &image) {
    std::vector<cv::KeyPoint> keypoints;
    detector_->detect(image, keypoints);

    std::vector<cv::Point2f> results;

    for(const auto &kp : keypoints){
        results.push_back(kp.pt);
    }
    return results;
}
