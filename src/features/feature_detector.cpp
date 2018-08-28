#include "feature_detector.h"

FeatureDetector::FeatureDetector(){
    detector_ = cv::FastFeatureDetector::create(30, true, cv::FastFeatureDetector::TYPE_9_16);
}

void FeatureDetector::detect(VOFrame *frame) {
    std::vector<cv::KeyPoint> keypoints;

    detector_->detect(frame->image, keypoints);

    frame->points.clear();

    for(const auto &kp : keypoints){
        frame->points.push_back(kp.pt);
    }
}

