#include "feature_detector.h"

FeatureDetector::FeatureDetector(){
    detector_ = cv::FastFeatureDetector::create(20, true, cv::FastFeatureDetector::TYPE_9_16);
    descriptor_ = cv::ORB::create(1000);
}

void FeatureDetector::detectFAST(VOFrame *frame) {
    std::vector<cv::KeyPoint> keypoints;

    detector_->detect(frame->image, keypoints);

    frame->points.clear();

    for(const auto &kp : keypoints){
        frame->points.push_back(kp.pt);
    }

}

void FeatureDetector::detectComputeORB(const VOFrame &frame, std::vector<cv::KeyPoint> *keypoints, cv::Mat *descriptors){
    descriptor_->detectAndCompute(frame.image, cv::noArray(), *keypoints, *descriptors);
}

