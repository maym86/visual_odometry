#include "feature_detector.h"

FeatureDetector::FeatureDetector(){
    detector_ = cv::FastFeatureDetector::create(20, true, cv::FastFeatureDetector::TYPE_9_16);
    descriptor_ = cv::xfeatures2d::BriefDescriptorExtractor::create(64);
}

void FeatureDetector::detect(VOFrame *frame) {
    std::vector<cv::KeyPoint> keypoints;

    detector_->detect(frame->image, keypoints);

    frame->points.clear();

    for(const auto &kp : keypoints){
        frame->points.push_back(kp.pt);
    }

}

void FeatureDetector::compute(VOFrame *frame){

    std::vector<cv::KeyPoint> keypoints;
    for (const auto &p : frame->points ){
        cv::KeyPoint kp;
        kp.pt = p;
        keypoints.push_back(std::move(kp));
    }
    descriptor_->compute(frame->image, keypoints, frame->descriptors);
}

