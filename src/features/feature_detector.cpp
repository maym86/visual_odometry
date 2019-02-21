#include "feature_detector.h"

FeatureDetector::FeatureDetector(){
    detector_ = cv::FastFeatureDetector::create(20, true, cv::FastFeatureDetector::TYPE_9_16);
    descriptor_ = cv::ORB::create(5000);

    akaze_ = cv::AKAZE::create();
    brisk_ = cv::BRISK::create();

}

void FeatureDetector::detectFAST(VOFrame *frame) {
    std::vector<cv::KeyPoint> keypoints;

    detector_->detect(frame->image, keypoints);

    for(const auto &kp : keypoints){
        frame->points.push_back(kp.pt);
    }
}


void FeatureDetector::detectFAST(const VOFrame &frame, std::vector<cv::KeyPoint> *keypoints) {
    detector_->detect( frame.image, *keypoints);
}

void FeatureDetector::detectComputeORB(const VOFrame &frame, std::vector<cv::KeyPoint> *keypoints, cv::Mat *descriptors){
    descriptor_->detectAndCompute(frame.image, cv::noArray(), *keypoints, *descriptors);
}


void FeatureDetector::computeORB(const VOFrame &frame, std::vector<cv::KeyPoint> *keypoints,  cv::Mat *descriptors){

    keypoints->clear();
    for (const auto &p: frame.points){
        cv::KeyPoint kp;
        kp.pt = p;
        keypoints->push_back(kp);
    }

    descriptor_->compute(frame.image, *keypoints, *descriptors);
}

void FeatureDetector::detectComputeAKAZE(const VOFrame &frame, std::vector<cv::KeyPoint> *keypoints, cv::Mat *descriptors){
    akaze_->detectAndCompute(frame.image, cv::noArray(), *keypoints, *descriptors);
}

void FeatureDetector::detectComputeBRISK(const VOFrame &frame, std::vector<cv::KeyPoint> *keypoints, cv::Mat *descriptors){
    brisk_->detectAndCompute(frame.image, cv::noArray(), *keypoints, *descriptors);
}
