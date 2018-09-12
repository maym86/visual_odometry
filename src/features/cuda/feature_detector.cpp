

#include "feature_detector.h"

FeatureDetector::FeatureDetector(){
    gpu_detector_ = cv::cuda::FastFeatureDetector::create(20, true, cv::FastFeatureDetector::TYPE_9_16, kMaxFeatures);
    descriptor_ = cv::cuda::ORB::create(3000);
}

void FeatureDetector::detectFAST(VOFrame *frame) {
    std::vector<cv::KeyPoint> keypoints;

    gpu_detector_->detect(frame->gpu_image, keypoints);

    frame->points.clear();
    for(const auto &kp : keypoints){
        frame->points.push_back(kp.pt);
    }
}


void FeatureDetector::detectComputeORB(const VOFrame &frame, std::vector<cv::KeyPoint> *keypoints, cv::Mat *descriptors){
    cv::cuda::GpuMat descriptors_gpu;
    descriptor_->detectAndCompute(frame.gpu_image, cv::noArray(), *keypoints, descriptors_gpu);
    descriptors_gpu.download(*descriptors);
}