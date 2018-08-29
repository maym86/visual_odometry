

#include "feature_detector.h"

FeatureDetector::FeatureDetector(){
    gpu_detector_ = cv::cuda::FastFeatureDetector::create(20, true, cv::FastFeatureDetector::TYPE_9_16, kMaxFeatures);
    descriptor_ = cv::cuda::ORB::create(); //TODO check for GPU version

}

void FeatureDetector::detect(VOFrame *frame) {
    std::vector<cv::KeyPoint> keypoints;

    gpu_detector_->detect(frame->gpu_image, keypoints);

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

    cv::cuda::GpuMat descriptors;
    descriptor_->compute(frame->gpu_image, keypoints, descriptors);
    descriptors.download(frame->descriptors);
}