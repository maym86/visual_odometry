
#ifndef VISUALODEMETRY_FEATURES_DETECTION_H
#define VISUALODEMETRY_FEATURES_DETECTION_H

#include <list>
#include <vector>

#include <opencv2/cudafeatures2d.hpp>


class FeatureDetector {

public:

    FeatureDetector();

    std::vector<cv::Point2f> detect(const cv::cuda::GpuMat &image_gpu);

private:
    const int kMaxFeatures = 10000;
    cv::Ptr<cv::cuda::FastFeatureDetector> gpu_detector_; //TODO add ORB as an optiion too maybe both
};


#endif //VISUALODEMETRY_FEATURES_DETECTION_H
