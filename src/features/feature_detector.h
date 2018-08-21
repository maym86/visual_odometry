
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
    const int kMinFeatures = 5000;
    cv::Ptr<cv::cuda::ORB> gpu_detector_;
};


#endif //VISUALODEMETRY_FEATURES_DETECTION_H
