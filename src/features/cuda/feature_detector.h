
#ifndef VISUALODEMETRY_FEATURES_DETECTION_CUDA_H
#define VISUALODEMETRY_FEATURES_DETECTION_CUDA_H

#if __has_include("opencv2/cudafeatures2d.hpp")

#include <list>
#include <vector>
#include <cv.hpp>

#include <opencv2/cudafeatures2d.hpp>

#include "src/visual_odemetry/vo_frame.h"

class FeatureDetector {

public:

    FeatureDetector();

    void detect(VOFrame *frame);

private:
    const int kMaxFeatures = 10000;

#ifdef HASCUDA
    cv::Ptr<cv::cuda::FastFeatureDetector> gpu_detector_;
#endif

    cv::Ptr<cv::FastFeatureDetector> detector_;
};

#endif
#endif //VISUALODEMETRY_FEATURES_DETECTION_CUDA_H
