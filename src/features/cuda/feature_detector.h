
#ifndef VO_FEATURES_CUDA_DETECTION_H
#define VO_FEATURES_CUDA_DETECTION_H

#include <list>
#include <vector>
#include <cv.hpp>

#include <opencv2/cudafeatures2d.hpp>

#include "src/visual_odometry/vo_frame.h"

class FeatureDetector {

public:

    FeatureDetector();

    void detect(VOFrame *frame);

private:
    const int kMaxFeatures = 10000;

    cv::Ptr<cv::cuda::FastFeatureDetector> gpu_detector_;
};

#endif //VO_FEATURES_CUDA_DETECTION_H
