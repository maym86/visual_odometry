
#ifndef VISUALODEMETRY_FETAURE_TRACKER_CUDA_H
#define VISUALODEMETRY_FETAURE_TRACKER_CUDA_H

#if __has_include("opencv2/cudaoptflow.hpp")

#include <list>
#include <vector>
#include <cv.hpp>

#include <opencv2/cudaoptflow.hpp>

#include "src/visual_odemetry/vo_frame.h"

class FeatureTracker {
public:
    FeatureTracker();

    void trackPointsGPU(VOFrame *vo0,  VOFrame *vo1);

    void trackPoints(VOFrame *vo0,  VOFrame *vo1);

private:
    void removePoints(VOFrame *vo0, VOFrame *vo1, std::vector<unsigned char> *status);

    cv::Ptr<cv::cuda::SparsePyrLKOpticalFlow> gpu_optical_flow_;

};

#endif
#endif //VISUALODEMETRY_FETAURE_TRACKER_CUDA_H
