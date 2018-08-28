
#ifndef VO_FETAURES_CUDA_TRACKER_H
#define VO_FETAURES_CUDA_TRACKER_H


#include <list>
#include <vector>
#include <cv.hpp>

#include <opencv2/cudaoptflow.hpp>

#include "src/visual_odemetry/vo_frame.h"

class FeatureTracker {
public:
    FeatureTracker();

    void trackPoints(VOFrame *vo0,  VOFrame *vo1);

private:
    cv::Ptr<cv::cuda::SparsePyrLKOpticalFlow> gpu_optical_flow_;

};

#endif //VO_FETAURES_CUDA_TRACKER_H
