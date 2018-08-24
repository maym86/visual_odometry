
#ifndef VISUALODEMETRY_FETAURE_TRACKER_H
#define VISUALODEMETRY_FETAURE_TRACKER_H

#include <list>
#include <vector>
#include <cv.hpp>

#include <opencv2/cudaoptflow.hpp>
#include "src/visual_odemetry/vo_frame.h"

class FeatureTracker {
public:
    FeatureTracker();

    void trackPoints(VOFrame *prev,  VOFrame *now);

private:
    cv::Ptr<cv::cuda::SparsePyrLKOpticalFlow> optical_flow_;
};
#endif //VISUALODEMETRY_FETAURE_TRACKER_H
