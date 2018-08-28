
#ifndef VISUALODEMETRY_FETAURE_TRACKER_H
#define VISUALODEMETRY_FETAURE_TRACKER_H

#include <list>
#include <vector>
#include <cv.hpp>

#include <opencv2/optflow.hpp>

#include "src/visual_odemetry/vo_frame.h"

class FeatureTracker {
public:
    FeatureTracker();

    void trackPoints(VOFrame *vo0,  VOFrame *vo1);

private:

    cv::Ptr<cv::SparsePyrLKOpticalFlow> optical_flow_;

};
#endif //VISUALODEMETRY_FETAURE_TRACKER_H
