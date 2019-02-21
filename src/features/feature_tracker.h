
#ifndef VO_FETAURES_TRACKER_H
#define VO_FETAURES_TRACKER_H

#include <list>
#include <vector>
#include <cv.hpp>

#include <opencv2/video/tracking.hpp>

#include "src/visual_odometry/vo_frame.h"

class FeatureTracker {
public:
    FeatureTracker();

    float trackPoints(VOFrame *vo0,  VOFrame *vo1);

private:

    cv::Ptr<cv::SparsePyrLKOpticalFlow> optical_flow_;

};
#endif //VO_FETAURES_TRACKER_H
