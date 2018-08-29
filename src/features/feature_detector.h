
#ifndef VO_FEATURES_DETECTION_H
#define VO_FEATURES_DETECTION_H

#include <list>
#include <vector>
#include <cv.hpp>

#include <opencv2/features2d.hpp>
#include "src/visual_odemetry/vo_frame.h"

class FeatureDetector {

public:
    FeatureDetector();

    void detect(VOFrame *frame);
    void compute(VOFrame *frame);

private:
    cv::Ptr<cv::FastFeatureDetector> detector_;
};


#endif //VO_FEATURES_DETECTION_H
