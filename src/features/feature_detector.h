
#ifndef VO_FEATURES_DETECTION_H
#define VO_FEATURES_DETECTION_H

#include <list>
#include <vector>
#include <cv.hpp>

#include "opencv2/features2d.hpp"
#include "src/visual_odometry/vo_frame.h"

class FeatureDetector {

public:
    FeatureDetector();

    void detectFAST(VOFrame *frame);
    void detectComputeORB(const VOFrame &frame, std::vector<cv::KeyPoint> *keypoints, cv::Mat *descriptors);
    void computeORB(const VOFrame &frame, std::vector<cv::KeyPoint> *keypoints, cv::Mat *descriptors);

private:
    cv::Ptr<cv::FastFeatureDetector> detector_;
    cv::Ptr<cv::ORB> descriptor_;
};


#endif //VO_FEATURES_DETECTION_H
