
#ifndef VISUALODEMETRY_FEATURES_DETECTION_H
#define VISUALODEMETRY_FEATURES_DETECTION_H

#include <list>
#include <vector>
#include "opencv2/features2d/features2d.hpp"

class FeatureDetector {

public:

    FeatureDetector();

    std::vector<cv::Point2f> detect(const cv::Mat &image);

private:
    const int kMinFeatures = 3000;
    cv::Ptr<cv::FeatureDetector> detector_;
};


#endif //VISUALODEMETRY_FEATURES_DETECTION_H
