
#ifndef VISUALODEMETRY_FETAURE_TRACKER_H
#define VISUALODEMETRY_FETAURE_TRACKER_H

#include <list>
#include <vector>
#include <cv.hpp>

class FeatureTracker {

public:
    void addFrame(const cv::Mat &images);
    std::vector<cv::Point2f> getMatches(std::vector<cv::Point2f>* prev_points);

private:
    std::list<cv::Mat> images_;
    std::vector<cv::Point2f>* current_points_;
};


#endif //VISUALODEMETRY_FETAURE_TRACKER_H
