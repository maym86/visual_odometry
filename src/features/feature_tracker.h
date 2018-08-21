
#ifndef VISUALODEMETRY_FETAURE_TRACKER_H
#define VISUALODEMETRY_FETAURE_TRACKER_H

#include <list>
#include <vector>
#include <cv.hpp>


std::vector<cv::Point2f> trackPoints(const cv::Mat &img0, const cv::Mat &img1, std::vector<cv::Point2f>* prev_points);


#endif //VISUALODEMETRY_FETAURE_TRACKER_H
