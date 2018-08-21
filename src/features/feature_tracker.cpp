
#include <opencv/cv.hpp>
#include "feature_tracker.h"

std::vector<cv::Point2f> trackPoints(const cv::Mat &img0, const cv::Mat &img1, std::vector<cv::Point2f> *prev_points) {

    std::vector<cv::Point2f> next_points;

    std::vector<float> err;
    cv::Size win_size(21, 21);
    cv::TermCriteria term_criteria = cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01);

    std::vector<uchar> status;

    cv::calcOpticalFlowPyrLK(img0, img1, *prev_points, next_points,
                             status, err, win_size, 3, term_criteria, 0, 0.001);

    //Remove bad points
    for (int i = status.size() - 1; i >= 0; --i) {

        cv::Point2f pt = next_points[i];
        if (status[i] == 0 || pt.x < 0 || pt.y < 0) {
            if (pt.x < 0 || pt.y < 0) {
                status[i] = 0;
            }
            prev_points->erase(prev_points->begin() + i);
            next_points.erase(next_points.begin() + i);
        }
    }

    return next_points;
}

