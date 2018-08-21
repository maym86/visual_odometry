
#include <opencv/cv.hpp>
#include "feature_tracker.h"


std::vector<cv::Point2f> trackPoints(const cv::Mat &img0, const cv::Mat &img1, std::vector<cv::Point2f>* prev_points){

    std::vector<cv::Point2f> next_points;

    std::vector<float> err;
    cv::Size win_size(21,21);
    cv::TermCriteria term_criteria=cv::TermCriteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS, 30, 0.01);

    std::vector<uchar> status;
    cv::calcOpticalFlowPyrLK(img0, img1, *prev_points, next_points,
            status, err, win_size, 3, term_criteria, 0, 0.001);

    //getting rid of points for which the KLT tracking failed or those who have gone outside the frame
    int index_correction = 0;
    for( int i=0; i< status.size(); i++)
    {  cv::Point2f pt = next_points.at(i- index_correction);
        if ((status.at(i) == 0)||(pt.x<0)||(pt.y<0))	{
            if((pt.x<0)||(pt.y<0))	{
                status.at(i) = 0;
            }

            prev_points->erase(prev_points->begin() + i - index_correction);
            next_points.erase(next_points.begin() + i - index_correction);
            index_correction++;
        }
    }

    return next_points;
}

