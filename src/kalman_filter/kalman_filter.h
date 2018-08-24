

#ifndef VO_KALMAN_FILTER_H
#define VO_KALMAN_FILTER_H

#include <cv.hpp>

#include <opencv2/video/tracking.hpp>


class KalmanFilter {

public:
    KalmanFilter();

    void setMeasurements(const cv::Mat &rotation_measured, const cv::Mat &translation_measured);

    void updateKalmanFilter(cv::Mat *rotation_estimated , cv::Mat *translation_estimated);
private:

    void initKalmanFilter(int nStates, int nMeasurements, int nInputs, double dt);

    cv::KalmanFilter kf_;
    cv::Mat measurements_;
};


#endif //VO_KALMAN_FILTER_H
