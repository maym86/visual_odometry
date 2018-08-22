

#include <src/utils/utils.h>
#include "kalman_filter.h"


KalmanFilter::KalmanFilter(){
    int nStates = 18;            // the number of states
    int nMeasurements = 6;       // the number of measured states
    int nInputs = 0;             // the number of action control
    double dt = 0.1;           // time between measurements (1/FPS)

    initKalmanFilter(nStates, nMeasurements, nInputs, dt);    // init function
}


void KalmanFilter::initKalmanFilter(int nStates, int nMeasurements, int nInputs, double dt) {
    kf_.init(nStates, nMeasurements, nInputs, CV_64F);                 // init Kalman Filter
    cv::setIdentity(kf_.processNoiseCov, cv::Scalar::all(1e-5));       // set process noise
    cv::setIdentity(kf_.measurementNoiseCov, cv::Scalar::all(1e-4));   // set measurement noise
    cv::setIdentity(kf_.errorCovPost, cv::Scalar::all(1));             // error covariance
    /* DYNAMIC MODEL */
    //  [1 0 0 dt  0  0 dt2   0   0 0 0 0  0  0  0   0   0   0]
    //  [0 1 0  0 dt  0   0 dt2   0 0 0 0  0  0  0   0   0   0]
    //  [0 0 1  0  0 dt   0   0 dt2 0 0 0  0  0  0   0   0   0]
    //  [0 0 0  1  0  0  dt   0   0 0 0 0  0  0  0   0   0   0]
    //  [0 0 0  0  1  0   0  dt   0 0 0 0  0  0  0   0   0   0]
    //  [0 0 0  0  0  1   0   0  dt 0 0 0  0  0  0   0   0   0]
    //  [0 0 0  0  0  0   1   0   0 0 0 0  0  0  0   0   0   0]
    //  [0 0 0  0  0  0   0   1   0 0 0 0  0  0  0   0   0   0]
    //  [0 0 0  0  0  0   0   0   1 0 0 0  0  0  0   0   0   0]
    //  [0 0 0  0  0  0   0   0   0 1 0 0 dt  0  0 dt2   0   0]
    //  [0 0 0  0  0  0   0   0   0 0 1 0  0 dt  0   0 dt2   0]
    //  [0 0 0  0  0  0   0   0   0 0 0 1  0  0 dt   0   0 dt2]
    //  [0 0 0  0  0  0   0   0   0 0 0 0  1  0  0  dt   0   0]
    //  [0 0 0  0  0  0   0   0   0 0 0 0  0  1  0   0  dt   0]
    //  [0 0 0  0  0  0   0   0   0 0 0 0  0  0  1   0   0  dt]
    //  [0 0 0  0  0  0   0   0   0 0 0 0  0  0  0   1   0   0]
    //  [0 0 0  0  0  0   0   0   0 0 0 0  0  0  0   0   1   0]
    //  [0 0 0  0  0  0   0   0   0 0 0 0  0  0  0   0   0   1]
    // position
    kf_.transitionMatrix.at<double>(0,3) = dt;
    kf_.transitionMatrix.at<double>(1,4) = dt;
    kf_.transitionMatrix.at<double>(2,5) = dt;
    kf_.transitionMatrix.at<double>(3,6) = dt;
    kf_.transitionMatrix.at<double>(4,7) = dt;
    kf_.transitionMatrix.at<double>(5,8) = dt;
    kf_.transitionMatrix.at<double>(0,6) = 0.5*pow(dt,2);
    kf_.transitionMatrix.at<double>(1,7) = 0.5*pow(dt,2);
    kf_.transitionMatrix.at<double>(2,8) = 0.5*pow(dt,2);
    // orientation
    kf_.transitionMatrix.at<double>(9,12) = dt;
    kf_.transitionMatrix.at<double>(10,13) = dt;
    kf_.transitionMatrix.at<double>(11,14) = dt;
    kf_.transitionMatrix.at<double>(12,15) = dt;
    kf_.transitionMatrix.at<double>(13,16) = dt;
    kf_.transitionMatrix.at<double>(14,17) = dt;
    kf_.transitionMatrix.at<double>(9,15) = 0.5*pow(dt,2);
    kf_.transitionMatrix.at<double>(10,16) = 0.5*pow(dt,2);
    kf_.transitionMatrix.at<double>(11,17) = 0.5*pow(dt,2);
    /* MEASUREMENT MODEL */
    //  [1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
    //  [0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
    //  [0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
    //  [0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0]
    //  [0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0]
    //  [0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0]
    kf_.measurementMatrix.at<double>(0,0) = 1;  // x
    kf_.measurementMatrix.at<double>(1,1) = 1;  // y
    kf_.measurementMatrix.at<double>(2,2) = 1;  // z
    kf_.measurementMatrix.at<double>(3,9) = 1;  // roll
    kf_.measurementMatrix.at<double>(4,10) = 1; // pitch
    kf_.measurementMatrix.at<double>(5,11) = 1; // yaw
}


void KalmanFilter::setMeasurements(const cv::Mat &rotation_measured, const cv::Mat &translation_measured) {
    // Convert rotation matrix to euler angles
    cv::Mat measured_eulers(3, 1, CV_64F);
    measured_eulers = eulerAnglesToRotationMatrix(rotation_measured);
    // Set measurement to predict
    measurements_ = cv::Mat(6,1, CV_64F);

    measurements_.at<double>(0) = translation_measured.at<double>(0); // x
    measurements_.at<double>(1) = translation_measured.at<double>(1); // y
    measurements_.at<double>(2) = translation_measured.at<double>(2); // z
    measurements_.at<double>(3) = measured_eulers.at<double>(0);      // roll
    measurements_.at<double>(4) = measured_eulers.at<double>(1);      // pitch
    measurements_.at<double>(5) = measured_eulers.at<double>(2);      // yaw
}

void KalmanFilter::updateKalmanFilter(cv::Mat *rotation_estimated , cv::Mat *translation_estimated)
{
    // First predict, to update the internal statePre variable
    cv::Mat prediction = kf_.predict();
    // The "correct" phase that is going to use the predicted value and our measurement
    cv::Mat estimated = kf_.correct(measurements_);
    // Estimated translation
    (*translation_estimated) = cv::Mat(3, 1, CV_64F);
    translation_estimated->at<double>(0) = estimated.at<double>(0);
    translation_estimated->at<double>(1) = estimated.at<double>(1);
    translation_estimated->at<double>(2) = estimated.at<double>(2);
    // Estimated euler angles
    cv::Mat eulers_estimated(3, 1, CV_64F);
    eulers_estimated.at<double>(0) = estimated.at<double>(9);
    eulers_estimated.at<double>(1) = estimated.at<double>(10);
    eulers_estimated.at<double>(2) = estimated.at<double>(11);
    // Convert estimated quaternion to rotation matrix
    (*rotation_estimated) = eulerAnglesToRotationMatrix(eulers_estimated);
}


