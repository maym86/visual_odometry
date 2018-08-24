#include "visual_odemetry.h"

#include "src/sfm/triangulation.h"

#include <glog/logging.h>

VisualOdemetry::VisualOdemetry(double focal, const cv::Point2d &pp) {

    tracking_ = false;
    focal_ = focal;
    pp_ = pp;
}

void VisualOdemetry::addImage(const cv::Mat &image, cv::Mat *pose, cv::Mat *pose_kalman){
    //Store previous frame data
    prev_ = now_;

    //Get new GPU image
    cv::Mat image_grey;
    cv::cvtColor(image, image_grey, CV_BGR2GRAY);
    now_.gpu_image.upload(image_grey);

    if (prev_.gpu_image.empty()) {
        prev_ = now_;
        return;
    }

    bool new_keypoints =  false;
    color_ = cv::Scalar(0,0,255);
    if (!tracking_) {
        color_ = cv::Scalar(255,0,0);
        prev_.points = feature_detector_.detect(prev_.gpu_image);
        tracking_ = true;
        new_keypoints = true;
    }

    feature_tracker_.trackPoints(&prev_, &now_); //TODO pass voframe and add fields for matching later

    LOG(INFO) << now_.points.size();
    if (now_.points.size() < kMinTrackedPoints) {
        tracking_ = false;
    }

    now_.E = cv::findEssentialMat( now_.points, prev_.points,  focal_, pp_, cv::RANSAC, 0.999, 1.0, now_.mask);
    int res = recoverPose(now_.E, now_.points, prev_.points, now_.R, now_.t, focal_, pp_, now_.mask);

    if(res > 10) {
        triangulate(&prev_, &now_); //Prob

        double scale = 1.0;
        /*if(!new_keypoints) {
            scale = getScale(prev_, now_, 500);
        }*/

        LOG(INFO) << new_keypoints << " " << scale;

        hconcat(now_.R, now_.t, now_.P);
        now_.pose_R = now_.R * prev_.pose_R;
        now_.pose_t += scale * (prev_.pose_R * now_.t);
    }

    hconcat(now_.pose_R, now_.pose_t, now_.pose);

    //Kalman Filter
    kf_.setMeasurements(now_.pose_R, now_.pose_t);
    cv::Mat k_R, k_t;
    kf_.updateKalmanFilter(&k_R, &k_t);

    hconcat(k_R, k_t, *pose_kalman);
    //LOG(INFO) << "\n" << pose_kalman;

    (*pose) = now_.pose;

    //TODO keep sliding window and use bundle adjeustment to correct pos of last frame
}

cv::Mat VisualOdemetry::drawMatches(const cv::Mat &image){
    cv::Mat output = image.clone();
    for (int i = 0; i < prev_.points.size(); i++) {
        if (now_.mask.at<bool>(i)) {
            cv::line(output, prev_.points[i], now_.points[i], color_, 2);
        }
    }

    return output;
}

