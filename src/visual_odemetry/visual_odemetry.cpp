#include "visual_odemetry.h"

#include <algorithm>
#include <random>
#include <random>



#include <glog/logging.h>

VisualOdemetry::VisualOdemetry(double focal, const cv::Point2d &pp) {

    tracking_ = false;
    focal_ = focal;
    pp_ = pp;
}


void VisualOdemetry::triangulate(VOFrame *prev, VOFrame *now){
    if(!now->P.empty()) {
        cv::Mat points_3d;
        cv::Mat_<double> p0(2,prev->points.size(),CV_64FC1);
        cv::Mat_<double> p1(2,now->points.size(),CV_64FC1);

        for (int i = 0; i < p0.cols; i++) {
            p0.at<double>(0, i)  = prev->points[i].x;
            p0.at<double>(1, i)  = prev->points[i].y;
            p1.at<double>(0, i)  = now->points[i].x;
            p1.at<double>(1, i)  = now->points[i].y;
        }

        cv::Mat P = cv::Mat::eye(3, 4, CV_64FC1);
        cv::triangulatePoints(P, now->P, p0, p1, points_3d); //Relative to previous frame center
        now->points_3d.clear();

        for (int i = 0; i < points_3d.cols; i++) {
            now->points_3d.push_back(cv::Point3d(points_3d.at<double>(0, i) / points_3d.at<double>(3, i),
                                                 points_3d.at<double>(1, i) / points_3d.at<double>(3, i),
                                                 points_3d.at<double>(2, i) / points_3d.at<double>(3, i)));
        }
    }
}


double VisualOdemetry::getScale(const VOFrame &prev, const VOFrame &now, int num_points) {


    if (prev.points_3d.size() == 0) {
        return 1;
    }

    //Pick random points in prev that match to two points in now;
    std::vector<int> points;

    for(int i = 0; i < num_points; i++){
        for(int j = 0; j < 1000; j++){
            int index = rand() % static_cast<int>(now.points_3d.size() + 1);
            if(now_.mask.at<bool>(index) && now_.mask.at<bool>(prev.tracked_index[index]) &&
                    std::find(points.begin(), points.end(), index) == points.end()) {

                points.push_back(index);
                break;
            }
        }
    }

    double now_sum = 0;
    double prev_sum = 0;
    for(int i = 0; i < points.size()-1; i++){
        int i0 = points[i];
        int i1 = points[i+1];
        double n0 = cv::norm(now.points_3d[i0] - now.points_3d[i1]);
        double n1 = cv::norm(prev.points_3d[prev.tracked_index[i0]] - prev.points_3d[prev.tracked_index[i1]]);

        if(!std::isnan(n0) && !std::isnan(n1) && !std::isinf(n0) && !std::isinf(n1)) {
            now_sum += n0;
            prev_sum += n1;
        }
    }

    return  now_sum / prev_sum ;
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

    color_ = cv::Scalar(0,0,255);
    if (!tracking_) {

        color_ = cv::Scalar(255,0,0);
        prev_.points = feature_detector_.detect(prev_.gpu_image);
        tracking_ = true;
    }

    feature_tracker_.trackPoints(&prev_, &now_); //TODO pass voframe and add fields for matching later

    LOG(INFO) << now_.points.size();
    if (now_.points.size() < kMinTrackedPoints) {
        tracking_ = false;
    }

    now_.E = cv::findEssentialMat( now_.points, prev_.points,  focal_, pp_, cv::RANSAC, 0.999, 1.0, now_.mask);
    int res = recoverPose(now_.E, now_.points, prev_.points, now_.R, now_.t, focal_, pp_, now_.mask);

    if(res > 10) {
        triangulate(&prev_, &now_);
        double scale = getScale(prev_, now_, 500);

        LOG(INFO) << scale;

        hconcat(now_.R, now_.t, now_.P);
        now_.pose_R = now_.R * prev_.pose_R;
        now_.pose_t += scale * (prev_.pose_R * now_.t);
    }

    hconcat(now_.pose_R, now_.pose_t, now_.pose);
    LOG(INFO) << "\n" << now_.pose;



    //Kalman Filter
    kf_.setMeasurements(now_.pose_R, now_.pose_t);
    cv::Mat k_R, k_t;
    kf_.updateKalmanFilter(&k_R, &k_t);

    hconcat(k_R, k_t, *pose_kalman);
    LOG(INFO) << "\n" << pose_kalman;

    (*pose) = now_.pose;
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

