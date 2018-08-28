#include "visual_odemetry.h"

#include "src/sfm/triangulation.h"

#include <glog/logging.h>

VisualOdemetry::VisualOdemetry(double focal, const cv::Point2d &pp) {

    tracking_ = false;
    focal_ = focal;
    pp_ = pp;
}

void VisualOdemetry::addImage(const cv::Mat &image, cv::Mat *pose, cv::Mat *pose_kalman){
    //Store previous frame data //TODO make buffer

    vo0_ = vo1_;
    vo1_ = vo2_;

    //Get new GPU image
    cv::Mat image_grey;
    cv::cvtColor(image, image_grey, CV_BGR2GRAY);
    vo2_.gpu_image.upload(image_grey);
    vo2_.image = image;

    if (vo1_.gpu_image.empty() || vo0_.gpu_image.empty()) {
        return;
    }

    bool new_keypoints =  false;
    color_ = cv::Scalar(0,0,255);
    if (!tracking_) {
        color_ = cv::Scalar(255,0,0);
        feature_detector_.detect(&vo1_);
        //TODO track back to vo-1 for 3d
        if(!vo0_.gpu_image.empty()){ //TODO verify this and clean up naming use buffer??
            feature_tracker_.trackPoints(&vo1_, &vo0_);
            vo1_.E = cv::findEssentialMat( vo1_.points, vo0_.points,  focal_, pp_, cv::RANSAC, 0.999, 1.0, vo1_.mask);

            //TODO This R|t is not the same as the pose it used before - WORK ONE FRAME IN THE PAST vo1
            int res = recoverPose(vo1_.E, vo1_.points, vo0_.points, vo1_.R, vo1_.t, focal_, pp_, vo1_.mask);

            if(res > 10) {
                triangulate(&vo0_, &vo1_);
            }

        }
        tracking_ = true;
        new_keypoints = true;
    }

    feature_tracker_.trackPoints(&vo1_, &vo2_);

    LOG(INFO) << vo2_.points.size();
    if (vo2_.points.size() < kMinTrackedPoints) {
        tracking_ = false;
    }

    vo2_.E = cv::findEssentialMat( vo2_.points, vo1_.points,  focal_, pp_, cv::RANSAC, 0.999, 1.0, vo2_.mask);
    int res = recoverPose(vo2_.E, vo2_.points, vo1_.points, vo2_.R, vo2_.t, focal_, pp_, vo2_.mask);

    if(res > 10) {

        hconcat(vo2_.R, vo2_.t, vo2_.P);
        triangulate(&vo1_, &vo2_);

        double scale = getScale(vo1_, vo2_, 10, 200);

        LOG(INFO) << new_keypoints << " " << scale;


        vo2_.pose_R = vo2_.R * vo1_.pose_R;
        vo2_.pose_t += scale * (vo1_.pose_R * vo2_.t);
    }

    hconcat(vo2_.pose_R, vo2_.pose_t, vo2_.pose);

    //Kalman Filter
    kf_.setMeasurements(vo2_.pose_R, vo2_.pose_t);
    cv::Mat k_R, k_t;
    kf_.updateKalmanFilter(&k_R, &k_t);

    hconcat(k_R, k_t, *pose_kalman);
    //LOG(INFO) << "\n" << pose_kalman;

    (*pose) = vo2_.pose;

    //TODO keep sliding window and use bundle adjeustment to correct pos of last frame
}

cv::Mat VisualOdemetry::drawMatches(const cv::Mat &image){
    cv::Mat output = image.clone();
    for (int i = 0; i < vo1_.points.size(); i++) {
        if (vo2_.mask.at<bool>(i)) {
            cv::line(output, vo1_.points[i], vo2_.points[i], color_, 2);
        }
    }

    return output;
}

