#include "visual_odemetry.h"

#include "src/sfm/triangulation.h"

#include <glog/logging.h>

VisualOdemetry::VisualOdemetry(double focal, const cv::Point2d &pp) {
    tracking_ = false;
    focal_ = focal;
    pp_ = pp;
}

void VisualOdemetry::addImage(const cv::Mat &image, cv::Mat *pose, cv::Mat *pose_kalman){
    //TODO make buffer
    vo0_ = vo1_;
    vo1_ = vo2_;
    vo2_.setImage(image);

    if (vo0_.image.empty() || vo1_.image.empty()) {
        return;
    }

    bool new_keypoints =  false;
    color_ = cv::Scalar(0,0,255);
    if (!tracking_) {
        color_ = cv::Scalar(255,0,0);
        feature_detector_.detect(&vo1_);
        //TODO track back to vo-1 for 3d
        if(!vo0_.image.empty() && !vo1_.E.empty()){ //Backtrack for new points for scale calculation later
            feature_tracker_.trackPoints(&vo1_, &vo0_);
            //This is just to find the mask using RANSAC - we already have R|t from vo0 to vo1
            //TODO change this so it just rejects based on existing R|t rather than new calc -- use perspective transform instead
            cv::findEssentialMat( vo1_.points, vo0_.points, focal_, pp_, cv::RANSAC, 0.999, 1.0, vo1_.mask);
            triangulate(&vo0_, &vo1_);
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

    if(res > kMinPosePoints) {

        hconcat(vo2_.R, vo2_.t, vo2_.P);
        triangulate(&vo1_, &vo2_);

        double scale = getScale(vo1_, vo2_, kMinPosePoints, 200);

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

    //TODO keep sliding window and use bundle adjustment to correct pos of last frame
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

