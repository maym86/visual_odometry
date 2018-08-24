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

    voOLD_ = vo0_;
    vo0_ = vo1_;

    //Get new GPU image
    cv::Mat image_grey;
    cv::cvtColor(image, image_grey, CV_BGR2GRAY);
    vo1_.gpu_image.upload(image_grey);

    if (vo0_.gpu_image.empty()) {
        return;
    }

    bool new_keypoints =  false;
    color_ = cv::Scalar(0,0,255);
    if (!tracking_) {
        color_ = cv::Scalar(255,0,0);
        vo0_.points = feature_detector_.detect(vo0_.gpu_image);
        //TODO track back to vo-1 for 3d
        if(!voOLD_.gpu_image.empty()){ //TODO verify this and clean up naming use buffer??
            feature_tracker_.trackPoints(&vo0_, &voOLD_);
            vo0_.E = cv::findEssentialMat( vo0_.points, voOLD_.points,  focal_, pp_, cv::RANSAC, 0.999, 1.0, vo0_.mask);

            //TODO This R|t is not the same as the pose it used before
            int res = recoverPose(vo0_.E, vo0_.points, voOLD_.points, vo0_.R, vo0_.t, focal_, pp_, vo0_.mask);

            if(res > 10) {
                triangulate(&voOLD_, &vo0_);
            }

        }
        tracking_ = true;
        new_keypoints = true;
    }

    feature_tracker_.trackPoints(&vo0_, &vo1_);

    LOG(INFO) << vo1_.points.size();
    if (vo1_.points.size() < kMinTrackedPoints) {
        tracking_ = false;
    }

    vo1_.E = cv::findEssentialMat( vo1_.points, vo0_.points,  focal_, pp_, cv::RANSAC, 0.999, 1.0, vo1_.mask);
    int res = recoverPose(vo1_.E, vo1_.points, vo0_.points, vo1_.R, vo1_.t, focal_, pp_, vo1_.mask);

    if(res > 10) {
        triangulate(&vo0_, &vo1_);

        double scale = getScale(vo0_, vo1_, 50);


        LOG(INFO) << new_keypoints << " " << scale;

        hconcat(vo1_.R, vo1_.t, vo1_.P);
        vo1_.pose_R = vo1_.R * vo0_.pose_R;
        vo1_.pose_t += scale * (vo0_.pose_R * vo1_.t);
    }

    hconcat(vo1_.pose_R, vo1_.pose_t, vo1_.pose);

    //Kalman Filter
    kf_.setMeasurements(vo1_.pose_R, vo1_.pose_t);
    cv::Mat k_R, k_t;
    kf_.updateKalmanFilter(&k_R, &k_t);

    hconcat(k_R, k_t, *pose_kalman);
    //LOG(INFO) << "\n" << pose_kalman;

    (*pose) = vo1_.pose;

    //TODO keep sliding window and use bundle adjeustment to correct pos of last frame
}

cv::Mat VisualOdemetry::drawMatches(const cv::Mat &image){
    cv::Mat output = image.clone();
    for (int i = 0; i < vo0_.points.size(); i++) {
        if (vo1_.mask.at<bool>(i)) {
            cv::line(output, vo0_.points[i], vo1_.points[i], color_, 2);
        }
    }

    return output;
}

