
#include "bundle_adjustment.h"

#include <glog/logging.h>

BundleAdjustment::BundleAdjustment(size_t max_frames) {
    adjuster_ = cv::makePtr<cv::detail::BundleAdjusterReproj>();
    matcher_ = cv::makePtr<cv::detail::BestOf2NearestRangeMatcher>(3, true, 0.3);
    max_frames_ =  max_frames;
}

void BundleAdjustment::addKeyFrame(const VOFrame &frame, float focal, cv::Point2d pp){
    cv::detail::CameraParams camera;
    cv::detail::ImageFeatures features;
    cv::detail::MatchesInfo pairwise_matches;

    //TODO populate
    camera.R = frame.pose_R.clone();
    camera.t = frame.pose_t.clone();
    camera.ppx = pp.x;
    camera.ppy = pp.y;
    camera.focal = focal;
    camera.aspect = static_cast<float>(frame.image.rows) / static_cast<float>(frame.image.cols);

    features.img_size = frame.image.size();
    features.descriptors = frame.descriptors;

    for (const auto &p : frame.points) {
        cv::KeyPoint kp;
        kp.pt = p;
        features.keypoints.emplace_back(kp);
    }

    //set points and descriptors


    cameras_.push_back(camera);

    if(cameras_.size() > max_frames_) {
        cameras_.erase(cameras_.begin());
        features_.erase(features_.begin());
    }

    //Do pairwise matching first
    (*matcher_)(features, pairwise_matches_);
    matcher_->collectGarbage();
}

int BundleAdjustment::slove(cv::Mat *R, cv::Mat *t) {
    if (!(*adjuster_)(features_, pairwise_matches_, cameras_)) {
        LOG(INFO) << "Camera parameters adjusting failed.";
        return 1;
    }

    return 0;

}