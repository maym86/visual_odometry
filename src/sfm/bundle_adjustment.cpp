
#include "bundle_adjustment.h"

#include <glog/logging.h>

void BundleAdjustment::init(size_t max_frames) {
    adjuster_ = cv::makePtr<cv::detail::BundleAdjusterReproj>();
    matcher_ = cv::makePtr<cv::detail::BestOf2NearestMatcher>(true, 0.3, 10, 10);
    max_frames_ =  max_frames;
}

void BundleAdjustment::addKeyFrame(const VOFrame &frame, float focal, cv::Point2d pp){
    cv::detail::CameraParams camera;
    cv::detail::ImageFeatures image_feature;
    cv::detail::MatchesInfo pairwise_matches;

    frame.pose_R.convertTo(camera.R, CV_32F);
    camera.t = frame.pose_t.clone();
    camera.ppx = pp.x;
    camera.ppy = pp.y;
    camera.focal = focal;
    camera.aspect = static_cast<float>(frame.image.rows) / static_cast<float>(frame.image.cols);

    //TODO probably want to use a subset of these points rather than all for speed - or pull out new ORB points here??



    image_feature.img_size = frame.image.size();
    image_feature.descriptors = frame.descriptors.getUMat(cv::USAGE_DEFAULT);
    image_feature.img_idx =  count_;
    for (const auto &p : frame.points) {
        cv::KeyPoint kp;
        kp.pt = p;
        image_feature.keypoints.push_back(std::move(kp));
    }

    features_.push_back(image_feature);
    cameras_.push_back(camera);

    if(cameras_.size() > max_frames_) {
        cameras_.erase(cameras_.begin());
        features_.erase(features_.begin());
    }

    //Do pairwise matching first
    (*matcher_)(features_, pairwise_matches_);
    //matcher_->collectGarbage();
    count_++;
}

int BundleAdjustment::slove(cv::Mat *R, cv::Mat *t) {

    if (!(*adjuster_)(features_, pairwise_matches_, cameras_)) {
        LOG(INFO) << "Camera parameters adjusting failed.";
        return 1;
    }

    auto last_cam = cameras_[cameras_.size() - 1];

    last_cam.R.convertTo(*R, CV_64F);
    *t = last_cam.t;

    return 0;

}