
#include "bundle_adjustment.h"

#include <glog/logging.h>

void BundleAdjustment::init(size_t max_frames) {

    ParallelBA::DeviceT device = ParallelBA::PBA_CUDA_DEVICE_DEFAULT; //device = ParallelBA::PBA_CPU_DOUBLE;
    pba_ = ParallelBA(device);
    pba_.SetFixedIntrinsics(true);
    pba_.SetNextBundleMode(ParallelBA::BUNDLE_ONLY_MOTION); //Solving for motion only


    ///TODO replace bundle adjustment algo https://stackoverflow.com/questions/13921720/bundle-adjustment-functions
    // TODO sba https://stackoverflow.com/questions/52005362/sparse-bundle-adjustment-using-fiducial-markers
    adjuster_ = cv::makePtr<cv::detail::BundleAdjusterReproj>();
    matcher_ = cv::makePtr<cv::detail::BestOf2NearestMatcher>(true, 0.2, 10, 10);
    max_frames_ =  max_frames;
}

void BundleAdjustment::addKeyFrame(const VOFrame &frame, float focal, cv::Point2d pp, int feature_count){
    cv::detail::ImageFeatures image_feature;
    cv::detail::MatchesInfo pairwise_matches;

    CameraT cam;
    cam.f = focal;
    for(int r=0; r < frame.pose_R.rows; r++){
        for(int c=0; c < frame.pose_R.cols; c++) {
            cam.m[r][c] = frame.pose_R.at<double>(r, c);
        }
    }
    for(int c=0; c < frame.pose_t.cols; c++) {
        cam.t[c] = frame.pose_t.at<double>(c);
    }

    pba_cameras_.push_back(cam);

    cv::Mat descriptors;
    feature_detector_.detectComputeORB(frame, &image_feature.keypoints, &descriptors);
    image_feature.descriptors = descriptors.getUMat(cv::USAGE_DEFAULT);
    image_feature.img_idx = count_;
    image_feature.img_size = frame.image.size();

    features_.push_back(image_feature);

    if(features_.size() > max_frames_) {
        features_.erase(features_.begin());
    }

    //Do pairwise matching first
    (*matcher_)(features_, pairwise_matches_);
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