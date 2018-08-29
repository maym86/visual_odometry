

#include "bundle_adjustment.h"


std::vector<cv::CameraParams> cameras;
std::vector<cv::MatchesInfo> pairwise_matches;
std::vector<cv::ImageFeatures> features(num_images);

BundleAdjustment::BundleAdjustmemt(){
    adjuster_ = makePtr<BundleAdjusterReproj>();
}

void BundleAdjustment::slove() {
    if (!(*adjuster)(features, pairwise_matches, cameras)) {
        LOG(INFO) << "Camera parameters adjusting failed.";
        return -1;
    }
}