
#include "feature_tracker.h"
#include "src/features/utils.h"

FeatureTracker::FeatureTracker(){
     optical_flow_ = cv::SparsePyrLKOpticalFlow::create();
}


void FeatureTracker::trackPoints(VOFrame *vo0, VOFrame *vo1) {
    std::vector<unsigned char> status;

    optical_flow_->calc(vo0->image, vo1->image, vo0->points, vo1->points, status);

    //Remove bad points
    removePoints(vo0, vo1, &status);
}
