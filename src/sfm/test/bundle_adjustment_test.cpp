
#include <gtest/gtest.h>

#include "src/sfm/bundle_adjustment.h"

TEST(BundleAdjustmentTest, Passes) {
    BundleAdjustment ba;

    ba.init(600 , cv::Point2f(10,10) , 10);

    //TODO load the ba data from pba and make sure this works


}
