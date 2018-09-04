
#include <gtest/gtest.h>

#include "src/sfm/bundle_adjustment.h"
#include <boost/filesystem.hpp>
#include <glog/logging.h>

TEST(BundleAdjustmentTest, Passes) {
    BundleAdjustment ba;

    VOFrame vo0;
    VOFrame vo1;

    boost::filesystem::path full_path(boost::filesystem::current_path());
    std::cout << "Current path is : " << full_path << std::endl;

    vo0.image = cv::imread("../src/sfm/test/test_data/image_0_000000.png");
    vo1.image = cv::imread("../src/sfm/test/test_data/image_1_000000.png");

    EXPECT_GT(vo0.image.rows, 0);
    EXPECT_GT(vo1.image.rows, 0);

    vo0.pose_R = cv::Mat::eye(3, 3, CV_64FC1);
    vo0.pose_t = cv::Mat::zeros(3, 1, CV_64FC1);
    hconcat(vo0.pose_R, vo0.pose_t, vo0.pose);

    vo1.pose_R = cv::Mat::eye(3, 3, CV_64FC1);

    float t_data[3] = {2.573699e-16, -1.059758e-16, 1.614870e-16};
    vo1.pose_t = cv::Mat(3, 1, CV_64FC1, t_data);
    hconcat(vo1.pose_R, vo1.pose_t, vo1.pose);

    ba.init(718.856 , cv::Point2f(607.193, 185.216) , 2);

    ba.addKeyFrame(vo0);
    ba.addKeyFrame(vo1);

    cv::Mat R, t;
    ba.slove(&R, &t);

    LOG(INFO) << R << " " << t;

}
