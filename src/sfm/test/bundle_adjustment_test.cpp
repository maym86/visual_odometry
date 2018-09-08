
#include <gtest/gtest.h>

#include "src/sfm/bundle_adjustment.h"

#include <glog/logging.h>


double r_data[9] = {9.993513e-01, 1.860866e-02, -3.083487e-02,
                    -1.887662e-02, 9.997863e-01, -8.421873e-03,
                    3.067156e-02, 8.998467e-03, 9.994890e-01};

double t_data[3] = {-5.370000e-01, 4.822061e-03, -1.252488e-02};

TEST(BundleAdjustmentTest, Passes) {
    BundleAdjustment ba;

    VOFrame vo0;
    VOFrame vo1;

    vo0.image = cv::imread("../src/sfm/test/test_data/000000.png");
    vo1.image = cv::imread("../src/sfm/test/test_data/000001.png");

    EXPECT_GT(vo0.image.rows, 0);
    EXPECT_GT(vo1.image.rows, 0);

    vo0.pose_R = cv::Mat::eye(3, 3, CV_64FC1);
    vo0.pose_t = cv::Mat::zeros(3, 1, CV_64FC1);

    hconcat(vo0.pose_R, vo0.pose_t, vo0.pose);

    vo1.pose_R = cv::Mat(3, 3, CV_64FC1, r_data);
    vo1.pose_t = cv::Mat(3, 1, CV_64FC1, t_data);

    hconcat(vo1.pose_R, vo1.pose_t, vo1.pose);

    ba.init(cv::Point2f(718.856,718.856), cv::Point2f(607.193, 185.216) , 2);

    ba.addKeyFrame(vo0);
    ba.addKeyFrame(vo1);

    cv::Mat R, t;
    ba.slove(&R, &t);

    LOG(INFO) << vo1.pose_t;
    LOG(INFO) << t;

    double dist = cv::norm(vo1.pose_t - t);
    LOG(INFO) << dist;
    EXPECT_NEAR(dist, 0.03, 0.02); //TODO very large allowed error

}

//TODO this is the bug
TEST(BundleAdjustmentOffsetTest, Passes) {
    BundleAdjustment ba;

    VOFrame vo0;
    VOFrame vo1;

    vo0.image = cv::imread("../src/sfm/test/test_data/image_0_000000.png");
    vo1.image = cv::imread("../src/sfm/test/test_data/image_1_000000.png");

    EXPECT_GT(vo0.image.rows, 0);
    EXPECT_GT(vo1.image.rows, 0);

    vo0.pose_R = cv::Mat::eye(3, 3, CV_64FC1);
    vo0.pose_t = cv::Mat::zeros(3, 1, CV_64FC1);

    vo0.pose_t.at<double>(0) += 100;
    vo0.pose_t.at<double>(1) += 100;
    vo0.pose_t.at<double>(2) += 100;

    hconcat(vo0.pose_R, vo0.pose_t, vo0.pose);

    vo1.pose_R = cv::Mat(3, 3, CV_64FC1, r_data);
    vo1.pose_t = cv::Mat(3, 1, CV_64FC1, t_data);

    vo1.pose_t.at<double>(0) += 100;
    vo1.pose_t.at<double>(1) += 100;
    vo1.pose_t.at<double>(2) += 100;


    hconcat(vo1.pose_R, vo1.pose_t, vo1.pose);

    ba.init(cv::Point2f(718.856,718.856), cv::Point2f(607.193, 185.216) , 2);

    ba.addKeyFrame(vo0);
    ba.addKeyFrame(vo1);

    cv::Mat R, t;
    ba.slove(&R, &t);

    LOG(INFO) << vo0.pose_t;
    LOG(INFO) << vo1.pose_t;
    LOG(INFO) << t;

    double dist = cv::norm(vo1.pose_t - t);
    LOG(INFO) << dist;
    EXPECT_NEAR(dist, 0.03, 0.02); //TODO very large allowed error

}
