
#include <gtest/gtest.h>

#include "src/sfm/triangulation.h"
#include "src/utils/draw.h"

#include <glog/logging.h>

double R1[12] = {0.9999947416807112, 0.0008410819119621252, -0.003131962985946667,
        -0.0008478311433543918, 0.9999973201938741, -0.002154248690917288,
        0.003130142693285286, 0.002156892738949201, 0.9999927749841158};

double t1[12] = { -0.04964047267880936, 0.01379588283514795, 0.9986718665753148};


cv::Point2d pp(607.1928, 185.2157);
cv::Point2d focal(718.856, 718.856);

TEST(TriangulationTest, Passes) {

    cv::Mat K = cv::Mat::eye(3,3, CV_64FC1);

    K.at<double>(0,0) = focal.x;
    K.at<double>(1,1) = focal.y;
    K.at<double>(0,2) = pp.x;
    K.at<double>(1,2) = pp.y;


    VOFrame vo0;
    VOFrame vo1;

    vo0.local_P = cv::Mat::eye(3, 4, CV_64FC1);

    cv::Mat R = cv::Mat(3, 3, CV_64F, R1);
    cv::Mat t = cv::Mat(3, 1, CV_64F, t1);

    hconcat(R.t(), -t, vo1.local_P);

    cv::FileStorage store("../src/sfm/test/test_data/test_points.bin", cv::FileStorage::READ);
    cv::FileNode n0 = store["p0"];
    cv::read(n0,vo0.points);
    cv::FileNode n1 = store["p1"];
    cv::read(n1,vo1.points);
    store.release();

    triangulateFrame(pp, focal, vo0, &vo1);

    draw3D(vo1);
    cv::waitKey(0);
}



