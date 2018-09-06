
#include <gtest/gtest.h>

#include "src/sfm/triangulation.h"

#include <glog/logging.h>

double P1[12] = {0.999994351093655, 0.00216084437549141, -0.002574593630994021, -0.08315501607210679,
        -0.002166127328107902, 0.9999955507641086, -0.002050937440673792, 0.02514760517749096,
        0.002570150419386047, 0.002056502752743189, 0.9999945825469503, 0.9962192736821971};

cv::Point2d pp(607.1928, 185.2157);
double focal = 718.856;

TEST(TriangulationTest, Passes) {

    VOFrame vo0;
    VOFrame vo1;

    vo0.local_P = cv::Mat::eye(3, 4, CV_64FC1);
    vo1.local_P = cv::Mat(3, 4, CV_64FC1, P1);

    cv::FileStorage store("../src/sfm/test/test_data/test_points.bin", cv::FileStorage::READ);
    cv::FileNode n0 = store["p0"];
    cv::read(n0,vo0.points);
    cv::FileNode n1 = store["p1"];
    cv::read(n1,vo1.points);
    store.release();

    triangulateFrame(pp, focal, vo0, &vo1);

    cv::Mat drawXY(800, 800, CV_8UC3, cv::Scalar(0, 0, 0));

    cv::line(drawXY, cv::Point(drawXY.cols / 2, 0), cv::Point(drawXY.cols / 2, drawXY.rows), cv::Scalar(0, 0, 255));
    cv::line(drawXY, cv::Point(0, drawXY.rows / 2), cv::Point(drawXY.cols, drawXY.rows / 2), cv::Scalar(0, 0, 255));


    cv::Point2d draw_pos = cv::Point2d(P1[3] + drawXY.cols / 2, P1[7]  + drawXY.rows / 2);

    cv::circle(drawXY, draw_pos, 3, cv::Scalar(255, 0, 0), 3);



    for (int j = 0; j < vo1.points_3d.size(); j++) {
        cv::Point2d draw_pos = cv::Point2d(vo1.points_3d[j].x + drawXY.cols / 2,
                                           vo1.points_3d[j].y + drawXY.rows / 2);

        cv::circle(drawXY, draw_pos, 1, cv::Scalar(0, 255, 0), 1);
    }

    cv::imshow("p3d", drawXY);
    cv::waitKey(0);
}

