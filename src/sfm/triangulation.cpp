
#include "triangulation.h"

#include <random>
#include <cv.hpp>
#include <glog/logging.h>

//Init random
std::random_device rd;
std::mt19937 rng(rd());

//http://www.morethantechnical.com/2012/01/04/simple-triangulation-with-opencv-from-harley-zisserman-w-code/
/**
 From "Triangulation", Hartley, R.I. and Sturm, P., Computer vision and image understanding, 1997
 */
cv::Mat LinearLSTriangulation(cv::Point3d u,       //homogenous image point (u,v,1)
                              cv::Matx34d P,       //camera 1 matrix
                              cv::Point3d u1,      //homogenous image point in 2nd camera
                              cv::Matx34d P1       //camera 2 matrix
) {
    //build matrix A for homogenous equation system Ax = 0
    //assume X = (x,y,z,1), for Linear-LS method
    //which turns it into a AX = B system, where A is 4x3, X is 3x1 and B is 4x1
    cv::Matx43d A(u.x * P(2, 0) - P(0, 0), u.x * P(2, 1) - P(0, 1), u.x * P(2, 2) - P(0, 2),
                  u.y * P(2, 0) - P(1, 0), u.y * P(2, 1) - P(1, 1), u.y * P(2, 2) - P(1, 2),
                  u1.x * P1(2, 0) - P1(0, 0), u1.x * P1(2, 1) - P1(0, 1), u1.x * P1(2, 2) - P1(0, 2),
                  u1.y * P1(2, 0) - P1(1, 0), u1.y * P1(2, 1) - P1(1, 1), u1.y * P1(2, 2) - P1(1, 2)
    );

    double data[4] = {-(u.x * P(2, 3) - P(0, 3)),
                      -(u.y * P(2, 3) - P(1, 3)),
                      -(u1.x * P1(2, 3) - P1(0, 3)),
                      -(u1.y * P1(2, 3) - P1(1, 3))};

    cv::Mat B(4, 1, CV_64F, data);
    cv::Mat X;
    solve(A, B, X, cv::DECOMP_SVD);

    return X;
}


cv::Mat_<double> IterativeLinearLSTriangulation(cv::Point3d u,    //homogenous image point (u,v,1)
                                                cv::Matx34d P,          //camera 1 matrix
                                                cv::Point3d u1,         //homogenous image point in 2nd camera
                                                cv::Matx34d P1          //camera 2 matrix
) {

    double EPSILON = 1e-6;
    double wi = 1, wi1 = 1;
    cv::Mat_<double> X(4, 1);
    for (int i = 0; i < 10; i++) { //Hartley suggests 10 iterations at most
        cv::Mat_<double> X_ = LinearLSTriangulation(u, P, u1, P1);
        X(0) = X_(0);
        X(1) = X_(1);
        X(2) = X_(2);
        X_(3) = 1.0;

        //recalculate weights
        double p2x = cv::Mat_<double>(cv::Mat_<double>(P).row(2) * X)(0);
        double p2x1 = cv::Mat_<double>(cv::Mat_<double>(P1).row(2) * X)(0);

        //breaking point
        if (fabsf(wi - p2x) <= EPSILON && fabsf(wi1 - p2x1) <= EPSILON) break;

        wi = p2x;
        wi1 = p2x1;

        //reweight equations and solve
        cv::Matx43d A((u.x * P(2, 0) - P(0, 0)) / wi, (u.x * P(2, 1) - P(0, 1)) / wi, (u.x * P(2, 2) - P(0, 2)) / wi,
                      (u.y * P(2, 0) - P(1, 0)) / wi, (u.y * P(2, 1) - P(1, 1)) / wi, (u.y * P(2, 2) - P(1, 2)) / wi,
                      (u1.x * P1(2, 0) - P1(0, 0)) / wi1, (u1.x * P1(2, 1) - P1(0, 1)) / wi1,
                      (u1.x * P1(2, 2) - P1(0, 2)) / wi1,
                      (u1.y * P1(2, 0) - P1(1, 0)) / wi1, (u1.y * P1(2, 1) - P1(1, 1)) / wi1,
                      (u1.y * P1(2, 2) - P1(1, 2)) / wi1
        );
        cv::Mat_<double> B = (cv::Mat_<double>(4, 1) << -(u.x * P(2, 3) - P(0, 3)) / wi,
                -(u.y * P(2, 3) - P(1, 3)) / wi,
                -(u1.x * P1(2, 3) - P1(0, 3)) / wi1,
                -(u1.y * P1(2, 3) - P1(1, 3)) / wi1
        );

        solve(A, B, X_, cv::DECOMP_SVD);
        X(0) = X_(0);
        X(1) = X_(1);
        X(2) = X_(2);
        X_(3) = 1.0;
    }
    return X;
}

//http://answers.opencv.org/question/118966/is-cvtriangulatepoints-returning-3d-points-in-world-coordinate-system/
std::vector<cv::Point3f>
triangulate(const std::vector<cv::Point2f> &points0, const std::vector<cv::Point2f> &points1, const cv::Mat &P0,
            const cv::Mat &P1) {
    std::vector<cv::Point3f> results;

    if (points0.size() == 0 || points1.size() == 0)
        return results;

    cv::Mat points_3d;
    cv::Mat_<float> p_mat0(2, static_cast<int>(points0.size()), CV_32FC1);
    cv::Mat_<float> p_mat1(2, static_cast<int>(points1.size()), CV_32FC1);

    for (int i = 0; i < p_mat0.cols; i++) {
        cv::Point3d p0(points0[i].x, points0[i].y, 1);
        cv::Point3d p1(points1[i].x, points1[i].y, 1);
        cv::Mat p3d = LinearLSTriangulation(p0, P0, p1, P1);

        auto p = cv::Point3f(static_cast<float >(p3d.at<double>(0)),
                             static_cast<float >(p3d.at<double>(1)),
                             static_cast<float >(p3d.at<double>(2)));

        results.push_back(p);

    }

    /*cv::triangulatePoints(P0, P1, p_mat0, p_mat1, points_3d);


    cv::Point3d c(P0.at<double>(0,3), P0.at<double>(1,3), P0.at<double>(2,3));
    LOG(INFO) << P0;
    double mean_dist = 0;
    for (int i = 0; i < points_3d.cols; i++) {
        auto p = cv::Point3d(points_3d.at<float>(0, i) / points_3d.at<float>(3, i),
                             points_3d.at<float>(1, i) / points_3d.at<float>(3, i),
                             points_3d.at<float>(2, i) / points_3d.at<float>(3, i));

        mean_dist += cv::norm(p - c);
        results.push_back(p);
    }

    LOG(INFO) << "Mean dist " << mean_dist / points_3d.cols;*/
    return results;
}


std::vector<cv::Point3f>
triangulate2(const std::vector<cv::Point2f> &points0, const std::vector<cv::Point2f> &points1, const cv::Mat &P0,
             const cv::Mat &P1) {
    std::vector<cv::Point3f> results;

    if (points0.size() == 0 || points1.size() == 0)
        return results;

    cv::Mat points_3d;
    cv::Mat_<float> p_mat0(2, static_cast<int>(points0.size()), CV_32FC1);
    cv::Mat_<float> p_mat1(2, static_cast<int>(points1.size()), CV_32FC1);


    for (int i = 0; i < p_mat0.cols; i++) {
        p_mat0.at<float>(0, i) = points0[i].x;
        p_mat0.at<float>(1, i) = points0[i].y;
        p_mat1.at<float>(0, i) = points1[i].x;
        p_mat1.at<float>(1, i) = points1[i].y;
    }

    cv::Mat P0_origin = cv::Mat::eye(3, 4, CV_64FC1);
    cv::Mat P1_origin = P1.clone();

    P1_origin.at<double>(0, 3) -= P0.at<double>(0, 3);
    P1_origin.at<double>(1, 3) -= P0.at<double>(1, 3);
    P1_origin.at<double>(2, 3) -= P0.at<double>(2, 3);

    LOG(INFO) << P1_origin;
    cv::triangulatePoints(P0_origin, P1_origin, p_mat0, p_mat1, points_3d);

    cv::Point3d c(P0.at<double>(0, 3), P0.at<double>(1, 3), P0.at<double>(2, 3));
    LOG(INFO) << P0;
    double mean_dist = 0;
    for (int i = 0; i < points_3d.cols; i++) {
        auto p = cv::Point3d(cv::Point3d(points_3d.at<float>(0, i) / points_3d.at<float>(3, i) + P0.at<double>(0, 3),
                                         points_3d.at<float>(1, i) / points_3d.at<float>(3, i) + P0.at<double>(1, 3),
                                         points_3d.at<float>(2, i) / points_3d.at<float>(3, i) + P0.at<double>(2, 3)));

        mean_dist += cv::norm(p - c);
        results.push_back(p);
    }

    LOG(INFO) << "Mean dist " << mean_dist / points_3d.cols;
    return results;

    return results;
}

void triangulateFrame(VOFrame *vo0, VOFrame *vo1) {
    if (!vo1->P.empty()) {
        cv::Mat P0 = cv::Mat::eye(3, 4, CV_64FC1);
        vo1->points_3d = triangulate(vo0->points, vo1->points, P0, vo1->P);
    }
}


float getScale(const VOFrame &vo0, const VOFrame &vo1, int min_points, int max_points) {

    if (vo0.points_3d.empty() || vo1.points_3d.empty()) {
        LOG(INFO) << "O point size";
        return 1;
    }

    //Pick random points in prev that match to two points in now;
    //TODO Maybe just use all the points???
    std::uniform_int_distribution<int> uni(0, static_cast<int>(vo1.points.size() - 1));
    std::vector<int> indices;
    int last = -1;
    for (int i = 0; i < max_points; i++) {
        for (int j = 0; j < 1000; j++) {
            int index = uni(rng);
            if (vo1.mask.at<bool>(index) && vo0.mask.at<bool>(vo0.tracked_index[index]) && index != last) {
                last = index;
                indices.push_back(index);
                break;
            }
        }
    }

    if (indices.empty()) {
        LOG(INFO) << "Empty indices: " << indices.size();
        return 1;
    }

    double vo0_sum = 0;
    double vo1_sum = 0;
    int count = 0;
    for (int i = 0; i < indices.size() - 1; i++) {
        int i0 = indices[i];
        int i1 = indices[i + 1];
        double n0 = cv::norm(vo0.points_3d[vo0.tracked_index[i0]] - vo0.points_3d[vo0.tracked_index[i1]]);
        double n1 = cv::norm(vo1.points_3d[i0] - vo1.points_3d[i1]);

        if (!std::isnan(n0) && !std::isnan(n1) && !std::isinf(n0) && !std::isinf(n1) && std::fabs(n1 - n0) < 200) {
            vo0_sum += n0;
            vo1_sum += n1;
            count++;
        }
    }

    if (count < min_points) {
        LOG(INFO) << "Counts less than min: " << count << " < " << min_points;
        return 1;
    }

    auto scale = static_cast<float>(vo1_sum / vo0_sum);

    if (std::isnan(scale) || std::isinf(scale) || scale == 0) {
        LOG(INFO) << "Scale invalid: " << scale;
        return 1;
    }

    if (scale > 5) { //TODO this is wrong - fix in a different way
        LOG(INFO) << "Scale is large: " << scale;
        //return 5; //TODO Arbitrary
    }

    return scale;
}

