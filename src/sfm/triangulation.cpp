
#include "triangulation.h"

#include <random>
#include <cv.hpp>

#include <opencv2/sfm.hpp>

#include <glog/logging.h>

//Init random
std::random_device rd;
std::mt19937 rng(rd());

//https://github.com/nghiaho12/SFM_example/blob/master/src/main.cpp

//http://nghiaho.com/

//https://gist.github.com/cashiwamochi/8ac3f8bab9bf00e247a01f63075fedeb


//http://answers.opencv.org/question/118966/is-cvtriangulatepoints-returning-3d-points-in-world-coordinate-system/

std::vector<cv::Point3d> points4dToVec(const cv::Mat &points4d){
    std::vector<cv::Point3d> results;
    for (int i = 0; i < points4d.cols; i++) {
        results.emplace_back(cv::Point3d(points4d.at<double>(0, i) / points4d.at<double>(3, i),
                                         points4d.at<double>(1, i) / points4d.at<double>(3, i),
                                         points4d.at<double>(2, i) / points4d.at<double>(3, i)));

    }

    return results;
}

std::vector<cv::Point3d> points3dToVec(const cv::Mat &points3d){
    std::vector<cv::Point3d> results;
    for (int i = 0; i < points3d.cols; i++) {
        results.emplace_back(cv::Point3d(points3d.at<double>(0, i),
                                         points3d.at<double>(1, i),
                                         points3d.at<double>(2, i)));

    }

    return results;
}

//http://answers.opencv.org/question/171898/sfm-triangulatepoints-input-array-assertion-failed/
std::vector<cv::Point3d> triangulate(const cv::Point2f &pp, const double focal, const std::vector<cv::Point2f> &points0, const std::vector<cv::Point2f> &points1, const cv::Mat &P0, const cv::Mat &P1){
    std::vector<cv::Point3d> results;

    if(points0.size() == 0 || points1.size() == 0) {
        return results;
    }

    cv::Mat p_mat0(2, static_cast<int>(points0.size()), CV_64FC1);
    cv::Mat p_mat1(2, static_cast<int>(points1.size()), CV_64FC1);

    for (int i = 0; i < p_mat0.cols; i++) {

        LOG(INFO) << points0[i] << points1[i];

        p_mat0.at<double>(0, i) = (points0[i].x  - pp.x) / focal;
        p_mat0.at<double>(1, i) = (points0[i].y  - pp.y) / focal;
        p_mat1.at<double>(0, i) = (points1[i].x - pp.x) / focal;
        p_mat1.at<double>(1, i) = (points1[i].y  - pp.y) / focal;
    }

    std::vector<cv::Mat> sfmPoints2d;
    sfmPoints2d.push_back(p_mat0);
    sfmPoints2d.push_back(p_mat1);

    std::vector<cv::Mat> sfmProjMats;
    sfmProjMats.push_back(P0);
    sfmProjMats.push_back(P1);
    cv::Mat points3d;

    cv::sfm::triangulatePoints(sfmPoints2d, sfmProjMats, points3d);

    results = points3dToVec(points3d);
    return results;
}
/*
std::vector<cv::Point3d> triangulate(const cv::Point2f &pp, const double focal, const std::vector<cv::Point2f> &points0, const std::vector<cv::Point2f> &points1, const cv::Mat &P0, const cv::Mat &P1){
    std::vector<cv::Point3d> results;

    if(points0.size() == 0 || points1.size() == 0) {
        return results;
    }

    cv::Mat points_4d;

    cv::Mat p_mat0(2, static_cast<int>(points0.size()), CV_64FC1);
    cv::Mat p_mat1(2, static_cast<int>(points1.size()), CV_64FC1);

    for (int i = 0; i < p_mat0.cols; i++) {
        p_mat0.at<double>(0, i) = (points0[i].x  - pp.x) / focal;
        p_mat0.at<double>(1, i) = (points0[i].y  - pp.y) / focal;
        p_mat1.at<double>(0, i) = (points1[i].x - pp.x) / focal;
        p_mat1.at<double>(1, i) = (points1[i].y  - pp.y) / focal;
    }

    cv::triangulatePoints(P0, P1, p_mat0, p_mat1, points_4d);

    results = points4dToVec(points_4d);
    return results;
}*/

void triangulateFrame(const cv::Point2f &pp, const double focal, const VOFrame &frame0, VOFrame *frame1) {
    if (!frame1->local_P.empty()) {
        frame1->points_3d = triangulate(pp, focal, frame0.points, frame1->points, cv::Mat::eye(3, 4, CV_64FC1), frame1->local_P);
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

    if(scale > 10){ //TODO this is wrong - fix in a different way
        LOG(INFO) << "Scale is large: " << scale;
        return 10; //TODO Arbitrary
    }

    return scale;
}