
#include "triangulation.h"

#include <random>
#include <cv.hpp>
#include <glog/logging.h>

//Init random
std::random_device rd;
std::mt19937 rng(rd());

//http://answers.opencv.org/question/118966/is-cvtriangulatepoints-returning-3d-points-in-world-coordinate-system/
std::vector<cv::Point3f> triangulate(const std::vector<cv::Point2f> &points0, const std::vector<cv::Point2f> &points1, const cv::Mat &P0, const cv::Mat &P1){
    std::vector<cv::Point3f> results;

    if(points0.size() == 0 || points1.size() == 0)
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

    cv::triangulatePoints(P0, P1, p_mat0, p_mat1, points_3d);

    for (int i = 0; i < points_3d.cols; i++) {
        results.push_back(cv::Point3d(points_3d.at<float>(0, i) / points_3d.at<float>(3, i),
                                      points_3d.at<float>(1, i) / points_3d.at<float>(3, i),
                                      points_3d.at<float>(2, i) / points_3d.at<float>(3, i)));
    }
    return results;
}

/*
std::vector<cv::Point3f> triangulate(const std::vector<cv::Point2f> &points0, const std::vector<cv::Point2f> &points1, const cv::Mat &P0, const cv::Mat &P1){
    std::vector<cv::Point3f> results;

    if(points0.size() == 0 || points1.size() == 0)
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

    P1_origin.at<double>(0,3) -= P0.at<double>(0,3);
    P1_origin.at<double>(1,3) -= P0.at<double>(1,3);
    P1_origin.at<double>(2,3) -= P0.at<double>(2,3);

    cv::triangulatePoints(P0_origin, P1_origin, p_mat0, p_mat1, points_3d);

    for (int i = 0; i < points_3d.cols; i++) {
        results.push_back(cv::Point3d(points_3d.at<float>(0, i) / points_3d.at<float>(3, i) + P0.at<double>(0,3),
                                      points_3d.at<float>(1, i) / points_3d.at<float>(3, i) + P0.at<double>(1,3),
                                      points_3d.at<float>(2, i) / points_3d.at<float>(3, i) + P0.at<double>(2,3)));
    }
    return results;
}*/

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

    if(scale > 5){ //TODO this is wrong - fix in a different way
        LOG(INFO) << "Scale is large: " << scale;
        return 5; //TODO Arbitrary
    }

    return scale;
}