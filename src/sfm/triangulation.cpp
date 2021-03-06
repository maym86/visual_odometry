
#include "triangulation.h"

#include <random>
#include <cv.hpp>

#include <glog/logging.h>

//Init random
std::random_device rd;
std::mt19937 rng(rd());

std::vector<cv::Point3d> points4DToVec(const cv::Mat &points4d) {
    std::vector<cv::Point3d> results;

    for (int i = 0; i < points4d.cols; i++) {
        results.emplace_back(cv::Point3d(points4d.at<double>(0, i) / points4d.at<double>(3, i),
                                         points4d.at<double>(1, i) / points4d.at<double>(3, i),
                                         points4d.at<double>(2, i) / points4d.at<double>(3, i)));
    }
    return results;
}

std::vector<cv::Point3d> points3DToVec(const cv::Mat &points3d) {
    std::vector<cv::Point3d> results;
    for (int i = 0; i < points3d.cols; i++) {
        results.emplace_back(cv::Point3d(points3d.at<double>(0, i),
                                         points3d.at<double>(1, i),
                                         points3d.at<double>(2, i)));
    }
    return results;
}

cv::Mat getProjectionMatrix(const cv::Mat &K, const cv::Mat &pose){

    cv::Mat R = pose(cv::Range(0, 3), cv::Range(0, 3));
    cv::Mat t = pose(cv::Range(0, 3), cv::Range(3, 4));

    cv::Mat P(3, 4, CV_64F);

    P(cv::Range(0, 3), cv::Range(0, 3)) = R.t();
    P(cv::Range(0, 3), cv::Range(3, 4)) = -R.t()*t;
    return K * P;
}

std::vector<cv::Point3d> triangulate(const std::vector<cv::Point2f> &points0, const std::vector<cv::Point2f> &points1,
        const cv::Mat &P0, const cv::Mat &P1) {

    if (points0.size() == 0 || points1.size() == 0) {
        return std::vector<cv::Point3d>();
    }

    cv::Mat p_mat0(2, static_cast<int>(points0.size()), CV_64F);
    cv::Mat p_mat1(2, static_cast<int>(points1.size()), CV_64F);

    for (int i = 0; i < p_mat0.cols; i++) {
        p_mat0.at<double>(0, i) = points0[i].x;
        p_mat0.at<double>(1, i) = points0[i].y;
        p_mat1.at<double>(0, i) = points1[i].x;
        p_mat1.at<double>(1, i) = points1[i].y;
    }

    cv::Mat points_4d;
    cv::triangulatePoints(P0, P1, p_mat0, p_mat1, points_4d);
    return points4DToVec(points_4d);
}

float getScale(const VOFrame &frame0, const VOFrame &frame1, size_t min_points, size_t max_points, float max_3d_dist) {
    if (frame0.points_3d.empty() || frame1.points_3d.empty()) {
        LOG(INFO) << "O point size";
        return 1;
    }
    //Pick random points in prev that match to two points in now;
    LOG(INFO) << frame0.points_3d.size();

    std::uniform_int_distribution<int> uni(0, static_cast<int>(frame0.points_3d.size() - 1));
    std::vector<int> indices;

    int last = -1;
    for (int i = 0; i < max_points; i++) {
        for (int j = 0; j < 10; j++) {
            int idx = uni(rng);
            if (frame1.mask.at<bool>(idx) && frame0.mask.at<bool>(idx) &&
                frame1.points_3d[idx].z > 0 && cv::norm(frame1.points_3d[idx]) < max_3d_dist &&
                frame0.points_3d[idx].z > 0 && cv::norm(frame0.points_3d[idx]) < max_3d_dist && idx != last) {
                last = idx;
                indices.push_back(idx);
                break;
            }
        }
    }

    if (indices.empty()) {
        LOG(INFO) << "Empty indices: " << indices.size();
        return 1;
    }

    std::vector<double> scales;
    for (int i = 0; i < indices.size() - 1; i++) {
        int i0 = indices[i];
        int i1 = indices[i + 1];
        double n0 = cv::norm(frame0.points_3d[i0] - frame0.points_3d[i1]);
        double n1 = cv::norm(frame1.points_3d[i0] - frame1.points_3d[i1]);
        scales.push_back(n0 / n1);
    }

    if (scales.size() < min_points) {
        LOG(INFO) << "Counts less than min: " << scales.size() << " < " << min_points;
        return 1;
    }

    std::sort(scales.begin(), scales.end());
    auto scale = scales[scales.size() / 2];  //static_cast<float>(vo0_sum / vo1_sum);

    if (std::isnan(scale) || std::isinf(scale) || scale == 0) {
        LOG(INFO) << "Scale invalid: " << scale;
        return 1;
    }

    if (scale > 5) {
        LOG(WARNING) << "Scale is too large: " << scale;
        return 1;  //TODO figure out why - Likely Mismatches?
    }

    return scale;
}