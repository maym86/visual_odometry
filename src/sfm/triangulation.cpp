
#include "triangulation.h"

#include <cv.hpp>
#include <glog/logging.h>

void triangulate(VOFrame *vo0, VOFrame *vo1) {
    if (!vo1->P.empty()) {
        cv::Mat points_3d;
        cv::Mat_<double> p0(2, vo0->points.size(), CV_64FC1);
        cv::Mat_<double> p1(2, vo1->points.size(), CV_64FC1);

        for (int i = 0; i < p0.cols; i++) {
            p0.at<double>(0, i) = vo0->points[i].x;
            p0.at<double>(1, i) = vo0->points[i].y;
            p1.at<double>(0, i) = vo1->points[i].x;
            p1.at<double>(1, i) = vo1->points[i].y;
        }

        cv::Mat P = cv::Mat::eye(3, 4, CV_64FC1);
        cv::triangulatePoints(P, vo1->P, p0, p1, points_3d); //TODO this is relative to previous frame center
        vo1->points_3d.clear();

        for (int i = 0; i < points_3d.cols; i++) {
            vo1->points_3d.push_back(cv::Point3d(points_3d.at<double>(0, i) / points_3d.at<double>(3, i),
                                                 points_3d.at<double>(1, i) / points_3d.at<double>(3, i),
                                                 points_3d.at<double>(2, i) / points_3d.at<double>(3, i)));
        }
    }
}


double getScale(const VOFrame &vo0, const VOFrame &vo1, int min_points, int max_points) {

    if (vo0.points_3d.size() == 0 || vo1.points_3d.size() == 0) {
        LOG(INFO) << "O point size";
        return 1;
    }

    //Pick random points in prev that match to two points in now;
    std::vector<int> indices;
    int last = -1;
    for (int i = 0; i < max_points; i++) {
        for (int j = 0; j < 1000; j++) {
            int index = rand() % static_cast<int>(vo1.points_3d.size());
            if (vo1.mask.at<bool>(index) && vo0.mask.at<bool>(vo0.tracked_index[index]) && index != last) {
                last = index;
                indices.push_back(index);
                break;
            }
        }
    }

    if (indices.empty()) {
        LOG(INFO) << "Empty indices " << indices.size();
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

    double scale = vo1_sum / vo0_sum;
    if (std::isnan(scale) || std::isnan(scale) || scale == 0) {
        LOG(INFO) << "NaN or inf";
        return 1;
    }

    return scale;
}

