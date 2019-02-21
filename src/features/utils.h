
#ifndef VO_FETAURES_UTILS_H
#define VO_FETAURES_UTILS_H

#include <vector>
#include <numeric>
#include "src/visual_odometry/vo_frame.h"

int removePoints(VOFrame *vo0, VOFrame *vo1, std::vector<unsigned char> *status) {
    //Remove bad points
    std::vector<float> dists;
    for (int i = 0; i < status->size(); i++) {
        dists.push_back(cv::norm(vo0->points[i] - vo1->points[i]));
    }

    std::sort(dists.begin(), dists.end());
    double sum = std::accumulate(dists.begin(), dists.end(), 0.0);
    double mean = sum / dists.size();
    double sq_sum = std::inner_product(dists.begin(), dists.end(), dists.begin(), 0.0);
    double stdev = std::sqrt(sq_sum / dists.size() - mean * mean);

    //Convert Mat to vector so we can remove some of the data - //TODO use something other than vector??
    std::vector<uchar> mask(vo0->mask.rows * vo0->mask.cols);
    if (vo0->mask.isContinuous()) {
        mask.assign(vo0->mask.datastart, vo0->mask.dataend);
    } else {
        for (int i = 0; i < vo0->mask.rows; ++i) {
            mask.insert(mask.end(), vo0->mask.ptr<uchar>(i), vo0->mask.ptr<uchar>(i) + vo0->mask.cols);
        }
    }

    for (int i = status->size() - 1; i >= 0; --i) {
        cv::Point2f pt = vo1->points[i];

        float dist = cv::norm(vo0->points[i] - vo1->points[i]);

        if ((*status)[i] == 0 || pt.x < 0 || pt.y < 0 || abs(dist - mean) > stdev * 2) {
            if (pt.x < 0 || pt.y < 0) {
                (*status)[i] = 0;
            }
            vo0->points.erase(vo0->points.begin() + i);
            dists.erase(dists.begin() + i);
            if (i < vo0->points_3d.size()) {
                vo0->points_3d.erase(vo0->points_3d.begin() + i);
            }

            if (i < mask.size()) {
                mask.erase(mask.begin() + i);
            }

            vo1->points.erase(vo1->points.begin() + i);
        }
    }

    vo0->mask = cv::Mat(mask.size(), 1, vo0->mask.type(), mask.data());

    return dists[dists.size() / 2];
};



#endif //VO_FETAURES_UTILS_H
