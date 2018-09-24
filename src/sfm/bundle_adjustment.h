

#ifndef VO_SFM_BUNDLE_ADJUSTMENT_H
#define VO_SFM_BUNDLE_ADJUSTMENT_H

#include <vector>
#include <unordered_map>

#include <opencv2/features2d.hpp>
#include <opencv2/stitching/detail/matchers.hpp>

#include "src/visual_odometry/vo_frame.h"

#if __has_include("opencv2/cudafeatures2d.hpp")
#include "src/features/cuda/feature_detector.h"
#else
#include "src/features/feature_detector.h"
#endif


#include <opencv2/viz/types.hpp>
#include <opencv2/viz/widgets.hpp>
#include <opencv2/viz/viz3d.hpp>
#include <opencv2/viz/vizcore.hpp>


#include <gtsam/geometry/Point2.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/ProjectionFactor.h>
#include <gtsam/slam/GeneralSFMFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/DoglegOptimizer.h>
#include <gtsam/nonlinear/Values.h>


class BundleAdjustment {

public:

    void init(const cv::Mat &K, size_t max_frames);

    void addKeyFrame(const VOFrame &frame);

    int slove(cv::Mat *R, cv::Mat *t);
    void draw(float scale=1.0);

    void drawViz();
private:

    cv::viz::Viz3d viz_;

    const float kMin3DDist = 10;
    const float kMax3DDist = 200;
    const float kMax3DWidth = 40;

    void setPBAPoints();

    FeatureDetector feature_detector_;

    std::vector< cv::Point3d > points_3d_;

    std::vector< std::vector< cv::Point2f > > points_img_;
    std::vector< std::vector< int > > cameras_visible_;
    std::vector< cv::Mat > camera_matrix_;
    std::vector< cv::Mat > dist_coeffs_;
    std::vector< cv::Mat > R_;
    std::vector< cv::Mat > t_;


    std::vector<cv::detail::ImageFeatures> features_;

    std::vector<cv::detail::MatchesInfo> pairwise_matches_;

    std::vector<std::vector<int>> match_matrix_;

    cv::Point2f pp_;
    cv::Point2f focal_;

    cv::Mat K_;
    size_t max_frames_;
    int count_ = 0;

};


#endif //VO_SFM_BUNDLE_ADJUSTMENT_H
