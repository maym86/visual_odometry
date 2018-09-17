

#ifndef VO_SFM_BUNDLE_ADJUSTMENT_H
#define VO_SFM_BUNDLE_ADJUSTMENT_H

#include <vector>
#include <unordered_map>

#include <opencv2/features2d.hpp>

#include <opencv2/stitching/detail/motion_estimators.hpp>
#include <opencv2/stitching/detail/matchers.hpp>

#include "src/visual_odometry/vo_frame.h"

#if __has_include("opencv2/cudafeatures2d.hpp")
#include "src/features/cuda/feature_detector.h"
#else

#include "src/features/feature_detector.h"

#endif


#include <cvsba/cvsba.h>

class BundleAdjustment {

public:

    void init(const cv::Mat &K, size_t max_frames);

    void addKeyFrame(const VOFrame &frame);

    int slove(cv::Mat *R, cv::Mat *t);

    void draw(float scale=1.0);
    void drawViz();
private:
    const float kMax3DDist = 200;

    void matcher();

    void setPBAPoints();

    FeatureDetector feature_detector_;



    std::vector< cv::Point3d > points_3d_;

    std::vector< std::vector< cv::Point2d > > points_img_;
    std::vector< std::vector< int > > visibility_;
    std::vector< cv::Mat > camera_matrix_;
    std::vector< cv::Mat > dist_coeffs_;
    std::vector< cv::Mat > R_;
    std::vector< cv::Mat > T_;


    std::vector<cv::detail::ImageFeatures> features_;

    std::vector<cv::detail::MatchesInfo> pairwise_matches_;

    cvsba::Sba sba_;

    std::vector<std::vector<std::vector<int>>> tracks_; //Vector with tracks starting at image index for vector
    cv::Point2f pp_;
    cv::Point2f focal_;

    cv::Mat K_;
    size_t max_frames_;
    int count_ = 0;

    void reprojectionInfo(const cv::Point2f &point, const cv::Point3f &point3d, const cv::Mat &proj_mat);

    void createTracks();

};


#endif //VO_SFM_BUNDLE_ADJUSTMENT_H
