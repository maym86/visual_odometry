

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


#include "pba/src/pba/pba.h"

class BundleAdjustment {

public:
    BundleAdjustment();

    void init(const cv::Point2f &focal, const cv::Point2f &pp, size_t max_frames);

    void addKeyFrame(const VOFrame &frame);

    int slove(cv::Mat *R, cv::Mat *t);

    void draw(float scale=1.0);
private:
    const float kMax3DDist = 200;

    void matcher();

    void setPBAPoints();

    FeatureDetector feature_detector_;

    cv::Ptr<cv::detail::FeaturesMatcher> matcher_;

    std::vector<CameraT> pba_cameras_;    //camera (input/ouput)
    std::vector<cv::detail::ImageFeatures> features_;

    ParallelBA pba_;

    std::vector<cv::detail::MatchesInfo> pairwise_matches_;
    std::vector<Point3D> pba_3d_points_;     //3D point(iput/output)
    std::vector<Point2D> pba_image_points_;   //measurment/projection vector
    std::vector<int> pba_2d3d_idx_, pba_cam_idx_;  //index of camera/point for each projection


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
