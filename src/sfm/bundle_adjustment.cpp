
#include "bundle_adjustment.h"

#include <glog/logging.h>

#include "triangulation.h"

BundleAdjustment::BundleAdjustment() : pba_(ParallelBA::DeviceT::PBA_CPU_DOUBLE) {

}

void BundleAdjustment::init(float focal, const cv::Point2d &pp, size_t max_frames) {

    char *argv[] = {"-lmi<100>", "-v", "1"};
    int argc = sizeof(argv) / sizeof(char *);

    pba_.ParseParam(argc, argv);

    pba_.SetFixedIntrinsics(true);

    matcher_ = cv::makePtr<cv::detail::BestOf2NearestMatcher>(true);
    max_frames_ = max_frames;

    pp_ = pp;
    focal_ = focal;
}

void BundleAdjustment::addKeyFrame(const VOFrame &frame) {

    CameraT cam;
    cam.f = focal_;

    cam.SetTranslation(reinterpret_cast<double *>(frame.pose_t.data));
    cam.SetMatrixRotation(reinterpret_cast<double *>(frame.pose_R.data));

    if (count_ == 0) {
        cam.SetConstantCamera();
    }

    pba_cameras_.push_back(cam);

    poses_.push_back(frame.pose.clone());

    cv::detail::ImageFeatures image_feature;
    cv::Mat descriptors;

    feature_detector_.detectComputeORB(frame, &image_feature.keypoints, &descriptors);
    image_feature.descriptors = descriptors.getUMat(cv::USAGE_DEFAULT);
    image_feature.img_idx = count_++;
    image_feature.img_size = frame.image.size();

    features_.push_back(image_feature);

    if (features_.size() > max_frames_) {
        features_.erase(features_.begin());
        poses_.erase(poses_.begin());
        pba_cameras_.erase(pba_cameras_.begin());
    }

    (*matcher_)(features_, pairwise_matches_);

    setPBAData(features_, pairwise_matches_, poses_, &pba_3d_points_, &pba_image_points_,
               &pba_2d3d_idx_, &pba_cam_idx_);

    pba_.SetCameraData(pba_cameras_.size(), &pba_cameras_[0]); //set camera parameters
    pba_.SetPointData(pba_3d_points_.size(), &pba_3d_points_[0]); //set 3D point data

    LOG(INFO) << pba_3d_points_.size() << " " << pba_image_points_.size();
    //set the projections
    pba_.SetProjection(pba_image_points_.size(), &pba_image_points_[0], &pba_2d3d_idx_[0], &pba_cam_idx_[0]);
    pba_.SetNextBundleMode(ParallelBA::BUNDLE_ONLY_MOTION); //Solving for motion only
}

// TODO This is wrong --- Use this as a template https://github.com/lab-x/SFM/blob/61bd10ab3f70a564b6c1971eaebc37211557ea78/SparseCloud.cpp
// Or this https://github.com/Zponpon/AR/blob/5d042ba18c1499bdb2ec8d5e5fae544e45c5bd91/PlanarAR/SFMUtil.cpp
// https://stackoverflow.com/questions/46875340/parallel-bundle-adjustment-pba
void BundleAdjustment::setPBAData(const std::vector<cv::detail::ImageFeatures> &features,
                                  const std::vector<cv::detail::MatchesInfo> &pairwise_matches,
                                  const std::vector<cv::Mat> &poses,
                                  std::vector<Point3D> *pba_3d_points,
                                  std::vector<Point2D> *pba_image_points, std::vector<int> *pba_2d3d_idx,
                                  std::vector<int> *pba_cam_idx) {

    pba_3d_points->clear();
    pba_image_points->clear();
    pba_cam_idx->clear();
    pba_2d3d_idx->clear();

    for (const auto &pwm : pairwise_matches) {
        int idx_cam0 = pwm.src_img_idx;
        int idx_cam1 = pwm.dst_img_idx;

        if (idx_cam0 != -1 && idx_cam1 != -1 && idx_cam0 != idx_cam1 && pwm.confidence > 0 && idx_cam0 < idx_cam1) { //TODO experiment with confidence thresh

            std::vector<cv::Point2f> points0;
            std::vector<cv::Point2f> points1;

            for (const auto &match : pwm.matches) {
                points0.push_back(features[idx_cam0].keypoints[match.queryIdx].pt);
                points1.push_back(features[idx_cam1].keypoints[match.trainIdx].pt);
            }

            std::vector<cv::Point3f> points3d = triangulate(points0, points1, poses[idx_cam0], poses[idx_cam1]);

            for (int j = 0; j < points3d.size(); j++) {

                if(j == 50) {
                    LOG(INFO) << points0[j] << " " << points1[j] << " " << points3d[j];
                }

                pba_3d_points->push_back(Point3D{points3d[j].x, points3d[j].y, points3d[j].z});

                //First 2dpoint that relates to 3d point
                pba_image_points->push_back(Point2D{points0[j].x, points0[j].y});
                pba_cam_idx->push_back(idx_cam0);
                pba_2d3d_idx->push_back(static_cast<int>(pba_3d_points->size() - 1));

                //Second 2dpoint that relates to 3D point
                pba_image_points->push_back(Point2D{points1[j].x, points1[j].y});
                pba_cam_idx->push_back(idx_cam1);
                pba_2d3d_idx->push_back(static_cast<int>(pba_3d_points->size() - 1));
            }
        }
    }
}

//TODO use full history rather than just updating the newest point
int BundleAdjustment::slove(cv::Mat *R, cv::Mat *t) {

    if (!pba_.RunBundleAdjustment()) {
        LOG(INFO) << "Camera parameters adjusting failed.";
        return 1;
    }

    const auto &last_cam = pba_cameras_[pba_cameras_.size() - 1];

    *R = cv::Mat::eye(3, 3, CV_64FC1);
    *t = cv::Mat::zeros(3, 1, CV_64FC1);

    last_cam.GetTranslation(reinterpret_cast<double *>(t->data));
    last_cam.GetMatrixRotation(reinterpret_cast<double *>(R->data));

    return 0;
}
