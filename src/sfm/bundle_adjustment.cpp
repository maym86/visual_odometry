
#include "bundle_adjustment.h"

#include <glog/logging.h>

#include "triangulation.h"

BundleAdjustment::BundleAdjustment() : pba_(ParallelBA::DeviceT::PBA_CPU_DOUBLE) {

}

void BundleAdjustment::init(const cv::Point2f &focal, const cv::Point2f &pp, size_t max_frames) {

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
    cam.f = focal_.x;

    cam.SetTranslation(reinterpret_cast<double *>(frame.pose_t.data));
    cam.SetMatrixRotation(reinterpret_cast<double *>(frame.pose_R.data));

    if (count_ == 0) {
        cam.SetConstantCamera();
    }

    pba_cameras_.push_back(cam);

    projection_matrices_.push_back(frame.pose);

    cv::detail::ImageFeatures image_feature;
    cv::Mat descriptors;

    feature_detector_.detectComputeORB(frame, &image_feature.keypoints, &descriptors);
    image_feature.descriptors = descriptors.getUMat(cv::USAGE_DEFAULT);
    image_feature.img_idx = count_++;
    image_feature.img_size = frame.image.size();

    features_.push_back(image_feature);

    if (features_.size() > max_frames_) {
        features_.erase(features_.begin());
        projection_matrices_.erase(projection_matrices_.begin());
        pba_cameras_.erase(pba_cameras_.begin());
    }

    (*matcher_)(features_, pairwise_matches_);

    setPBAData( &pba_3d_points_, &pba_image_points_, &pba_2d3d_idx_, &pba_cam_idx_);
    if(pba_3d_points_.size() > 0 && pba_image_points_.size()>0) {

        pba_.SetCameraData(pba_cameras_.size(), &pba_cameras_[0]); //set camera parameters
        pba_.SetPointData(pba_3d_points_.size(), &pba_3d_points_[0]); //set 3D point data

        //set the projections
        pba_.SetProjection(pba_image_points_.size(), &pba_image_points_[0], &pba_2d3d_idx_[0], &pba_cam_idx_[0]);
        pba_.SetNextBundleMode(ParallelBA::BUNDLE_ONLY_MOTION); //Solving for motion only
    }
}

// TODO This is wrong --- Use this as a template https://github.com/lab-x/SFM/blob/61bd10ab3f70a564b6c1971eaebc37211557ea78/SparseCloud.cpp
// Or this https://github.com/Zponpon/AR/blob/5d042ba18c1499bdb2ec8d5e5fae544e45c5bd91/PlanarAR/SFMUtil.cpp
// https://stackoverflow.com/questions/46875340/parallel-bundle-adjustment-pba
void BundleAdjustment::setPBAData(std::vector<Point3D> *pba_3d_points,
                                  std::vector<Point2D> *pba_image_points, std::vector<int> *pba_2d3d_idx,
                                  std::vector<int> *pba_cam_idx) {

    pba_3d_points->clear();
    pba_image_points->clear();
    pba_cam_idx->clear();
    pba_2d3d_idx->clear();

    for (const auto &pwm : pairwise_matches_) {
        int idx_cam0 = pwm.src_img_idx;
        int idx_cam1 = pwm.dst_img_idx;

        if (idx_cam0 != -1 && idx_cam1 != -1 && idx_cam0 != idx_cam1 && pwm.confidence > 0 &&
            idx_cam0 < idx_cam1) { //TODO experiment with confidence thresh

            std::vector<cv::Point2f> points0;
            std::vector<cv::Point2f> points1;

            for (const auto &match : pwm.matches) {
                points0.push_back(features_[idx_cam0].keypoints[match.queryIdx].pt);
                points1.push_back(features_[idx_cam1].keypoints[match.trainIdx].pt);
            }

            std::vector<cv::Point3d> points3d = triangulate(pp_, focal_, points0, points1, projection_matrices_[idx_cam0], projection_matrices_[idx_cam1]);

            //TODO clean 3D points here - remove far points and backward points.
            cv::Mat pose_t = projection_matrices_[idx_cam0].col(3).t();
            cv::Mat R = projection_matrices_[idx_cam0].colRange(cv::Range(0,3)).clone();

            //TODO get inliers and plot them so we can see the 3d points
            for (int j = 0; j < points3d.size(); j++) {

                cv::Mat p(1,3, CV_64F);
                p.at<double>(0,0) = points3d[j].x;
                p.at<double>(0,1) = points3d[j].y;
                p.at<double>(0,2) = points3d[j].z;
                float d = cv::norm(p - pose_t);
                cv::Mat p_rot = p * R.t();

                if(d < 100 && p_rot.at<double>(0,2) >  pose_t.at<double>(0,2) ) {
                    pba_3d_points->push_back(Point3D{static_cast<float>(points3d[j].x),
                                                     static_cast<float>(points3d[j].y),
                                                     static_cast<float>(points3d[j].z)});

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

            LOG(INFO ) << pba_3d_points->size();
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
