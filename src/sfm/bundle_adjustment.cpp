
#include "bundle_adjustment.h"

#include <glog/logging.h>

#include "triangulation.h"
#include <opencv2/sfm/triangulation.hpp>
#include <iostream>

#include "src/utils/draw.h"
#include "src/matcher/matcher.h"

#include <opencv2/viz/types.hpp>
#include <opencv2/viz/widgets.hpp>
#include <opencv2/viz/viz3d.hpp>
#include <opencv2/viz/vizcore.hpp>


BundleAdjustment::BundleAdjustment() : pba_(ParallelBA::DeviceT::PBA_CPU_DOUBLE) {
}

void BundleAdjustment::init(const cv::Mat &K, size_t max_frames) {

    char *argv[] = {(char *) "-lmi<100>", (char *) "-v", (char *) "1", nullptr};
    int argc = sizeof(argv) / sizeof(char *);

    pba_.ParseParam(argc, argv);
    pba_.SetFixedIntrinsics(true);
    matcher_ = cv::makePtr<cv::detail::BestOf2NearestMatcher>(true);
    max_frames_ = max_frames;

    K_ = K.clone();

    focal_ = cv::Point2d(K.at<double>(0, 0), K.at<double>(1, 1));
    pp_ = cv::Point2d(K.at<double>(0, 2), K.at<double>(1, 2));

    viz_ = cv::viz::Viz3d("Coordinate Frame");
    viz_.showWidget("Coordinate Widget", cv::viz::WCoordinateSystem());
}


//// THIS IS THE PROBLEM http://www.land-of-kain.de/docs/coords/
void BundleAdjustment::addKeyFrame(const VOFrame &frame) {

    CameraT cam;
    cam.f = (focal_.x + focal_.y) / 2;

    cam.SetTranslation(reinterpret_cast<double *>(frame.pose_t.data));

    //cv::Mat R  = -frame.pose_R.t(); //http://www.cs.cornell.edu/~snavely/bundler/bundler-v0.3-manual.html TODO figure out coordinate stytem that the R and t from recover Pose are in

    cam.SetMatrixRotation(reinterpret_cast<double *>(frame.pose_R.data));

    pba_cameras_.push_back(cam);

    cv::detail::ImageFeatures image_feature;
    cv::Mat descriptors;

    feature_detector_.detectComputeAKAZE(frame, &image_feature.keypoints, &descriptors);

    image_feature.descriptors = descriptors.getUMat(cv::USAGE_DEFAULT);
    image_feature.img_idx = count_++;
    image_feature.img_size = frame.image.size();

    features_.push_back(image_feature);

    if (features_.size() > max_frames_) {
        features_.erase(features_.begin());
        pba_cameras_.erase(pba_cameras_.begin());
    }

    pairwise_matches_ = matcher(features_, K_);
    match_matrix_ = createMatchMatrix(pairwise_matches_, pba_cameras_.size());
    setPBAPoints();
}

// TODO This is wrong --- Use this as a template https://github.com/lab-x/SFM/blob/61bd10ab3f70a564b6c1971eaebc37211557ea78/SparseCloud.cpp
// Or this https://github.com/Zponpon/AR/blob/5d042ba18c1499bdb2ec8d5e5fae544e45c5bd91/PlanarAR/SFMUtil.cpp
// https://stackoverflow.com/questions/46875340/parallel-bundle-adjustment-pba
void BundleAdjustment::setPBAPoints() {

    pba_3d_points_.clear();
    pba_image_points_.clear();
    pba_cam_idx_.clear();
    pba_2d3d_idx_.clear();
    points_3d_.clear();

    std::vector<cv::Mat> poses;
    for (auto c : pba_cameras_) {
        cv::Mat t = cv::Mat::zeros(3, 1, CV_64FC1);
        cv::Mat R = cv::Mat::eye(3, 3, CV_64FC1);

        c.GetTranslation(reinterpret_cast<double *>(t.data));
        c.GetMatrixRotation(reinterpret_cast<double *>(R.data));

        cv::Mat P;
        hconcat(R, t, P);
        poses.push_back(std::move(P));
    }

    cv::Mat tracks = cv::Mat::zeros(pp_.y * 2, pp_.x * 2, CV_8UC3);

    for (const auto &row : match_matrix_) {


            std::vector<cv::Mat_<double>> sfm_points_2d;
            std::vector<cv::Mat_<double>> projection_matrices;
            std::vector<cv::Point2f> points;

            std::vector<int> cams;
            for (int cam_idx = 0; cam_idx < row.size(); cam_idx++) {

                if (row[cam_idx] == -1) {
                    continue;
                }

                cams.push_back(cam_idx);

                cv::Point2f p2d = features_[cam_idx].keypoints[row[cam_idx]].pt;
                points.push_back(p2d);
                sfm_points_2d.emplace_back((cv::Mat_<double>(2, 1) << p2d.x, p2d.y));
                projection_matrices.push_back(getProjectionMatrix(K_, poses[cam_idx]));
            }

            if (cams.size() < 3) {
                continue;
            }

            cv::Mat point_3d_mat;
            cv::sfm::triangulatePoints(sfm_points_2d, projection_matrices, point_3d_mat); //What is ging on wth this result

            cv::Mat R = cv::Mat::eye(3, 3, CV_64FC1);
            cv::Mat t = cv::Mat::zeros(3, 1, CV_64FC1);

            pba_cameras_[cams[0]].GetTranslation(reinterpret_cast<double *>(t.data));
            pba_cameras_[cams[0]].GetMatrixRotation(reinterpret_cast<double *>(R.data));
            cv::Mat p_origin = R.t() * (point_3d_mat - t);
            double dist = cv::norm(p_origin);

            //TODO make sure z is pos
            if (dist < kMax3DDist && p_origin.at<double>(2) > kMin3DDist &&
                std::fabs(p_origin.at<double>(0)) < kMax3DWidth) {

                points_3d_.emplace_back(cv::Point3d(point_3d_mat.at<double>(0, 0),
                                                    point_3d_mat.at<double>(0, 1),
                                                    point_3d_mat.at<double>(0, 2)));

                pba_3d_points_.emplace_back(Point3D{static_cast<float>(point_3d_mat.at<double>(0, 0)),
                                                    static_cast<float>(point_3d_mat.at<double>(0, 1)),
                                                    static_cast<float>(point_3d_mat.at<double>(0, 2))});

                for (int i = 0; i < points.size(); i++) {


                    pba_image_points_.emplace_back(Point2D{(points[i].x - pp_.x), (points[i].y - pp_.y)});
                    pba_cam_idx_.push_back(cams[i]);
                    pba_2d3d_idx_.push_back(static_cast<int>(pba_3d_points_.size() - 1));


                    if (i < points.size() - 1) {
                        cv::line(tracks, points[i], points[i + 1], cv::Scalar(0, 255, 0), 1);
                        cv::circle(tracks, points[i], 2, cv::Scalar(255, 0, 0), 2);
                    }
                }
         }

    }

    imshow("tracks", tracks);
    LOG(INFO) << pba_3d_points_.size() << " " << pba_image_points_.size();
}


//TODO use full history rather than just updating the newest point
int BundleAdjustment::slove(cv::Mat *R, cv::Mat *t) {

    if (pba_3d_points_.empty() || pba_image_points_.empty()) {
        LOG(INFO) << "Bundle adjustment points are empty";
        return 1;
    }

    pba_.SetCameraData(pba_cameras_.size(), &pba_cameras_[0]); //set camera parameters
    pba_.SetPointData(pba_3d_points_.size(), &pba_3d_points_[0]); //set 3D point data

    //set the projections
    pba_.SetProjection(pba_image_points_.size(), &pba_image_points_[0], &pba_2d3d_idx_[0], &pba_cam_idx_[0]);

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

void BundleAdjustment::draw(float scale) {
    cv::Mat ba_map(800, 800, CV_8UC3, cv::Scalar(0, 0, 0));

    cv::line(ba_map, cv::Point(ba_map.cols / 2, 0), cv::Point(ba_map.cols / 2, ba_map.rows), cv::Scalar(0, 0, 255));
    cv::line(ba_map, cv::Point(0, ba_map.rows / 2), cv::Point(ba_map.cols, ba_map.rows / 2), cv::Scalar(0, 0, 255));

    for (const auto &p : pba_3d_points_) {
        cv::Point2d draw_pos = cv::Point2d(p.xyz[0] * scale + ba_map.cols / 2, p.xyz[2] * scale + ba_map.rows / 2);
        cv::circle(ba_map, draw_pos, 1, cv::Scalar(0, 255, 0), 1);
    }

    for (const auto &cam : pba_cameras_) {
        cv::Point2d draw_pos = cv::Point2d(cam.t[0] * scale + ba_map.cols / 2, cam.t[2] * scale + ba_map.rows / 2);
        cv::circle(ba_map, draw_pos, 2, cv::Scalar(255, 0, 0), 2);

        cv::Mat R = cv::Mat::eye(3, 3, CV_64FC1);
        cam.GetMatrixRotation(reinterpret_cast<double *>(R.data));

        double data[3] = {0, 0, 1};
        cv::Mat dir(3, 1, CV_64FC1, data);

        dir = (R * dir) * 10 * scale;

        cv::line(ba_map, draw_pos, cv::Point2d(dir.at<double>(0, 0), dir.at<double>(0, 2)) + draw_pos,
                 cv::Scalar(0, 255, 255), 2);

    }

    if (!pba_cameras_.empty()) {
        const auto &cam = pba_cameras_[pba_cameras_.size() - 1];
        cv::Point2d draw_pos = cv::Point2d(cam.t[0] * scale + ba_map.cols / 2, cam.t[2] * scale + ba_map.rows / 2);
        cv::circle(ba_map, draw_pos, 2, cv::Scalar(0, 0, 255), 2);
    }

    cv::imshow("BA_Map", ba_map);
}

void BundleAdjustment::drawViz() {

    int count = 0;
    for (auto c : pba_cameras_) {

        auto col = cv::viz::Color::red();
        if (count % 3 == 1) {
            col = cv::viz::Color::green();
        } else if (count % 3 == 2) {
            col = cv::viz::Color::blue();
        }

        cv::viz::WCameraPosition cam(cv::Matx33d(K_), 3, col);
        viz_.showWidget("c" + std::to_string(count), cam);

        cv::Mat t = cv::Mat::zeros(3, 1, CV_64FC1);
        cv::Mat R = cv::Mat::eye(3, 3, CV_64FC1);

        c.GetTranslation(reinterpret_cast<double *>(t.data));
        c.GetMatrixRotation(reinterpret_cast<double *>(R.data));

        cv::Affine3d pose(R, t);
        viz_.setWidgetPose("c" + std::to_string(count), pose);

        if (count == 0) {
            viz_.setViewerPose(pose);
        }
        count++;
    }

    if (!points_3d_.empty()) {
        cv::viz::WCloud cloud_widget(points_3d_, cv::viz::Color::green());

        viz_.showWidget("cloud", cloud_widget);
    }
    viz_.spinOnce(50);
}