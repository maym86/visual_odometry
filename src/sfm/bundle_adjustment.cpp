
#include "bundle_adjustment.h"

#include <glog/logging.h>

#include "triangulation.h"
#include <opencv2/sfm/triangulation.hpp>
#include <iostream>

#include "src/utils/draw.h"

#include <opencv2/viz/types.hpp>
#include <opencv2/viz/widgets.hpp>
#include <opencv2/viz/viz3d.hpp>
#include <opencv2/viz/vizcore.hpp>


BundleAdjustment::BundleAdjustment(bool global_pose) : pba_(ParallelBA::DeviceT::PBA_CPU_DOUBLE) {
    global_pose_ = global_pose;
}

void BundleAdjustment::init(const cv::Mat &K, size_t max_frames) {

    char *argv[] = {(char *) "-lmi<100>", (char *) "-v", (char *) "1", nullptr};
    int argc = sizeof(argv) / sizeof(char *);

    pba_.ParseParam(argc, argv);
    pba_.SetFixedIntrinsics(true);
    matcher_ = cv::makePtr<cv::detail::BestOf2NearestMatcher>(true);
    max_frames_ = max_frames;

    K_ = K.clone();

    focal_ = cv::Point2d(K.at<double>(0,0), K.at<double>(1,1));
    pp_ = cv::Point2d(K.at<double>(0,2), K.at<double>(1,2));
}


void BundleAdjustment::matcher() {
    const float ratio = 0.8; // As in Lowe's paper; can be tuned

    cv::BFMatcher matcher;
    pairwise_matches_.clear();
    for (int i = 0; i < features_.size() - 1; i++) {
        int j = i + 1;

        std::vector<std::vector<cv::DMatch>> matches;
        matcher.knnMatch(features_[i].descriptors, features_[j].descriptors, matches, 2);  // Find two nearest matches

        cv::detail::MatchesInfo good_matches;

        std::vector<cv::Point2f> points0;
        std::vector<cv::Point2f> points1;
        for (int k = 0; k < matches.size(); k++) {

            if (matches[k][0].distance < ratio * matches[k][1].distance) {

                const auto &p0 = features_[i].keypoints[matches[k][0].queryIdx];
                const auto &p1 = features_[j].keypoints[matches[k][0].trainIdx];

                if (cv::norm(cv::Mat(p0.pt) - cv::Mat(p1.pt)) < 100) {
                    good_matches.matches.push_back(matches[k][0]);
                    good_matches.inliers_mask.push_back(1);

                    points0.push_back(p0.pt);
                    points1.push_back(p1.pt);
                }

            }
        }
        cv::Mat mask, R, t;
        cv::Mat E = cv::findEssentialMat(points0, points1, K_, cv::RANSAC, 0.999, 1.0, mask);

        for (int k = 0; k < good_matches.inliers_mask.size(); k++) {
            if (!mask.at<bool>(k)) {
                good_matches.inliers_mask[k] = 0;
            }
        }

        good_matches.src_img_idx = i;
        good_matches.dst_img_idx = j;

        pairwise_matches_.push_back(good_matches);
    }

    createTracks();
}

void BundleAdjustment::createTracks() {

    tracks_.clear();
    std::vector<std::unordered_map<int, int>> pairs(pba_cameras_.size() - 1);
    for (auto &pwm : pairwise_matches_) {
        int idx_cam0 = pwm.src_img_idx;
        for (int i = 0; i < pwm.matches.size(); i++) {
            auto &match = pwm.matches[i];

            if (pwm.inliers_mask[i] != 0) {
                pairs[idx_cam0][match.queryIdx] = match.trainIdx;
            }
        }
    }

    tracks_.resize(pba_cameras_.size() - 1);

    for (int cam_idx = 0; cam_idx < pairs.size(); cam_idx++) {

        for (std::pair<int, int> element : pairs[cam_idx]) {

            if (element.second == -1) {
                continue;
            }

            std::vector<int> track;
            track.push_back(element.first);
            track.push_back(element.second);
            int key = element.second;
            int cam = cam_idx + 1;

            if (cam < pairs.size()) {
                while (pairs[cam].find(key) != pairs[cam].end()) {
                    key = pairs[cam][key];
                    if (key == -1) {
                        break;
                    }

                    pairs[cam][key] = -1; //seen
                    track.push_back(key);
                    cam++;

                    if (cam == pairs.size()) {
                        break;
                    }
                }
            }
            tracks_[cam_idx].push_back(track);
        }
    }
}

void BundleAdjustment::addKeyFrame(const VOFrame &frame) {

    CameraT cam;
    cam.f = (focal_.x + focal_.y) / 2;
    cam.SetTranslation(reinterpret_cast<double *>(frame.pose_t.data));

    //cv::Mat R  = -frame.pose_R.t(); //http://www.cs.cornell.edu/~snavely/bundler/bundler-v0.3-manual.html TODO figure out coordinate stytem that the R and t from recover Pose are in
    cam.SetMatrixRotation(reinterpret_cast<double *>( frame.pose_R.data));

    pba_cameras_.push_back(cam);

    cv::detail::ImageFeatures image_feature;
    cv::Mat descriptors;

    feature_detector_.computeORB(frame, &image_feature.keypoints, &descriptors);
    image_feature.descriptors = descriptors.getUMat(cv::USAGE_DEFAULT);
    image_feature.img_idx = count_++;
    image_feature.img_size = frame.image.size();

    features_.push_back(image_feature);

    if (features_.size() > max_frames_) {
        features_.erase(features_.begin());
        pba_cameras_.erase(pba_cameras_.begin());
    }

    pba_cameras_[0].SetConstantCamera();

    matcher();

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

    for (int cam_idx = 0; cam_idx < tracks_.size(); cam_idx++) {

        for (const auto &track : tracks_[cam_idx]) {

            std::vector<cv::Point2f> points;
            for (int i = 0; i < track.size(); i++) {
                points.push_back(features_[cam_idx + i].keypoints[track[i]].pt);
            }

            if (points.size() < 2) {
                continue;
            }

            std::vector<cv::Point2f> points0 = {points[0]};
            std::vector<cv::Point2f> points1 = {points[1]};

            std::vector<cv::Mat> sfm_points_2d;
            std::vector<cv::Mat> projection_matrices;
            double dist = 0;
            cv::Mat p, p_up;
            std::vector<cv::Point3d> points3d;

            if (!global_pose_) { //reset the points to 0,0 before calc
                cv::Mat t = cv::Mat::zeros(3, 1, CV_64FC1);
                cv::Mat R = cv::Mat::eye(3, 3, CV_64FC1);

                pba_cameras_[cam_idx].GetTranslation(reinterpret_cast<double *>(t.data));
                pba_cameras_[cam_idx].GetMatrixRotation(reinterpret_cast<double *>(R.data));

                for (int i = 0; i < points.size(); i++) {
                    cv::Mat mat_point = (cv::Mat_<double>(2, 1) << points[i].x, points[i].y);
                    sfm_points_2d.push_back(mat_point);

                    cv::Mat pose = poses[cam_idx + i].clone();

                    pose.col(3) -= t;
                    pose.colRange(cv::Range(0, 3)) *= R.t();

                    projection_matrices.push_back(K_ * pose);
                }

                cv::Mat point_3d_mat;
                cv::sfm::triangulatePoints(sfm_points_2d, projection_matrices, point_3d_mat);
                points3d = points3DToVec(point_3d_mat);

                p = cv::Mat(points3d[0]);
                dist = cv::norm(p);
                p = (R * p) + t;

            } else {

                for (int i = 0; i < points.size(); i++) {
                    sfm_points_2d.emplace_back((cv::Mat_<double>(2, 1) << points[i].x, points[i].y));
                    projection_matrices.push_back(K_ * poses[cam_idx + i]);
                }

                cv::Mat point_3d_mat;
                cv::sfm::triangulatePoints(sfm_points_2d, projection_matrices, point_3d_mat);
                points3d = points3DToVec(point_3d_mat);

                p = cv::Mat(points3d[0]);
                dist = cv::norm(p - poses[cam_idx].col(3));

                cv::Mat R = cv::Mat::eye(3, 3, CV_64FC1);
                cv::Mat t = cv::Mat::zeros(3, 1, CV_64FC1);

                pba_cameras_[cam_idx].GetTranslation(reinterpret_cast<double *>(t.data));
                pba_cameras_[cam_idx].GetMatrixRotation(reinterpret_cast<double *>(R.data));
                p_up = (R.t() * p) - t;

            }

            //TODO make sure z is pos
            if (dist < kMax3DDist ){ //&& p_up.at<double>(0,2) < 0) { //TODO why are points wrong when I draw them

                points_3d_.push_back(cv::Point3d(static_cast<float>(p.at<double>(0, 0)),
                        static_cast<float>(p.at<double>(0, 1)),
                                                         static_cast<float>(p.at<double>(0, 2))));

                pba_3d_points_.emplace_back(Point3D{static_cast<float>(-p.at<double>(0, 0)),
                                                    static_cast<float>(-p.at<double>(0, 1)),
                                                    static_cast<float>(-p.at<double>(0, 2))});

                for (int i = 0; i < points.size(); i++) {

                    //LOG(INFO) << cam_idx + i << " " << i;
                    //reprojectionInfo(points[i], points3d[0], poses[cam_idx + i]); //TODO For info - remove later

                    pba_image_points_.emplace_back(Point2D{(points[i].x - pp_.x), (points[i].y - pp_.y)});
                    pba_cam_idx_.push_back(cam_idx + i);
                    pba_2d3d_idx_.push_back(static_cast<int>(pba_3d_points_.size() - 1));


                    if (i < points.size() - 1) {
                        cv::line(tracks, points[i], points[i + 1], cv::Scalar(0, 255, 0), 1);
                        cv::circle(tracks, points[i], 2, cv::Scalar(255, 0, 0), 2);
                    }
                }
            }
        }
    }

    imshow("tracks", tracks);
    LOG(INFO) << pba_3d_points_.size() << " " << pba_image_points_.size();
}

void BundleAdjustment::reprojectionInfo(const cv::Point2f &point, const cv::Point3f &point3d, const cv::Mat &proj_mat) {
    cv::Mat p_h = cv::Mat::ones(4, 1, CV_64FC1);

    p_h.at<double>(0) = point3d.x;
    p_h.at<double>(1) = point3d.y;
    p_h.at<double>(2) = point3d.z;

    cv::Mat repo = proj_mat * p_h;

    repo.at<double>(0) /= repo.at<double>(2);
    repo.at<double>(1) /= repo.at<double>(2);
    repo.at<double>(2) /= repo.at<double>(2);

    cv::Mat p_2d = cv::Mat::ones(3, 1, CV_64FC1);
    p_2d.at<double>(0) = (point.x - pp_.x) / focal_.x;
    p_2d.at<double>(1) = (point.y - pp_.y) / focal_.y;

    LOG(INFO) << cv::norm(p_2d - repo) << " " << repo.t()
                << (point.x - pp_.x) / focal_.x << ","
                << (point.y - pp_.y) / focal_.y;
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
    pba_.SetNextBundleMode(ParallelBA::BUNDLE_ONLY_MOTION); //Solving for motion only

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
    cv::line(ba_map, cv::Point(0, ba_map.rows / 4), cv::Point(ba_map.cols, ba_map.rows / 4), cv::Scalar(0, 0, 255));

    for (const auto &p : pba_3d_points_) {
        cv::Point2d draw_pos = cv::Point2d(p.xyz[0] * scale + ba_map.cols / 2, p.xyz[2] * scale + ba_map.rows / 4);
        cv::circle(ba_map, draw_pos, 1, cv::Scalar(0, 255, 0), 1);
    }

    for (const auto &cam : pba_cameras_) {
        cv::Point2d draw_pos = cv::Point2d(cam.t[0] * scale + ba_map.cols / 2, cam.t[2] * scale + ba_map.rows / 4);
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
        cv::Point2d draw_pos = cv::Point2d(cam.t[0] * scale + ba_map.cols / 2, cam.t[2] * scale + ba_map.rows / 4);
        cv::circle(ba_map, draw_pos, 2, cv::Scalar(0, 0, 255), 2);
    }

    cv::imshow("BA_Map", ba_map);
}



void BundleAdjustment::drawViz(){

    cv::viz::Viz3d myWindow("Coordinate Frame");
    myWindow.showWidget("Coordinate Widget", cv::viz::WCoordinateSystem());

    int count = 0;
    for (auto c : pba_cameras_) {

        auto col = cv::viz::Color::red();
        if(count % 3 == 1) {
            col = cv::viz::Color::green();
        } else if(count % 3 == 2) {
            col = cv::viz::Color::blue();
        }

        cv::viz::WCameraPosition cam(cv::Matx33d(K_), 3, col);
        myWindow.showWidget("c" + std::to_string(count), cam);

        cv::Mat t = cv::Mat::zeros(3, 1, CV_64FC1);
        cv::Mat R = cv::Mat::eye(3, 3, CV_64FC1);

        c.GetTranslation(reinterpret_cast<double *>(t.data));
        c.GetMatrixRotation(reinterpret_cast<double *>(R.data));

        cv::Affine3d pose(R, t);
        myWindow.setWidgetPose("c" + std::to_string(count), pose);
        count++;
    }
    cv::viz::WCloud cloud_widget1(points_3d_, cv::viz::Color::green());

    //myWindow.showWidget("cloud 2", cloud_widget1);

    myWindow.spin();
}
