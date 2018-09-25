
#include "bundle_adjustment.h"

#include <glog/logging.h>

#include "triangulation.h"
#include <opencv2/sfm/triangulation.hpp>
#include <iostream>

#include "src/utils/draw.h"
#include "src/matcher/matcher.h"

#include <opencv2/core/eigen.hpp>


void BundleAdjustment::init(const cv::Mat &K, size_t max_frames) {

    max_frames_ = max_frames;

    K_ = K.clone();

    focal_ = cv::Point2d(K.at<double>(0, 0), K.at<double>(1, 1));
    pp_ = cv::Point2d(K.at<double>(0, 2), K.at<double>(1, 2));

    viz_ = cv::viz::Viz3d("Coordinate Frame");
    viz_.showWidget("Coordinate Widget", cv::viz::WCoordinateSystem());
}

void BundleAdjustment::addKeyFrame(const VOFrame &frame) {

    camera_matrix_.push_back(K_.clone());
    R_.push_back(frame.pose_R.clone());
    t_.push_back(frame.pose_t.clone());
    dist_coeffs_.push_back(cv::Mat::zeros(5, 1, CV_64F));

    cv::detail::ImageFeatures image_feature;
    cv::Mat descriptors;

    feature_detector_.detectComputeAKAZE(frame, &image_feature.keypoints, &descriptors);

    image_feature.descriptors = descriptors.getUMat(cv::USAGE_DEFAULT);
    image_feature.img_idx = count_++;
    image_feature.img_size = frame.image.size();

    features_.push_back(image_feature);

    if (features_.size() > max_frames_) {
        features_.erase(features_.begin());
        camera_matrix_.erase(camera_matrix_.begin());
        R_.erase(R_.begin());
        t_.erase(t_.begin());
        dist_coeffs_.erase(dist_coeffs_.begin());
    }

    pairwise_matches_ = matcher(features_, K_);
    match_matrix_ = createMatchMatrix(pairwise_matches_, R_.size());
    setPBAPoints();
}


void BundleAdjustment::setPBAPoints() {

    points_3d_.clear();
    points_img_.clear();
    cameras_visible_.clear();
    points_img_.clear();

    cv::Mat tracks = cv::Mat::zeros(pp_.y * 2, pp_.x * 2, CV_8UC3);

    for (const auto &row : match_matrix_) {

        std::vector<cv::Point2f> points;
        std::vector<cv::Mat_<double>> sfm_points_2d;
        std::vector<cv::Mat_<double>> projection_matrices;

        std::vector<int> cams;
        for (int cam_idx = 0; cam_idx < row.size(); cam_idx++) {

            if (row[cam_idx] == -1) {
                continue;
            }

            cams.push_back(cam_idx);

            points.push_back(features_[cam_idx].keypoints[row[cam_idx]].pt);

            sfm_points_2d.push_back(cv::Mat(points[cam_idx]).reshape(1));
            cv::Mat P;
            hconcat(R_[cam_idx], t_[cam_idx], P);
            projection_matrices.push_back(getProjectionMatrix(K_, P));
        }

        if (points.size() < 3) {
            continue;
        }

        cv::Mat_<double> point_3d_mat;
        cv::sfm::triangulatePoints(sfm_points_2d, projection_matrices, point_3d_mat);
        cv::Point3d point_3d(point_3d_mat);

        cv::Mat p_origin = R_[cams[0]].t() * (point_3d_mat - t_[cams[0]]);
        double dist = cv::norm(p_origin);

        if (dist < kMax3DDist && p_origin.at<double>(2) > kMin3DDist &&
            std::fabs(p_origin.at<double>(0)) < kMax3DWidth) {

            points_3d_.push_back(point_3d);
            cameras_visible_.push_back(cams);
            points_img_.push_back(points);

            //Reprojection error for INFO
            //cv::Point2f proj_point = reprojectPoint(cams[0], point_3d);
            //LOG(INFO) << proj_point << points[0] << " " << cv::norm(cv::Mat(proj_point) - cv::Mat(points[0]));

            for (int i = 0; i < points.size(); i++) {
                if (i < points.size() - 1) {
                    cv::line(tracks, points[i], points[i + 1], cv::Scalar(0, 255, 0), 1);
                    cv::circle(tracks, points[i], 2, cv::Scalar(255, 0, 0), 2);
                }
            }
        }
    }
    drawViz();
    imshow("tracks", tracks);
}

cv::Point2f BundleAdjustment::reprojectPoint(const int cam, cv::Point3d &point_3d) const {
    std::vector<cv::Point2f> image_points;
    std::vector<cv::Point3f> object_points;

    object_points.push_back(point_3d);

    cv::Mat R = R_[cam].t();
    cv::Mat t = -R_[cam].t() * t_[cam];

    projectPoints(object_points, R, t, K_, cv::noArray(), image_points );

    return image_points[0];
}


int BundleAdjustment::slove(cv::Mat *R, cv::Mat *t) {

    LOG(INFO) << points_3d_.size();
    if (points_3d_.size() < 5 * R_.size()) {
        LOG(INFO) << "Low number of triangulated points: " << points_3d_.size();
        return 1;
    }

    using namespace gtsam;

    Values result;

    Cal3_S2 K(focal_.x, focal_.y, 0 /* skew */, pp_.x, pp_.y);
    noiseModel::Isotropic::shared_ptr measurement_noise = noiseModel::Isotropic::Sigma(2, 2.0); // pixel error in (x,y)

    NonlinearFactorGraph graph;
    Values initial;

    // Poses
    for (size_t pose_idx = 0; pose_idx < R_.size(); pose_idx++) {

        Rot3 R(
                R_[pose_idx].at<double>(0, 0),
                R_[pose_idx].at<double>(0, 1),
                R_[pose_idx].at<double>(0, 2),

                R_[pose_idx].at<double>(1, 0),
                R_[pose_idx].at<double>(1, 1),
                R_[pose_idx].at<double>(1, 2),

                R_[pose_idx].at<double>(2, 0),
                R_[pose_idx].at<double>(2, 1),
                R_[pose_idx].at<double>(2, 2)
        );

        Point3 t;

        t(0) = t_[pose_idx].at<double>(0);
        t(1) = t_[pose_idx].at<double>(1);
        t(2) = t_[pose_idx].at<double>(2);

        Pose3 pose(R, t);

        // Add prior for the first image
        if (pose_idx == 0) {
            noiseModel::Diagonal::shared_ptr pose_noise = noiseModel::Diagonal::Sigmas(
                    (Vector(6) << Vector3::Constant(0.01), Vector3::Constant(0.01)).finished()); // Noise for x,y,z and rad on roll,pitch,yaw
            graph.push_back(PriorFactor<Pose3>(Symbol('x', pose_idx), pose, pose_noise)); // add directly to graph
        }

        initial.insert(Symbol('x', pose_idx), pose);
    }

    for (int kp_idx = 0; kp_idx < points_img_.size(); kp_idx++) {

        for (int i = 0; i < points_img_[kp_idx].size(); i++) {
            Point2 pt;
            pt(0) = points_img_[kp_idx][i].x;
            pt(1) = points_img_[kp_idx][i].y;
            int pose_idx = cameras_visible_[kp_idx][i];

            graph.emplace_shared<GeneralSFMFactor2<Cal3_S2>>(pt, measurement_noise, Symbol('x', pose_idx), Symbol('l', kp_idx), Symbol('K', 0));
        }

    }

    // Add a prior on the calibration.
    initial.insert(Symbol('K', 0), K);

    noiseModel::Diagonal::shared_ptr cal_noise = noiseModel::Diagonal::Sigmas((Vector(5) << 5, 5, 0 , 5, 5).finished());
    graph.emplace_shared<PriorFactor<Cal3_S2>>(Symbol('K', 0), K, cal_noise);

    // Initialize estimate for landmarks
    bool init_prior = false;

    for (size_t kp_idx = 0; kp_idx < points_3d_.size(); kp_idx++) {
        initial.insert<Point3>(Symbol('l', kp_idx),
                               Point3(points_3d_[kp_idx].x, points_3d_[kp_idx].y, points_3d_[kp_idx].z));

        if (!init_prior) {
            init_prior = true;

            noiseModel::Isotropic::shared_ptr point_noise = noiseModel::Isotropic::Sigma(3, 0.1);
            Point3 p(points_3d_[kp_idx].x, points_3d_[kp_idx].y, points_3d_[kp_idx].z);
            graph.emplace_shared<PriorFactor<Point3>>(Symbol('l', kp_idx), p, point_noise);
        }
    }

    result = LevenbergMarquardtOptimizer(graph, initial).optimize();

    LOG(INFO) << "initial graph error = " << graph.error(initial);
    LOG(INFO) << "final graph error = " << graph.error(result);

    for (size_t pose_idx = 0; pose_idx < R_.size(); pose_idx++) {
        cv::eigen2cv(result.at<Pose3>(Symbol('x', pose_idx)).rotation().matrix(), R_[pose_idx]);
        cv::eigen2cv(result.at<Pose3>(Symbol('x', pose_idx)).translation().vector(), t_[pose_idx]);
    }

    *R = R_[R_.size() - 1];
    *t = t_[t_.size() - 1];

    return 0;
}


void BundleAdjustment::draw(float scale) {
    cv::Mat ba_map(800, 800, CV_8UC3, cv::Scalar(0, 0, 0));

    cv::line(ba_map, cv::Point(ba_map.cols / 2, 0), cv::Point(ba_map.cols / 2, ba_map.rows), cv::Scalar(0, 0, 255));
    cv::line(ba_map, cv::Point(0, ba_map.rows / 2), cv::Point(ba_map.cols, ba_map.rows / 2), cv::Scalar(0, 0, 255));

    for (const auto &p : points_3d_) {
        cv::Point2d draw_pos = cv::Point2d(p.x * scale + ba_map.cols / 2, p.z * scale + ba_map.rows / 2);
        cv::circle(ba_map, draw_pos, 1, cv::Scalar(0, 255, 0), 1);
    }

    for (int i = 0; i < t_.size(); i++) {

        const cv::Mat &t = t_[i];
        cv::Point2d draw_pos = cv::Point2d(t.at<double>(0) * scale + ba_map.cols / 2,
                                           t.at<double>(2) * scale + ba_map.rows / 2);
        cv::circle(ba_map, draw_pos, 2, cv::Scalar(255, 0, 0), 2);

        const cv::Mat &R = R_[i];
        double data[3] = {0, 0, 1};
        cv::Mat dir(3, 1, CV_64FC1, data);

        dir = (R * dir) * 10 * scale;

        cv::line(ba_map, draw_pos, cv::Point2d(dir.at<double>(0, 0), dir.at<double>(0, 2)) + draw_pos,
                 cv::Scalar(0, 255, 255), 2);

    }

    if (!t_.empty()) {
        const auto &t = t_[t_.size() - 1];
        cv::Point2d draw_pos = cv::Point2d(t.at<double>(0) * scale + ba_map.cols / 2,
                                           t.at<double>(2) * scale + ba_map.rows / 2);
        cv::circle(ba_map, draw_pos, 2, cv::Scalar(0, 0, 255), 2);
    }

    cv::imshow("BA_Map", ba_map);
}


void BundleAdjustment::drawViz() {

    int count = 0;
    for (int i = 0; i < t_.size(); i++) {

        const cv::Mat &t = t_[i];
        const cv::Mat &R = R_[i];

        auto col = cv::viz::Color::red();
        if (count % 3 == 1) {
            col = cv::viz::Color::green();
        } else if (count % 3 == 2) {
            col = cv::viz::Color::blue();
        }

        cv::viz::WCameraPosition cam(cv::Matx33d(K_), 3, col);
        viz_.showWidget("c" + std::to_string(count), cam);

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
