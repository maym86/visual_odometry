
#include "main.h"

#include <boost/range/iterator_range.hpp>

#include "src/visual_odometry/visual_odometry.h"
#include "src/kitti/kitti.h"

#include <opencv2/viz/types.hpp>
#include <opencv2/viz/widgets.hpp>
#include <opencv2/viz/viz3d.hpp>
#include <opencv2/viz/vizcore.hpp>

using namespace boost::filesystem;

int main(int argc, char *argv[]) {

    gflags::ParseCommandLineFlags(&argc, &argv, true);
    google::InitGoogleLogging(argv[0]);
    FLAGS_logtostderr = true;

    std::string data_dir = FLAGS_data_dir + "/" + FLAGS_seq + "/";

    //Iterate through directory
    path p(data_dir + FLAGS_image_dir);

    if (!is_directory(p)) {
        LOG(INFO) << data_dir + FLAGS_image_dir + " is not a directory";
        return 0;
    }

    std::vector<std::string> file_names;
    for (auto &entry : boost::make_iterator_range(directory_iterator(p), {})) {
        file_names.push_back(entry.path().string());
    }
    std::sort(file_names.begin(), file_names.end());

    cv::Mat output;

    cv::Mat K = loadKittiCalibration(data_dir + FLAGS_calib_file, FLAGS_calib_line_number);

    K.at<double>(0, 0) *= FLAGS_image_scale;
    K.at<double>(1, 1) *= FLAGS_image_scale;
    K.at<double>(0, 2) *= FLAGS_image_scale;
    K.at<double>(1, 2) *= FLAGS_image_scale;

    LOG(INFO) << "Camera matrix: " << K;

    //Load ground truth
    std::vector<Matrix> result_poses;

    cv::Mat_<double> pose = cv::Mat::eye(3, 4, CV_64FC1);
    cv::Mat_<double> pose_kalman = cv::Mat::eye(3, 4, CV_64FC1);

    bool done = false;

    VisualOdometry vo(K, FLAGS_min_tracked_features);
    bool resize = true;

    std::vector<cv::Point3d> positions;


    cv::viz::Viz3d map_window("Map");

    map_window.showWidget("Map", cv::viz::WCoordinateSystem());
    for (const auto &file_name : file_names) {

        cv::Mat image = cv::imread(file_name);
        cv::resize(image, image, cv::Size(), FLAGS_image_scale, FLAGS_image_scale);

        vo.addImage(image, &pose, &pose_kalman);

        result_poses.emplace_back(kittiResultMat(pose)); //TODO coversion to kitti coordinates

        //Draw results
        cv::Mat map(1500, 1500, CV_8UC3, cv::Scalar(0, 0, 0));

        cv::line(map, cv::Point(map.cols / 2, 0), cv::Point(map.cols / 2, map.rows), cv::Scalar(0, 0, 255));
        cv::line(map, cv::Point(0, map.rows / 4), cv::Point(map.cols, map.rows / 4), cv::Scalar(0, 0, 255));

        positions.push_back(cv::Point3d(pose.col(3)));

        cv::viz::WCameraPosition cam(cv::Matx33d(K), 3, cv::viz::Color::white());
        map_window.showWidget("cam", cam);

        cv::Affine3d cam_pose(pose.colRange(cv::Range(0,3)), pose.col(3));
        map_window.setWidgetPose("cam", cam_pose);

        cv::viz::WCloud cloud_widget1(positions, cv::viz::Color::green());

        map_window.showWidget("cloud 2", cloud_widget1);

        map_window.spinOnce();

        cv::imshow("Features", vo.drawMatches(image));
        cv::imshow("3D", vo.draw3D());

        char key = static_cast<char>(cv::waitKey(1));
        if (key == 27) {
            done = true;
            break;
        } else if (key == 'r') {
            resize = !resize;
        }
    }

    std::string res_dir = FLAGS_res_dir + "/" + FLAGS_seq;
    std::string gt_poses_path = FLAGS_poses + "/" + FLAGS_seq + ".txt";

    saveResults(gt_poses_path, res_dir, FLAGS_seq, result_poses);

    while (!done) {
        char key = static_cast<char>(cv::waitKey(33));
        if (key == 27) {
            done = true;
        }
    }


}