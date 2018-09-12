
#include "main.h"

#include <boost/range/iterator_range.hpp>

#include "src/visual_odometry/visual_odometry.h"
#include "src/kitti/kitti.h"

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

    cv::Mat intrinsics = loadKittiCalibration(data_dir + FLAGS_calib_file, FLAGS_calib_line_number);
    LOG(INFO) << "Camera matrix: " << intrinsics;

    cv::Point2f focal(intrinsics.at<double>(0, 0) * FLAGS_image_scale, intrinsics.at<double>(1, 1) * FLAGS_image_scale);
    cv::Point2d pp(intrinsics.at<double>(0, 2) * FLAGS_image_scale, intrinsics.at<double>(1, 2) * FLAGS_image_scale);

    LOG(INFO) << "Focal length " << focal << ", principal point: " << pp;

    //Load ground truth
    std::vector<Matrix> result_poses;


    cv::Mat_<double> pose = cv::Mat::eye(4, 3, CV_64FC1);
    cv::Mat_<double> pose_kalman = cv::Mat::eye(4, 3, CV_64FC1);

    bool done = false;

    VisualOdometry vo(focal, pp, FLAGS_min_tracked_features);
    bool resize = true;

    std::vector<cv::Point2d> positions;

    for (const auto &file_name : file_names) {

        cv::Mat image = cv::imread(file_name);
        cv::resize(image, image, cv::Size(), FLAGS_image_scale, FLAGS_image_scale);

        vo.addImage(image, &pose, &pose_kalman);

        result_poses.emplace_back(kittiResultMat(pose)); //TODO coversion to kitti coordinates

        //Draw results
        cv::Mat map(1500, 1500, CV_8UC3, cv::Scalar(0, 0, 0));
        cv::Point2d last_pos(kDrawScale * pose.at<double>(0, 3) + map.cols / 2,
                    kDrawScale * pose.at<double>(2, 3) + map.rows / 4);

        positions.push_back(last_pos);

        for (const auto &pos : positions) {
            cv::circle(map, pos, 1, cv::Scalar(0, 255, 0), 2);
        }


        double data[3] = {0,0,1};
        cv::Mat dir(3,1,CV_64FC1, data);

        cv::Mat R = pose.colRange(cv::Range(0, 3));

        dir = (R * dir) * 50;

        cv::line(map, last_pos, cv::Point2d(dir.at<double>(0,0), dir.at<double>(0,2) ) + last_pos, cv::Scalar(0,255,255), 2 );



        if(resize){
            cv::resize(map, map, cv::Size(), 0.5, 0.5);
        }

        cv::imshow("Map", map);
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