
#include "main.h"

#include <boost/range/iterator_range.hpp>
#include <boost/algorithm/string.hpp>

#include <vector>
#include <fstream>

#include <cv.hpp>
#include <cxcore.h>
#include <highgui.h>


#include "src/kitti_devkit/cpp/evaluate_odometry.h"
#include "vo.h"

using namespace boost::filesystem;


cv::Mat loadKittiCalibration(std::string calib_file, int line_number) {
    std::string line;
    std::ifstream file_stream(calib_file);
    if (file_stream.is_open()) {
        int count = 0;
        while (std::getline(file_stream, line)) {
            if (count == line_number) {
                break;
            }
            count++;
        }
        file_stream.close();
    } else {
        LOG(FATAL) << "Unable to open file";
    }

    line.erase(0, 4);

    LOG(INFO) << line;
    std::vector<std::string> elements;
    boost::algorithm::split(elements, line, boost::is_any_of(" "));

    double data[12];
    for (int i = 0; i < elements.size(); i++) {
        data[i] = std::stod(elements[i]);
    }

    return cv::Mat(3, 4, CV_64F, &data).clone();
}


Matrix kittiResultMat(cv::Mat R, cv::Mat t) {
    Matrix pose = Matrix::eye(4);
    pose.val[0][0] = R.at<double>(0, 0);
    pose.val[1][0] = R.at<double>(1, 0);
    pose.val[2][0] = R.at<double>(2, 0);
    pose.val[0][1] = R.at<double>(0, 1);
    pose.val[1][1] = R.at<double>(1, 1);
    pose.val[2][1] = R.at<double>(2, 1);
    pose.val[0][2] = R.at<double>(0, 2);
    pose.val[1][2] = R.at<double>(1, 2);
    pose.val[2][2] = R.at<double>(2, 2);

    pose.val[0][3] = t.at<double>(0, 0);
    pose.val[1][3] = t.at<double>(1, 0);
    pose.val[2][3] = t.at<double>(2, 0);
    return pose;
}

void saveResults(const std::string &gt_poses_path, const std::string &res_dir, const std::string &seq, std::vector<Matrix> &result_poses){

    boost::filesystem::create_directory(res_dir);
    std::vector<Matrix> gt_poses = loadPoses(gt_poses_path);
    std::string plot_dir = res_dir + "/plot/";

    boost::filesystem::create_directory(plot_dir);

    savePathPlot(gt_poses,result_poses, plot_dir + seq + ".txt");
    std::vector<int32_t> roi = computeRoi(gt_poses,result_poses);
    plotPathPlot(plot_dir,roi,stoi(FLAGS_seq));

    /*
    std::string err_dir = res_dir + "/err/";
    boost::filesystem::create_directory(err_dir);
    std::vector<errors> seq_err = calcSequenceErrors(gt_poses,result_poses);
    saveSequenceErrors(seq_err,err_dir + "errors.txt");
    */
}

int main(int argc, char *argv[]) {

    gflags::ParseCommandLineFlags(&argc, &argv, true);
    google::InitGoogleLogging(argv[0]);
    FLAGS_logtostderr = 1;


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

    double focal = intrinsics.at<double>(0, 0);
    cv::Point2d pp(intrinsics.at<double>(0, 2), intrinsics.at<double>(1, 2));

    LOG(INFO) << "Focal length " << focal << ", principal point: " << pp;


    //Load ground truth
    std::vector<Matrix> result_poses;

    cv::Mat map(1500, 1500, CV_8UC3, cv::Scalar(0, 0, 0));
    cv::Mat_<double> pose_t = cv::Mat::zeros(3, 1, CV_64FC1);
    cv::Mat_<double> pose_R = cv::Mat::eye(3, 3, CV_64FC1);

    bool done = false;

    VisualOdemetry vo(focal, pp);

    for (const auto &file_name : file_names) {

        cv::Mat image = cv::imread(file_name);

        vo.addImage(image, &pose_R, &pose_t);

        result_poses.emplace_back(kittiResultMat(pose_R, pose_t));

        cv::Point2d draw_pos = cv::Point2d(pose_t.at<double>(0) + map.cols / 2, -pose_t.at<double>(2) + map.rows / 2);
        cv::circle(map, draw_pos, 2, cv::Scalar(255, 0, 0), 2);

        cv::imshow("Map", map);
        cv::imshow("Features", vo.drawMatches(image));

        char key = static_cast<char>(cv::waitKey(1));
        if (key == 27) {
            done = true;
            break;
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