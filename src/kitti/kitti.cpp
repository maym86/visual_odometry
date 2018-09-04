#include "kitti.h"

#include <fstream>
#include <glog/logging.h>

#include <boost/algorithm/string.hpp>
#include <boost/filesystem/operations.hpp>


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


Matrix kittiResultMat(cv::Mat pose) {
    Matrix pose_kitti = Matrix::eye(4);
    for(int r=0; r < pose.rows; r++){
        for(int c=0; c < pose.cols; c++){
            pose_kitti.val[r][c] = pose.at<double>(r, c);
        }
    }
    return pose_kitti;
}

void saveResults(const std::string &gt_poses_path, const std::string &res_dir, const std::string &seq, std::vector<Matrix> &result_poses){

    boost::filesystem::create_directory(res_dir);
    std::vector<Matrix> gt_poses = loadPoses(gt_poses_path);

    gt_poses.resize(result_poses.size()); //Resize gt poses if we stopped early
    std::string plot_dir = res_dir + "/plot/";

    boost::filesystem::create_directory(plot_dir);

    savePathPlot(gt_poses,result_poses, plot_dir + seq + ".txt");
    std::vector<int32_t> roi = computeRoi(gt_poses,result_poses);
    plotPathPlot(plot_dir,roi,stoi(seq));

    /*
    std::string err_dir = res_dir + "/err/";
    boost::filesystem::create_directory(err_dir);
    std::vector<errors> seq_err = calcSequenceErrors(gt_poses,result_poses);
    saveSequenceErrors(seq_err,err_dir + "errors.txt");
    */
}