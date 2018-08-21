
#include "main.h"

#include <boost/range/iterator_range.hpp>
#include <boost/algorithm/string.hpp>

#include <vector>
#include <fstream>

#include <opencv2/imgcodecs.hpp>
#include <cv.hpp>
#include <cxcore.h>
#include <highgui.h>

#include "src/features/feature_matcher.h"
#include "src/features/feature_tracker.h"

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


cv::Mat drawMatches(const cv::Mat &img, const std::vector<cv::Point2f> &p0, const std::vector<cv::Point2f> &p1,
                    const cv::Mat &mask, const cv::Scalar &color) {

    cv::Mat output = img.clone();
    for (int i = 0; i < p0.size(); i++) {
        if (mask.at<bool>(i)) {
            cv::line(output, p0[i], p1[i], color, 2);
        }
    }

    return output;

}

int checkMask(const cv::Mat &mask, int size) {

    int count = 0;
    for (int i = 0; i < size; i++) {
        if (mask.at<bool>(i)) {
            count++;
        }
    }
    return count;

}


int main(int argc, char *argv[]) {

    gflags::ParseCommandLineFlags(&argc, &argv, true);
    google::InitGoogleLogging(argv[0]);
    FLAGS_logtostderr = 1;

    //Iterate through directory
    path p(FLAGS_data_dir + "/" + FLAGS_image_dir);

    if (!is_directory(p)) {
        LOG(INFO) << FLAGS_data_dir + "/" + FLAGS_image_dir + " is not a directory";
        return 0;
    }

    std::vector<std::string> file_names;
    for (auto &entry : boost::make_iterator_range(directory_iterator(p), {})) {
        file_names.push_back(entry.path().string());
    }
    std::sort(file_names.begin(), file_names.end());

    cv::Mat output;

    cv::Mat intrinsics = loadKittiCalibration(FLAGS_data_dir + "/" + FLAGS_calib_file, FLAGS_calib_line_number);
    LOG(INFO) << "Camera matrix: " << intrinsics;

    double focal = intrinsics.at<double>(0, 0);
    cv::Point2d pp(intrinsics.at<double>(0, 2), intrinsics.at<double>(1, 2));
    LOG(INFO) << "Focal length " << focal << ", principal point: " << pp;


    cv::Mat map(1500, 1500, CV_8UC3, cv::Scalar(0, 0, 0));
    cv::Mat_<double> pos_t = cv::Mat::zeros(3, 1, CV_64FC1);
    cv::Mat_<double> pos_R = cv::Mat::eye(3, 3, CV_64FC1);

    FeatureMatcher feature_matcher;

    bool done = false;
    int match_count = 0;


    cv::Mat prev_img;
    std::vector<cv::DMatch> matches;
    std::vector<cv::Point2f> points0, points1;

    bool tracking = false;
    cv::Scalar draw_color(0, 255, 0);
    for (const auto &file_name : file_names) {

        cv::Mat img = cv::imread(file_name);
        if (prev_img.empty()) {
            prev_img = img.clone();
            continue;
        }

        if (!tracking) {
            draw_color = cv::Scalar(255, 0, 0);
            feature_matcher.addFrame(prev_img);
            feature_matcher.addFrame(img);
            feature_matcher.getMatches(&points0, &points1);
            if (points0.size() < 8) {
                LOG(WARNING) << "Too few good matches";
                continue;
            }
            tracking = true;

        } else {

            draw_color = cv::Scalar(0, 0, 255);
            points1 = trackPoints(prev_img, img, &points0);
        }

        if (points1.size() < kMinTrackedPoints) {
            tracking = false;
        }

        LOG(INFO) << "Points size: " << points0.size() << ", " << points1.size();

        cv::Mat E, R, t, mask;

        E = cv::findEssentialMat(points0, points1, focal, pp, cv::RANSAC, 0.999, 1.0, mask);
        int res = recoverPose(E, points0, points1, R, t, focal, pp, mask);

        if(res > 10) {
            pos_R = R * pos_R;
            pos_t += kScale * (pos_R * t);

            cv::Point2d draw_pos = cv::Point2d(pos_t.at<double>(0) + map.cols / 2, pos_t.at<double>(2) + map.rows / 2);
            cv::circle(map, draw_pos, 2, cv::Scalar(255, 0, 0), 2);
        }

        cv::imshow("Map", map);
        cv::imshow("Features", drawMatches(img, points0, points1, mask, draw_color));

        char key = static_cast<char>(cv::waitKey(1));
        if (key == 27) {
            done = true;
            break;
        }

        points0 = points1;
        prev_img = img.clone();
    }


    while (!done) {
        char key = static_cast<char>(cv::waitKey(33));
        if (key == 27) {
            done = true;
        }
    }

}