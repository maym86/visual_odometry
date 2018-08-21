

#ifndef VISUALODEMETRY_EVALUATE_ODOMETRY_H
#define VISUALODEMETRY_EVALUATE_ODOMETRY_H

#include <vector>

#include "matrix.h"



struct errors {
    int32_t first_frame;
    float   r_err;
    float   t_err;
    float   len;
    float   speed;
    errors (int32_t first_frame,float r_err,float t_err,float len,float speed) :
            first_frame(first_frame),r_err(r_err),t_err(t_err),len(len),speed(speed) {}
};


std::vector<Matrix> loadPoses(std::string file_name);

std::vector<float> trajectoryDistances (std::vector<Matrix> &poses);

int32_t lastFrameFromSegmentLength(std::vector<float> &dist,int32_t first_frame,float len);

inline float rotationError(Matrix &pose_error);

inline float translationError(Matrix &pose_error);

std::vector<errors> calcSequenceErrors (std::vector<Matrix> &poses_gt,std::vector<Matrix> &poses_result);

void saveSequenceErrors (std::vector<errors> &err,std::string file_name);

void savePathPlot (std::vector<Matrix> &poses_gt,std::vector<Matrix> &poses_result,std::string file_name);

std::vector<int32_t> computeRoi (std::vector<Matrix> &poses_gt,std::vector<Matrix> &poses_result) ;

void plotPathPlot (std::string dir,std::vector<int32_t> &roi,int32_t idx) ;

void saveErrorPlots(std::vector<errors> &seq_err,std::string plot_error_dir,char* prefix);

void plotErrorPlots (std::string dir,char* prefix);

void saveStats (std::vector<errors> err,std::string dir);

//bool eval (std::string result_sha,Mail* mail);



#endif //VISUALODEMETRY_EVALUATE_ODOMETRY_H
