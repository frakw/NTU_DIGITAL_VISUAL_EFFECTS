#ifndef _SIFT_H_
#define _SIFT_H_
#define _USE_MATH_DEFINES
#include <cmath>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <tuple>
#include <utility>
#include <algorithm>

#include "sift_define.h"
#include "feature_point.h"
#include "match_info.h"
using namespace std;
using namespace cv;

vector<FeaturePoint> SIFT(Mat img);

vector<Mat> get_gaussian_pyramid(Mat img);

vector<Mat> difference_of_gaussian_pyramid(const vector<Mat>& gaussian_pyramid);

vector<FeaturePoint> find_feature_points(vector<Mat> dogs);

bool is_extremum(const Mat& prev,const Mat& current, const Mat& next,int row,int col);

FeaturePoint generate_feature_point(const vector<Mat>& dogs, int row, int col, int octave, int layer_index);

tuple<float, float, float> update_feature_point(FeaturePoint& fp, const vector<Mat>& dogs);

bool on_edge(FeaturePoint, const const vector<Mat>& dogs);

vector<Mat> generate_gradient_pyramid(const vector<Mat>& gaussian_pyramid);

vector<float> get_orientations(FeaturePoint fp,vector<Mat>& gradient_pyramid);

FeaturePoint compute_keypoint_descriptor(FeaturePoint fp,float orientation, vector<Mat>& gradient_pyramid);

void update_histograms(float hist[SIFT_N_HIST][SIFT_N_HIST][SIFT_N_ORI], float x, float y, float contrib, float theta_mn, float lambda_desc);

vector<uint8_t> hists_to_vec(float histograms[SIFT_N_HIST][SIFT_N_HIST][SIFT_N_ORI]);

Mat draw_keypoints(const Mat& target,vector<FeaturePoint>& fps, int size);

vector<MatchInfo> match(Mat img1, vector<FeaturePoint> fp1, Mat img2, vector<FeaturePoint> fp2);

#endif