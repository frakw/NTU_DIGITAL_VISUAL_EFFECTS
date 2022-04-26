#ifndef _SIFT_H_
#define _SIFT_H_
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <tuple>
#include <utility>
#include <algorithm>
#define _USE_MATH_DEFINES
#include <cmath>
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

vector<MatchInfo> match(Mat img1, vector<FeaturePoint> fp1, Mat img2, vector<FeaturePoint> fp2);

#endif