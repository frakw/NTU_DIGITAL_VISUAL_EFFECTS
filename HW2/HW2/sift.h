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
#include <ANN/ANN.h>
#include "sift_define.h"
#include "feature_point.h"
#include "match_info.h"
using namespace std;
using namespace cv;

vector<FeaturePoint> SIFT(Mat img);

std::vector<std::pair<int, int>> find_keypoint_matches(std::vector<FeaturePoint>& a,
	std::vector<FeaturePoint>& b,
	float thresh_relative = 0.7f,
	float thresh_absolute = 350.0f);

Mat draw_matches(const Mat& a, const Mat& b, std::vector<FeaturePoint>& kps_a,
	std::vector<FeaturePoint>& kps_b);

void featureMatch(vector< vector<FeaturePoint> >& img_fps_list);

#endif