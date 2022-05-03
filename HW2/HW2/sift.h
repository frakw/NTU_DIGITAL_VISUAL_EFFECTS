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


std::vector<FeaturePoint> SIFT(cv::Mat img);

std::vector<std::pair<int, int>> find_keypoint_matches(std::vector<FeaturePoint>& a,
	std::vector<FeaturePoint>& b,
	float thresh_relative = 0.7f,
	float thresh_absolute = 350.0f);

cv::Mat draw_keypoints(const cv::Mat& target, std::vector<FeaturePoint>& fps, int size);

cv::Mat draw_matches(const cv::Mat& a, const cv::Mat& b, std::vector<FeaturePoint>& kps_a,
	std::vector<FeaturePoint>& kps_b);

cv::Mat draw_matches2(const cv::Mat& a, const cv::Mat& b, std::vector<FeaturePoint>& kps_a,
	std::vector<FeaturePoint>& kps_b);

void featureMatch(std::vector< std::vector<FeaturePoint> >& img_fps_list);

#endif