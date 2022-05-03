#ifndef _COMBINE_H_
#define _COMBINE_H_
#include <cmath>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <limits>
#include <utility>
#include "feature_point.h"
int get_right_img_index(const cv::Mat& img, const std::vector<FeaturePoint>& img_fps, int img_count);
std::vector<int> get_image_order(const std::vector<cv::Mat>&,const std::vector<std::vector<FeaturePoint>>&);
std::pair<int, int> get_two_img_move(const std::vector<cv::Mat>&, const std::vector<std::vector<FeaturePoint>>&, int a_index, int b_index);

cv::Mat generateNewImage(std::vector<cv::Mat>& warp_imgs, std::vector<int> img_order, std::vector<std::pair<int, int>> img_moves);
#endif