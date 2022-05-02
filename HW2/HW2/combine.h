#ifndef _COMBINE_H_
#define _COMBINE_H_
#include <cmath>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <limits>
#include <utility>
#include "feature_point.h"
using namespace std;
using namespace cv;
int get_right_img_index(const Mat& img, const vector<FeaturePoint>& img_fps, int img_count);
vector<int> get_image_order(const vector<Mat>&,const vector<vector<FeaturePoint>>&);
pair<int, int> get_two_img_move(const vector<Mat>&, const vector<vector<FeaturePoint>>&, int a_index, int b_index);

Mat generateNewImage(vector<Mat>& warp_imgs, vector<int> img_order, vector<pair<int, int>> img_moves);
#endif