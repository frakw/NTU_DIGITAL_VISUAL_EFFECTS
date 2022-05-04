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
#include "feature_point.h"


//defines
#define SIFT_SIGMA 1.6f
#define SIFT_INIT_SIGMA 0.5f
//每層octave有幾個layer
#define SIFT_INTERVALS 3
#define SIFT_LAYER_PER_OCT  (SIFT_INTERVALS + 3)
#define SIFT_DOG_LAYER_PER_OCT  (SIFT_LAYER_PER_OCT - 1)
#define SIFT_C_EDGE 10
#define SIFT_CONTR_THR 0.03f
#define SIFT_IMG_BORDER 5
#define SIFT_N_BINS  36
#define parabola_interpolate(l, c, r) (0.5*((l)-(r))/((l)-2.0*(c)+(r))) 
#define SIFT_ORI_SMOOTH_TIMES 2
#define SIFT_DESCR_SCALE_ADJUST 3
#define SIFT_DESCR_MAG_THR 0.2f
#define SIFT_ORI_PEAK_RATIO 0.8f
#define SIFT_ORI_SIG_FCTR 1.5f
#define SIFT_ORI_RADIUS  (3 * SIFT_ORI_SIG_FCTR)
#define SIFT_INT_DESCR_FCTR 512.f
#define SIFT_LAMBDA_ORI  1.5f
//


std::vector<FeaturePoint> SIFT(cv::Mat img);

void match_feature_points(std::vector< std::vector<FeaturePoint> >& img_fps_list);

#endif