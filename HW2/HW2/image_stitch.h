#ifndef _IMAGE_STITCH_H_
#define _IMAGE_STITCH_H_
#include <iostream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include "sift.h"
#include "warping.h"
#include "combine.h"

cv::Mat image_stitch(std::vector<std::string> filenames, int limit_size = 500);
#endif // !_IMAGE_STITCH_
