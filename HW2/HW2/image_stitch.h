#ifndef _IMAGE_STITCH_H_
#define _IMAGE_STITCH_H_
#include <iostream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include "sift.h"
#include "warping.h"
#include "combine.h"
using namespace std;
using namespace cv;

Mat image_stitch(vector<string> filenames);

#endif // !_IMAGE_STITCH_
