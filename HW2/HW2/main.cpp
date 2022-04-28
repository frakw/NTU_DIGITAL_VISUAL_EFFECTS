#include<opencv2/opencv.hpp>
#include "image_stitch.h"
#include "sift.h"
using namespace cv;
using namespace std;
int main()
{
	//Mat img = imread("./Lenna.jpg");
	Mat img = imread("./Lenna_rotate_scale.png");
	//Mat img = imread("./Lenna_rotate_scale1.png");
	//GaussianBlur(img, img, Size(3, 3), 1.6f, 1.6f);
	SIFT(img);
	waitKey(0);
	return 0;
}