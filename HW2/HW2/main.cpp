#include<opencv2/opencv.hpp>
#include "image_stitch.h"
#include "sift.h"
using namespace cv;
using namespace std;
int main()
{
	Mat img = imread("./Lenna.jpg");
	//GaussianBlur(img, img, Size(3, 3), 1.6f, 1.6f);
	SIFT(img);
	waitKey(0);
	return 0;
}