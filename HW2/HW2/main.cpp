#include<opencv2/opencv.hpp>
#include <ANN/ANN.h>
#include "image_stitch.h"
#include "sift.h"
using namespace cv;
using namespace std;
int main()
{
	vector<string> filenames = { "./Lenna.jpg" ,"./Lenna_rotate_scale.png","./Lenna_rotate_scale2.png" };
	//Mat img1 = imread("./Lenna.jpg");
	//Mat img2 = imread("./Lenna_rotate_scale.png");
	//Mat img = imread("./Lenna_rotate_scale.png");
	//Mat img = imread("./Lenna_rotate_scale1.png");
	//GaussianBlur(img, img, Size(3, 3), 1.6f, 1.6f);
	//vector<FeaturePoint> fps1 = SIFT(img1);
	//vector<FeaturePoint> fps2 = SIFT(img2);

	//std::vector<std::pair<int, int>> matches = find_keypoint_matches(fps1, fps2);
	//Mat result = draw_matches(img1, img2, fps1, fps2, matches);
	//imshow("match", result);

	image_stitch(filenames);
	cout << "complete" << endl;
	waitKey(0);
	return 0;
}