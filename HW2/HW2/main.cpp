#define _CRT_SECURE_NO_WARNINGS
#include<opencv2/opencv.hpp>
#include <ANN/ANN.h>
#include "image_stitch.h"
#include "sift.h"
#include "warping.h"
#include "exif.h"
using namespace cv;
using namespace std;
int main()
{
	//FILE* fp = fopen("./square00.jpg", "rb");
	//fseek(fp, 0, SEEK_END);
	//unsigned long fsize = ftell(fp);
	//rewind(fp);
	//unsigned char* buf = new unsigned char[fsize];
	//if (fread(buf, 1, fsize, fp) != fsize) {
	//	printf("Can't read file.\n");
	//	delete[] buf;
	//	return -2;
	//}
	//fclose(fp);

	//// Parse EXIF
	//easyexif::EXIFInfo result;
	//int code = result.parseFrom(buf, fsize);
	//delete[] buf;
	//if (code) {
	//	printf("Error parsing EXIF: code %d\n", code);
	//	return -3;
	//}
	//cout << "focal len: " << result.FocalLengthIn35mm << endl;
	Mat img1 = imread("./Lenna.jpg");
	double f = ((int)img1.rows / 10) * 10;
	vector<FeaturePoint> tmp;
	imshow("warping",cylindrical_warping(img1, tmp, 1008.632f));
	waitKey(0);
	return 0;

	vector<string> filenames = { "./Lenna.jpg" ,"./P_20220430_174302.jpg"/*,"./Lenna_rotate_scale2.png" */};
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