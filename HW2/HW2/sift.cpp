#include "sift.h"


vector<FeaturePoint> SIFT(Mat img) {
	vector<FeaturePoint> result;
	Mat dst;
	//vector<Mat> gaussianPyr, dogPyr;
	//mysift::CreateInitialImage(img, dst, SIFT_IMG_DBL);
	//int firstOctave = SIFT_IMG_DBL ? -1 : 0;
	//int nOctaves = (int)(log((double)std::min(img.cols, img.rows)) / log(2.) - 2 - firstOctave);
	//mysift::BuildGaussianPyramid(dst, gaussianPyr, nOctaves);
	vector<Mat> gaussian_pyramid = get_gaussian_pyramid(img);
	vector<Mat> dogs = difference_of_gaussian_pyramid(gaussian_pyramid);
	//for (int i = 0; i < gaussian_pyramid.size(); i++) {
	//	imshow(to_string(i), gaussian_pyramid[i]);
	//}
	for (int i = 0; i < dogs.size(); i++) {
		imshow(to_string(i), dogs[i]);
	}
	return result;
}

vector<Mat> get_gaussian_pyramid(Mat img) {
	if (img.type() == CV_8UC3)
	{
		cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
	}

	if (img.type() == CV_8UC1)
	{
		img.convertTo(img, CV_32FC1);
	}


	vector<Mat> result(SIFT_N_OCTAVE * SIFT_INTVLS); //計算高斯金字塔總layer數(幾個octave * 每個octave幾個layer)
	vector<double> sigmas(SIFT_INTVLS);
	int result_index = 0;
	double k = pow(2.0f, 1.0f / (double)(SIFT_INTVLS - 3));//-3不明，似乎是配合下面的sigma計算
	int img_row = img.rows;
	int img_col = img.cols;
	
	for (int i = 0; i < SIFT_N_OCTAVE;i++) {
		double total_sigma = 1.6f;
		double pre_total_sigma = 1.6f;
		Mat octave_base;
		if (i != 0) {//每層octave的第一張圖不用GaussianBlur，因為縮小就具有模糊的效果
			img_row /= 2.0f;
			img_col /= 2.0f;
			resize(img, octave_base, Size(img_col, img_row), 0.0, 0.0, INTER_NEAREST);
			result[result_index] = octave_base;
			result_index++;
		}
		else { //first image 第一層octave的第一張圖
			//縮小再放大形成模糊效果
			resize(img, octave_base, Size(img_col * 0.5f, img_row * 0.5f));
			resize(octave_base, octave_base, Size(img_col, img_row));
			result[result_index] = octave_base;
			result_index++;
		}

		for (int j = 1; j < SIFT_INTVLS; j++) { //每層octave第一張後的圖
			total_sigma *= k;
			double sigma_diff = total_sigma * total_sigma - pre_total_sigma - pre_total_sigma;//不明
			cout << sigma_diff << endl;
			GaussianBlur(octave_base, result[result_index],Size(3,3), sigma_diff, sigma_diff);
			result_index++;
			pre_total_sigma = total_sigma;
		}
	}
	return result;
}

vector<Mat> difference_of_gaussian_pyramid(const vector<Mat>& gaussian_pyramid) {
	vector<Mat> result(SIFT_N_OCTAVE * (SIFT_INTVLS - 1));//計算DOG金字塔總layer數(幾個octave * (兩個高斯圖相減，因此會少一個))	
	int result_index = 0;
	for (int i = 0; i < SIFT_N_OCTAVE; i++) {
		for (int j = 1; j < SIFT_INTVLS; j++) {
			int src_index = i * SIFT_INTVLS + j;
			subtract(gaussian_pyramid[src_index], gaussian_pyramid[src_index - 1], result[result_index], noArray(), CV_32FC1);
			result_index++;
		}
	}
	return result;
}