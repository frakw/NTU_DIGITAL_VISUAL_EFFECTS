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


	vector<Mat> result(SIFT_N_OCTAVE * SIFT_INTVLS); //�p�Ⱚ�����r���`layer��(�X��octave * �C��octave�X��layer)
	vector<double> sigmas(SIFT_INTVLS);
	int result_index = 0;
	double k = pow(2.0f, 1.0f / (double)(SIFT_INTVLS - 3));//-3�����A���G�O�t�X�U����sigma�p��
	int img_row = img.rows;
	int img_col = img.cols;
	
	for (int i = 0; i < SIFT_N_OCTAVE;i++) {
		double total_sigma = 1.6f;
		double pre_total_sigma = 1.6f;
		Mat octave_base;
		if (i != 0) {//�C�hoctave���Ĥ@�i�Ϥ���GaussianBlur�A�]���Y�p�N�㦳�ҽk���ĪG
			img_row /= 2.0f;
			img_col /= 2.0f;
			resize(img, octave_base, Size(img_col, img_row), 0.0, 0.0, INTER_NEAREST);
			result[result_index] = octave_base;
			result_index++;
		}
		else { //first image �Ĥ@�hoctave���Ĥ@�i��
			//�Y�p�A��j�Φ��ҽk�ĪG
			resize(img, octave_base, Size(img_col * 0.5f, img_row * 0.5f));
			resize(octave_base, octave_base, Size(img_col, img_row));
			result[result_index] = octave_base;
			result_index++;
		}

		for (int j = 1; j < SIFT_INTVLS; j++) { //�C�hoctave�Ĥ@�i�᪺��
			total_sigma *= k;
			double sigma_diff = total_sigma * total_sigma - pre_total_sigma - pre_total_sigma;//����
			cout << sigma_diff << endl;
			GaussianBlur(octave_base, result[result_index],Size(3,3), sigma_diff, sigma_diff);
			result_index++;
			pre_total_sigma = total_sigma;
		}
	}
	return result;
}

vector<Mat> difference_of_gaussian_pyramid(const vector<Mat>& gaussian_pyramid) {
	vector<Mat> result(SIFT_N_OCTAVE * (SIFT_INTVLS - 1));//�p��DOG���r���`layer��(�X��octave * (��Ӱ����Ϭ۴�A�]���|�֤@��))	
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