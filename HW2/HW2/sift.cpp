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
	int img_row = img.rows * 2;
	int img_col = img.cols * 2;
	
	for (int i = 0; i < SIFT_N_OCTAVE;i++) {
		double total_sigma = SIFT_SIGMA_MIN;
		double pre_total_sigma = total_sigma;
		Mat octave_base;
		if (i != 0) {//�C�hoctave���Ĥ@�i�Ϥ���GaussianBlur�A�]���Y�p�N�㦳�ҽk���ĪG
			img_row /= 2.0f;
			img_col /= 2.0f;
			resize(img, octave_base, Size(img_col, img_row), 0.0f, 0.0f, INTER_LINEAR);
			result[result_index] = octave_base;
			result_index++;
		}
		else { //first image �Ĥ@�hoctave���Ĥ@�i��
			resize(img, octave_base, Size(img_col, img_row), 0.0f, 0.0f, INTER_LINEAR);
			double sigma_diff = sqrt((total_sigma * total_sigma) / (0.25f) - 1.0f);//����
			GaussianBlur(octave_base, octave_base, Size(3, 3), sigma_diff, sigma_diff);
			result[result_index] = octave_base;
			result_index++;
		}

		for (int j = 1; j < SIFT_INTVLS; j++) { //�C�hoctave�Ĥ@�i�᪺��
			total_sigma *= k;
			double sigma_diff = sqrt(total_sigma * total_sigma - pre_total_sigma * pre_total_sigma);//����
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

vector<FeaturePoint> find_feature_points(vector<Mat> dogs) {
	vector<FeaturePoint> result;
	for (int i = 0; i < SIFT_N_OCTAVE; i++) {		
		for (int j = 1; j < SIFT_INTVLS - 1; j++) {//�|�ݭn�e����layer����T�A�]���Ĥ@�Ӹ�̫�@��layer���]
			Mat& prev = dogs[i * (SIFT_INTVLS - 1) + j - 1];
			Mat& current = dogs[i * (SIFT_INTVLS - 1) + j];
			Mat& next = dogs[i * (SIFT_INTVLS - 1) + j + 1];
			int img_row = current.rows;
			int img_col = current.cols;
			//�]���n3*3�ϰ�A�]���Ĥ@�ӻP�̫�@�ӹ�������
			for (int row = 1; row < img_row - 1; row++) {
				//�]���n3*3�ϰ�A�]���Ĥ@�ӻP�̫�@�ӹ�������
				for (int col = 1; col < img_col - 1; col++) {
					if (abs(current.at<float>(row, col)) < (0.8f * SIFT_C_DOG)) continue;
					if (is_extremum(prev,current,next,row,col)) {
						//����FeaturePoint�A���˴��O�_valid
						FeaturePoint fp = generate_feature_point(dogs, row, col, i, j);
						if (fp.valid) {
							result.push_back(fp);
						}
					}
				}
			}
		}
	}
	return result;
}

//�ˬd���I�b�T�h3*3���������A�O�_���̤j�γ̤p��(����)
bool is_extremum(const Mat& prev, const Mat& current, const Mat& next, int row, int col) {
	vector<int> all_vals;
	int check_val = current.at<float>(row, col);
	for (int i = -1; i <= 1; i++) {
		for (int j = -1; j <= 1; j++) {
			int r = row + i;
			int c = col + j;
			all_vals.push_back(prev.at<float>(r, c));
			all_vals.push_back(current.at<float>(r, c));
			all_vals.push_back(next.at<float>(r, c));
		}
	}
	sort(all_vals.begin(), all_vals.end());
	return (check_val == all_vals.front()) || (check_val == all_vals.back());
}


FeaturePoint generate_feature_point(const vector<Mat>& dogs, int row, int col, int octave, int layer_index) {
	FeaturePoint result(row, col, octave, layer_index);
	for (int i = 0; i < SIFT_MAX_REFINE_ITER; i++) {
		tuple<float, float, float> offsets = update_feature_point(result, dogs);
		float max_offset = max(abs(get<0>(offsets)), abs(get<1>(offsets)), abs(get<2>(offsets)));
		if (result.layer_index >= (SIFT_INTVLS - 1) || result.layer_index < 1) break;
		if (max_offset < 0.6f && abs(result.extremum_val) > SIFT_C_DOG && !on_edge(result, dogs)) {			
			result.sigma = std::pow(2, result.octave) * SIFT_SIGMA_MIN * std::pow(2, (abs(get<0>(offsets)) + result.layer_index) / SIFT_N_SPO);
			result.x = SIFT_MIN_PIX_DIST * std::pow(2, result.octave) * (abs(get<1>(offsets)) + result.col);
			result.y = SIFT_MIN_PIX_DIST * std::pow(2, result.octave) * (abs(get<2>(offsets)) + result.row);
			result.valid = true;
			break;
		}
	}
	return result;
}


/// �ݭק�//////////////////////////////////////////////
tuple<float, float, float> update_feature_point(FeaturePoint& fp, const vector<Mat>& dogs) {
	float g1, g2, g3;
	float h11, h12, h13, h22, h23, h33;
	int x = fp.col, y = fp.row;

	const Mat& prev = dogs[fp.octave * (SIFT_INTVLS - 1) + fp.layer_index - 1];
	const Mat& current = dogs[fp.octave * (SIFT_INTVLS - 1)];
	const Mat& next = dogs[fp.octave * (SIFT_INTVLS - 1) + fp.layer_index + 1];

	// gradient 
	g1 = (current.at<float>(x, y) - prev.at<float>(x, y)) * 0.5;
	g2 = (current.at<float>(x + 1, y) - current.at<float>(x - 1, y)) * 0.5;
	g3 = (current.at<float>(x, y + 1) - current.at<float>(x, y - 1)) * 0.5;

	// hessian
	h11 = next.at<float>(x, y) + prev.at<float>(x, y, 0) - 2 * current.at<float>(x, y);
	h22 = current.at<float>(x + 1, y) + current.at<float>(x - 1, y) - 2 * current.at<float>(x, y);
	h33 = current.at<float>(x, y + 1) + current.at<float>(x, y - 1) - 2 * current.at<float>(x, y);
	h12 = (next.at<float>(x + 1, y) - next.at<float>(x - 1, y)
		- prev.at<float>(x + 1, y) + prev.at<float>(x - 1, y)) * 0.25;
	h13 = (next.at<float>(x, y + 1) - next.at<float>(x, y - 1)
		- prev.at<float>(x, y + 1) + prev.at<float>(x, y - 1)) * 0.25;
	h23 = (current.at<float>(x + 1, y + 1) - current.at<float>(x + 1, y - 1)
		- current.at<float>(x - 1, y + 1) + current.at<float>(x - 1, y - 1)) * 0.25;

	// invert hessian
	float hinv11, hinv12, hinv13, hinv22, hinv23, hinv33;
	float det = h11 * h22 * h33 - h11 * h23 * h23 - h12 * h12 * h33 + 2 * h12 * h13 * h23 - h13 * h13 * h22;
	hinv11 = (h22 * h33 - h23 * h23) / det;
	hinv12 = (h13 * h23 - h12 * h33) / det;
	hinv13 = (h12 * h23 - h13 * h22) / det;
	hinv22 = (h11 * h33 - h13 * h13) / det;
	hinv23 = (h12 * h13 - h11 * h23) / det;
	hinv33 = (h11 * h22 - h12 * h12) / det;

	// find offsets of the interpolated extremum from the discrete extremum
	float offset_s = -hinv11 * g1 - hinv12 * g2 - hinv13 * g3;
	float offset_x = -hinv12 * g1 - hinv22 * g2 - hinv23 * g3;
	float offset_y = -hinv13 * g1 - hinv23 * g3 - hinv33 * g3;

	float interpolated_extrema_val = current.at<float>(x, y)
		+ 0.5 * (g1 * offset_s + g2 * offset_x + g3 * offset_y);
	fp.extremum_val = interpolated_extrema_val;
	fp.row = round(offset_y);
	fp.col = round(offset_x);
	fp.layer_index = round(offset_s);
	return make_tuple(offset_s, offset_x, offset_y);
}


/// �ݭק�//////////////////////////////////////////////
bool on_edge(FeaturePoint fp, const vector<Mat>& dogs) {
	const Mat& img = dogs[fp.layer_index];
	float h11, h12, h22;
	int x = fp.col, y = fp.row;
	h11 = img.at<float>(x + 1, y) + img.at<float>(x - 1, y) - 2 * img.at<float>(x, y);
	h22 = img.at<float>(x, y + 1) + img.at<float>(x, y - 1) - 2 * img.at<float>(x, y);
	h12 = (img.at<float>(x + 1, y + 1) - img.at<float>(x + 1, y - 1)
		- img.at<float>(x - 1, y + 1) + img.at<float>(x - 1, y - 1)) * 0.25;

	float det_hessian = h11 * h22 - h12 * h12;
	float tr_hessian = h11 + h22;
	float edgeness = tr_hessian * tr_hessian / det_hessian;

	if (edgeness > std::pow(SIFT_C_EDGE + 1, 2) / SIFT_C_EDGE)
		return true;
	else
		return false;
}