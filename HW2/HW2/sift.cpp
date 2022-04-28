#include "sift.h"

auto get_pixel = [&](const Mat& mat, int row, int col) -> float {
	if (col < 0) { col = 0; }
	else if (col >= mat.cols) { col = mat.cols - 1; }

	if (row < 0) { row = 0; }

	else if (row >= mat.rows) { row = mat.rows - 1; }

	return mat.at<float>(row, col);
};

auto get_pixel_v2f = [&](const Mat& mat, int row, int col) -> Vec2f {
	if (col < 0) { col = 0; }
	else if (col >= mat.cols) { col = mat.cols - 1; }

	if (row < 0) { row = 0; }

	else if (row >= mat.rows) { row = mat.rows - 1; }

	return mat.at<Vec2f>(row, col);
};

vector<FeaturePoint> SIFT(Mat img) {
	vector<FeaturePoint> result;
	Mat dst;
	vector<Mat> gaussian_pyramid = get_gaussian_pyramid(img);
	cout << "gaussian_pyramid.octave.size: " << gaussian_pyramid.size() << endl;

	vector<Mat> dogs = difference_of_gaussian_pyramid(gaussian_pyramid);

	cout << "dogs.octave.size: " << dogs.size() << endl;

	//for (int i = 0; i < dogs.size(); i++) {
	//	
	//	Mat tmp;
	//	dogs[i].convertTo(tmp,CV_8UC1,255);
	//	imwrite(to_string(i) + ".jpg", tmp);
	//}
	
	//for (int i = 0; i < gaussian_pyramid.size(); i++) {
	//	imshow(to_string(i), gaussian_pyramid[i]);
	//}
	//for (int i = 0; i < dogs.size(); i++) {
	//	imshow(to_string(i), dogs[i]);
	//}

	//return result;

	vector<FeaturePoint> feature_points = find_feature_points(dogs);

	cout << "feature_points.size: " << feature_points.size() << endl;


	vector<Mat> gradient_pyramid = generate_gradient_pyramid(gaussian_pyramid);

	for (int i = 0; i < feature_points.size(); i++) {
		vector<float> orientations = get_orientations(feature_points[i], gradient_pyramid);
		for (int j = 0; j < orientations.size(); j++) {
			
			result.push_back(compute_keypoint_descriptor(feature_points[i], orientations[j], gradient_pyramid));
		}
	}
	imshow("result",draw_keypoints(img,result,3));
	return result;
}

Mat m_gaussian_blur(const Mat& img, float sigma)
{

	int size = std::ceil(6 * sigma);
	if (size % 2 == 0)
		size++;
	int center = size / 2;
	Mat kernel = Mat::zeros(1, size, CV_32FC1);
	float sum = 0;
	for (int k = -size / 2; k <= size / 2; k++) {
		float val = std::exp(-(k * k) / (2 * sigma * sigma));
		kernel.at<float>(0, center + k) = val;
		sum += val;
	}
	for (int k = -size / 2; k <= size / 2; k++) {
		kernel.at<float>(0, center + k) /= sum;
	}

	Mat tmp = Mat::zeros(img.size(), CV_32FC1);
	Mat filtered = Mat::zeros(img.size(), CV_32FC1);

	// convolve vertical
	for (int x = 0; x < img.cols; x++) {
		for (int y = 0; y < img.rows; y++) {
			float sum = 0;
			for (int k = 0; k < size; k++) {
				int dy = -center + k;
				sum += get_pixel(img,y + dy, x) * kernel.at<float>(0, k);
			}
			tmp.at<float>(y,x) = sum;
		}
	}
	// convolve horizontal
	for (int x = 0; x < img.cols; x++) {
		for (int y = 0; y < img.rows; y++) {
			float sum = 0;
			for (int k = 0; k < size; k++) {
				int dx = -center + k;
				sum += get_pixel(img,y, x + dx) * kernel.at<float>(0, k);
			}
			filtered.at<float>(y, x) = sum;
		}
	}
	return filtered;
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


	//vector<Mat> result(SIFT_N_OCTAVE * SIFT_INTVLS); //計算高斯金字塔總layer數(幾個octave * 每個octave幾個layer)
	//vector<double> sigmas(SIFT_INTVLS);
	//int result_index = 0;
	//double k = pow(2.0f, 1.0f / (double)(SIFT_INTVLS - 3));//-3不明，似乎是配合下面的sigma計算
	//int img_row = img.rows * 2;
	//int img_col = img.cols * 2;
	//
	//for (int i = 0; i < SIFT_N_OCTAVE;i++) {
	//	double total_sigma = SIFT_SIGMA_MIN;
	//	double pre_total_sigma = total_sigma;
	//	Mat octave_base;
	//	if (i != 0) {//每層octave的第一張圖不用GaussianBlur，因為縮小就具有模糊的效果
	//		img_row /= 2.0f;
	//		img_col /= 2.0f;
	//		resize(img, octave_base, Size(img_col, img_row), 0.0f, 0.0f, INTER_LINEAR);
	//		result[result_index] = octave_base;
	//		result_index++;
	//	}
	//	else { //first image 第一層octave的第一張圖
	//		resize(img, octave_base, Size(img_col, img_row), 0.0f, 0.0f, INTER_LINEAR);
	//		double sigma_diff = sqrt((total_sigma * total_sigma) / (0.25f) - 1.0f);//不明
	//		GaussianBlur(octave_base, octave_base, Size(3, 3), sigma_diff, sigma_diff);
	//		result[result_index] = octave_base;
	//		result_index++;
	//	}

	//	for (int j = 1; j < SIFT_INTVLS; j++) { //每層octave第一張後的圖
	//		total_sigma *= k;
	//		double sigma_diff = sqrt(total_sigma * total_sigma - pre_total_sigma * pre_total_sigma);//不明
	//		//cout << sigma_diff << endl;
	//		GaussianBlur(octave_base, result[result_index],Size(3,3), sigma_diff, sigma_diff);
	//		result_index++;
	//		pre_total_sigma = total_sigma;
	//	}
	//}
	//return result;

	vector<Mat> result(SIFT_N_OCTAVE * SIFT_INTVLS);
	float base_sigma = SIFT_SIGMA_MIN / SIFT_MIN_PIX_DIST;
	Mat base_img;
	resize(img, base_img, Size(img.cols * 2, img.rows * 2), 0.0f, 0.0f, INTER_LINEAR);
	//Image base_img = img.resize(img.width * 2, img.height * 2, Interpolation::BILINEAR);
	float sigma_diff = std::sqrt(base_sigma * base_sigma - 1.0f);
	GaussianBlur(base_img, base_img,Size(), sigma_diff, sigma_diff);

	int imgs_per_octave = 6;

	// determine sigma values for bluring
	float k = std::pow(2, 1.0 / 3.0f);
	std::vector<float> sigma_vals{ base_sigma };
	for (int i = 1; i < imgs_per_octave; i++) {
		float sigma_prev = base_sigma * std::pow(k, i - 1);
		float sigma_total = k * sigma_prev;
		sigma_vals.push_back(std::sqrt(sigma_total * sigma_total - sigma_prev * sigma_prev));
	}
	int index = 0;
	for (int i = 0; i < SIFT_N_OCTAVE; i++) {
		//cout << "index " << index << endl;
		int octave_base = index;
		result[index] = base_img;
		index++;
		for (int j = 1; j < sigma_vals.size(); j++) {
			GaussianBlur(result[index - 1], result[index], Size(), sigma_vals[j], sigma_vals[j]);
			//result[index] = m_gaussian_blur(result[index - 1], sigma_vals[j]);
			index++;
		}
		resize(result[octave_base + 3], base_img,Size(base_img.cols / 2, base_img.rows/2), 0.0f, 0.0f, INTER_NEAREST);
	}
	return result;
}

vector<Mat> difference_of_gaussian_pyramid(const vector<Mat>& gaussian_pyramid) {
	vector<Mat> result(SIFT_N_OCTAVE * (SIFT_INTVLS - 1));//計算DOG金字塔總layer數(幾個octave * (兩個高斯圖相減，因此會少一個))	
	int result_index = 0;
	for (int i = 0; i < SIFT_N_OCTAVE; i++) {
		for (int j = 1; j < SIFT_INTVLS; j++) {
			int src_index = i * SIFT_INTVLS + j;
			subtract(gaussian_pyramid[src_index-1], gaussian_pyramid[src_index], result[result_index], noArray(), CV_32FC1);
			result_index++;
		}
	}
	return result;
}

vector<FeaturePoint> find_feature_points(vector<Mat> dogs) {
	int count = 0;
	vector<FeaturePoint> result;
	int threshold = cvFloor(0.5f * SIFT_CONTR_THR / 3.0f * 255);
	for (int i = 0; i < SIFT_N_OCTAVE; i++) {		
		for (int j = 1; j < (SIFT_INTVLS - 1) - 1; j++) {//會需要前後兩個layer的資訊，因此第一個跟最後一個layer不跑
			Mat& prev = dogs[i * (SIFT_INTVLS - 1) + j - 1];
			Mat& current = dogs[i * (SIFT_INTVLS - 1) + j];
			Mat& next = dogs[i * (SIFT_INTVLS - 1) + j + 1];
			int img_row = current.rows;
			int img_col = current.cols;
			//因為要3*3區域，因此第一個與最後一個像素不做
			for (int row = 1; row < img_row - 1; row++) {
				//因為要3*3區域，因此第一個與最後一個像素不做
				for (int col = 1; col < img_col - 1; col++) {
					//cout << current.at<float>(row, col) << endl;
					if (abs(current.at<float>(row, col)) < threshold) {
						count++;
						continue;
					}
					if (is_extremum(prev,current,next,row,col)) {
						
						//產生FeaturePoint，並檢測是否valid
						FeaturePoint fp = generate_feature_point(dogs, row, col, i, j);
						if (fp.valid) {
							result.push_back(fp);
						}
					}
				}
			}
		}
	}
	cout << "count: " << count << endl;
	return result;
}

//檢查該點在三層3*3的像素中，是否為最大或最小值(極端)
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
		//cout << row << " " << col << endl;
		tuple<float, float, float> offsets = update_feature_point(result, dogs);
		float max_offset = max({ abs(get<0>(offsets)), abs(get<1>(offsets)), abs(get<2>(offsets)) });
		//result.layer_index += round(get<0>(offsets));
		//result.col += round(get<1>(offsets));
		//result.row += round(get<2>(offsets));
		if (result.layer_index >= (SIFT_INTVLS - 2) || result.layer_index < 1) break;
		if (max_offset < 0.6f && abs(result.extremum_val) > SIFT_CONTR_THR && !on_edge(result, dogs)) {
			result.sigma = std::pow(2, result.octave) * SIFT_SIGMA_MIN * std::pow(2, (abs(get<0>(offsets)) + result.layer_index) / SIFT_N_SPO);
			result.x = SIFT_MIN_PIX_DIST * std::pow(2, result.octave) * (abs(get<1>(offsets)) + result.col);
			result.y = SIFT_MIN_PIX_DIST * std::pow(2, result.octave) * (abs(get<2>(offsets)) + result.row);
			result.valid = true;
			break;
		}
	}
	return result;
}




/// 待修改//////////////////////////////////////////////
tuple<float, float, float> update_feature_point(FeaturePoint& fp, const vector<Mat>& dogs) {
	float g1, g2, g3;
	float h11, h12, h13, h22, h23, h33;
	int x = fp.col, y = fp.row;

	int base_index = fp.octave * (SIFT_INTVLS - 1) + fp.layer_index;
	const Mat& prev = dogs[base_index - 1];
	const Mat& current = dogs[base_index];
	const Mat& next = dogs[base_index + 1];

	// gradient 
	g1 = (get_pixel(next,y, x) - get_pixel(prev,y, x)) * 0.5;
	g2 = (get_pixel(current,y, x + 1) - get_pixel(current,y, x - 1)) * 0.5;
	g3 = (get_pixel(current,y + 1, x) - get_pixel(current,y - 1, x)) * 0.5;

	// hessian
	h11 = get_pixel(next,y, x) + get_pixel(prev,y, x) - 2 * get_pixel(current,y, x);
	h22 = get_pixel(current,y, x + 1) + get_pixel(current,y, x - 1) - 2 * get_pixel(current,y, x);
	h33 = get_pixel(current,y + 1, x) + get_pixel(current,y - 1, x) - 2 * get_pixel(current,y, x);
	h12 = (get_pixel(next,y, x + 1) - get_pixel(next,y, x - 1)
		- get_pixel(prev,y, x + 1) + get_pixel(prev,y, x - 1)) * 0.25;
	h13 = (get_pixel(next,y + 1, x) - get_pixel(next,y - 1, x)
		- get_pixel(prev,y + 1, x) + get_pixel(prev,y - 1, x)) * 0.25;
	h23 = (get_pixel(current,y + 1, x + 1) - get_pixel(current,y - 1, x + 1)
		- get_pixel(current,y + 1, x - 1) + get_pixel(current,y - 1, x - 1)) * 0.25;

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

	float interpolated_extrema_val = get_pixel(current,y, x)
		+ 0.5 * (g1 * offset_s + g2 * offset_x + g3 * offset_y);
	fp.extremum_val = interpolated_extrema_val;
	fp.row += round(offset_y);
	fp.col += round(offset_x);
	fp.layer_index += round(offset_s);
	return make_tuple(offset_s, offset_x, offset_y);
}


/// 待修改//////////////////////////////////////////////
bool on_edge(FeaturePoint fp, const vector<Mat>& dogs) {
	const Mat& img = dogs[fp.layer_index];
	float h11, h12, h22;
	int x = fp.col, y = fp.row;
	h11 = get_pixel(img, y, x + 1) + get_pixel(img, y, x - 1) - 2 * get_pixel(img, y, x);
	h22 = get_pixel(img, y + 1, x) + get_pixel(img, y - 1, x) - 2 * get_pixel(img, y, x);
	h12 = (get_pixel(img, y + 1, x + 1) - get_pixel(img, y - 1, x + 1)
		- get_pixel(img, y + 1, x - 1) + get_pixel(img, y - 1, x - 1)) * 0.25;

	float det_hessian = h11 * h22 - h12 * h12;
	float tr_hessian = h11 + h22;
	float edgeness = tr_hessian * tr_hessian / det_hessian;

	if (edgeness > std::pow(SIFT_C_EDGE + 1, 2) / SIFT_C_EDGE)
		return true;
	else
		return false;
}

vector<Mat> generate_gradient_pyramid(const vector<Mat>& gaussian_pyramid) {
	vector<Mat> result(gaussian_pyramid.size());

	for (int i = 0; i < SIFT_N_OCTAVE; i++) {
		int octave_base_index = i * SIFT_INTVLS;
		int img_row = gaussian_pyramid[octave_base_index].rows;
		int img_col = gaussian_pyramid[octave_base_index].cols;
		for (int j = 0; j < SIFT_INTVLS; j++) {
			int index = octave_base_index + j;
			result[index] = Mat::zeros(Size(img_col,img_row),CV_32FC2);
			
			for (int y = 1; y < img_row - 1; y++) {
				for (int x = 1; x < img_col - 1; x++) {
					float gx, gy;
					gx = (gaussian_pyramid[index].at<float>(y,x + 1)
						- gaussian_pyramid[index].at<float>(y, x - 1)) * 0.5;
					result[index].at<Vec2f>(y, x)[0] = gx;
					gy = (gaussian_pyramid[index].at<float>(y + 1, x)
						- gaussian_pyramid[index].at<float>(y - 1, x)) * 0.5;
					result[index].at<Vec2f>(y, x)[1] = gy;
				}
			}
		}
	}
	return result;
}

vector<float> get_orientations(FeaturePoint fp, vector<Mat>& gradient_pyramid) {
	vector<float> result;
	float pix_dist = SIFT_MIN_PIX_DIST * std::pow(2,fp.octave);
	const Mat& img = gradient_pyramid[fp.octave * SIFT_INTVLS + fp.layer_index];
	int img_row = img.rows;
	int img_col = img.cols;
	float min_dist_from_border = std::min({ (float)fp.x, (float)fp.y, pix_dist * img_col - fp.x,pix_dist * img_row - fp.y });
	if (min_dist_from_border <= std::sqrt(2) * SIFT_LAMBDA_DESC * fp.sigma) {
		return result;
	}

	float hist[SIFT_N_BINS] = { 0.0f};
	int bin;
	float gx, gy, grad_norm, weight, theta;
	float patch_sigma = SIFT_LAMBDA_ORI * fp.sigma;
	float patch_radius = 3 * patch_sigma;
	int x_start = std::round((fp.x - patch_radius) / pix_dist);
	int x_end = std::round((fp.x + patch_radius) / pix_dist);
	int y_start = std::round((fp.y - patch_radius) / pix_dist);
	int y_end = std::round((fp.y + patch_radius) / pix_dist);

	// accumulate gradients in orientation histogram
	for (int x = x_start; x <= x_end; x++) {
		for (int y = y_start; y <= y_end; y++) {
			gx = get_pixel_v2f(img, y, x)[0];
			gy = get_pixel_v2f(img, y, x)[1];
			grad_norm = std::sqrt(gx * gx + gy * gy);
			weight = std::exp(-(std::pow(x * pix_dist - fp.x, 2) + std::pow(y * pix_dist - fp.y, 2))
				/ (2 * patch_sigma * patch_sigma));
			theta = std::fmod(std::atan2(gy, gx) + 2 * M_PI, 2 * M_PI);
			bin = (int)std::round(SIFT_N_BINS / (2 * M_PI) * theta) % SIFT_N_BINS;
			hist[bin] += weight * grad_norm;
		}
	}

	float tmp_hist[SIFT_N_BINS] = { 0.0f };
	//smooth_histogram
	for (int i = 0; i < 6; i++) {
		for (int j = 0; j < SIFT_N_BINS; j++) {
			int prev_idx = (j - 1 + SIFT_N_BINS) % SIFT_N_BINS;
			int next_idx = (j + 1) % SIFT_N_BINS;
			tmp_hist[j] = (hist[prev_idx] + hist[j] + hist[next_idx]) / 3;
		}
		for (int j = 0; j < SIFT_N_BINS; j++) {
			hist[j] = tmp_hist[j];
		}
	}

	// extract reference orientations
	float ori_thresh = 0.8, ori_max = 0;
	for (int j = 0; j < SIFT_N_BINS; j++) {
		if (hist[j] > ori_max) {
			ori_max = hist[j];
		}
	}
	for (int j = 0; j < SIFT_N_BINS; j++) {
		if (hist[j] >= ori_thresh * ori_max) {
			float prev = hist[(j - 1 + SIFT_N_BINS) % SIFT_N_BINS], next = hist[(j + 1) % SIFT_N_BINS];
			if (prev > hist[j] || next > hist[j])
				continue;
			float theta = 2 * M_PI * (j + 1) / SIFT_N_BINS + M_PI / SIFT_N_BINS * (prev - next) / (prev - 2 * hist[j] + next);
			result.push_back(theta);
		}
	}

	return result;
}

FeaturePoint compute_keypoint_descriptor(FeaturePoint fp, float orientation, vector<Mat>& gradient_pyramid) {
	FeaturePoint result = fp;
	float pix_dist = SIFT_MIN_PIX_DIST * std::pow(2, fp.octave);
	const Mat& img_grad = gradient_pyramid[fp.octave * SIFT_INTVLS + fp.layer_index];
	float histograms[SIFT_N_HIST][SIFT_N_HIST][SIFT_N_ORI] = { 0 };

	//find start and end coords for loops over image patch
	float half_size = std::sqrt(2) * SIFT_LAMBDA_DESC * fp.sigma * (SIFT_N_HIST + 1.) / SIFT_N_HIST;
	int x_start = std::round((fp.x - half_size) / pix_dist);
	int x_end = std::round((fp.x + half_size) / pix_dist);
	int y_start = std::round((fp.y - half_size) / pix_dist);
	int y_end = std::round((fp.y + half_size) / pix_dist);

	float cos_t = std::cos(orientation), sin_t = std::sin(orientation);
	float patch_sigma = SIFT_LAMBDA_DESC * fp.sigma;
	//accumulate samples into histograms
	for (int m = x_start; m <= x_end; m++) {
		for (int n = y_start; n <= y_end; n++) {
			// find normalized coords w.r.t. kp position and reference orientation
			float x = ((m * pix_dist - fp.x) * cos_t
				+ (n * pix_dist - fp.y) * sin_t) / fp.sigma;
			float y = (-(m * pix_dist - fp.x) * sin_t
				+ (n * pix_dist - fp.y) * cos_t) / fp.sigma;

			// verify (x, y) is inside the description patch
			if (std::max(std::abs(x), std::abs(y)) > SIFT_LAMBDA_DESC * (SIFT_N_HIST + 1.) / SIFT_N_HIST)
				continue;

			float gx = get_pixel_v2f(img_grad, n, m)[0], gy = get_pixel_v2f(img_grad, n, m)[1];
			float theta_mn = std::fmod(std::atan2(gy, gx) - orientation + 4 * M_PI, 2 * M_PI);
			float grad_norm = std::sqrt(gx * gx + gy * gy);
			float weight = std::exp(-(std::pow(m * pix_dist - fp.x, 2) + std::pow(n * pix_dist - fp.y, 2))
				/ (2 * patch_sigma * patch_sigma));
			float contribution = weight * grad_norm;

			update_histograms(histograms, x, y, contribution, theta_mn, SIFT_LAMBDA_DESC);
		}
	}
	result.descriptor = hists_to_vec(histograms);
	return result;
}

void update_histograms(float hist[SIFT_N_HIST][SIFT_N_HIST][SIFT_N_ORI], float x, float y,float contrib, float theta_mn, float lambda_desc)
{
	float x_i, y_j;
	for (int i = 1; i <= SIFT_N_HIST; i++) {
		x_i = (i - (1 + (float)SIFT_N_HIST) / 2) * 2 * lambda_desc / SIFT_N_HIST;
		if (std::abs(x_i - x) > 2 * lambda_desc / SIFT_N_HIST)
			continue;
		for (int j = 1; j <= SIFT_N_HIST; j++) {
			y_j = (j - (1 + (float)SIFT_N_HIST) / 2) * 2 * lambda_desc / SIFT_N_HIST;
			if (std::abs(y_j - y) > 2 * lambda_desc / SIFT_N_HIST)
				continue;

			float hist_weight = (1 - SIFT_N_HIST * 0.5 / lambda_desc * std::abs(x_i - x))
				* (1 - SIFT_N_HIST * 0.5 / lambda_desc * std::abs(y_j - y));

			for (int k = 1; k <= SIFT_N_ORI; k++) {
				float theta_k = 2 * M_PI * (k - 1) / SIFT_N_ORI;
				float theta_diff = std::fmod(theta_k - theta_mn + 2 * M_PI, 2 * M_PI);
				if (std::abs(theta_diff) >= 2 * M_PI / SIFT_N_ORI)
					continue;
				float bin_weight = 1 - SIFT_N_ORI * 0.5 / M_PI * std::abs(theta_diff);
				hist[i - 1][j - 1][k - 1] += hist_weight * bin_weight * contrib;
			}
		}
	}
}


vector<uint8_t> hists_to_vec(float histograms[SIFT_N_HIST][SIFT_N_HIST][SIFT_N_ORI])
{
	vector<uint8_t> result;
	int size = SIFT_N_HIST * SIFT_N_HIST * SIFT_N_ORI;
	float* hist = reinterpret_cast<float*>(histograms);

	float norm = 0;
	for (int i = 0; i < size; i++) {
		norm += hist[i] * hist[i];
	}
	norm = std::sqrt(norm);
	float norm2 = 0;
	for (int i = 0; i < size; i++) {
		hist[i] = std::min(hist[i], 0.2f * norm);
		norm2 += hist[i] * hist[i];
	}
	norm2 = std::sqrt(norm2);
	for (int i = 0; i < size; i++) {
		float val = std::floor(512 * hist[i] / norm2);
		result.push_back(std::min((int)val, 255));
	}
	return result;
}

Mat draw_keypoints(const Mat& target, vector<FeaturePoint>& fps,int size) {
	Mat result = target;
	for (const FeaturePoint& fp : fps) {
		int x = fp.x;
		int y = fp.y;
		for (int i = x - size / 2; i <= x + size / 2; i++) {
			for (int j = y - size / 2; j <= y + size / 2; j++) {
				if (i < 0 || i >= target.cols) continue;
				if (j < 0 || j >= target.rows) continue;
				if (std::abs(i - x) + std::abs(j - y) > size / 2) continue;
				result.at<Vec3b>(j, i)[0] = 0;
				result.at<Vec3b>(j, i)[1] = 0;
				result.at<Vec3b>(j, i)[2] = 255;
			}
		}
	}
	return result;
}