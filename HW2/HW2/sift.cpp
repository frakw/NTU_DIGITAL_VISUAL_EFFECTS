#include "sift.h"
using namespace std;
using namespace cv;

auto get_pixel = [&](const Mat& mat, int row, int col) -> float {
	if (col < 0) { col = 0; }
	else if (col >= mat.cols) { col = mat.cols - 1; }

	if (row < 0) { row = 0; }

	else if (row >= mat.rows) { row = mat.rows - 1; }

	return mat.at<double>(row, col);
};

auto get_pixel_v2f = [&](const Mat& mat, int row, int col) -> Vec2f {
	if (col < 0) { col = 0; }
	else if (col >= mat.cols) { col = mat.cols - 1; }

	if (row < 0) { row = 0; }

	else if (row >= mat.rows) { row = mat.rows - 1; }

	return mat.at<Vec2f>(row, col);
};
typedef double pixel_t;

Mat bgr2gray(const Mat& src)
{
	Mat result = Mat::zeros(src.size(),CV_64FC1);
	cv::Size size = src.size();
	for (int j = 0; j < src.cols; j++)
	{
		for (int i = 0; i < src.rows; i++)
		{
			double b = src.at<Vec3b>(i,j)[0] / 255.0f;
			double g = src.at<Vec3b>(i, j)[1] / 255.0f;
			double r = src.at<Vec3b>(i, j)[2] / 255.0f;
			result.at<double>(i, j) = (r + g + b) / 3.0f;
		}
	}
	return result;
}
Mat init_gaussian_pyramid(const Mat& src, double sigma = SIFT_SIGMA)
{
	Mat result;
	cv::Mat gray;      
	cv::Mat up;
	gray = bgr2gray(src);
	resize(gray, up, Size(gray.cols * 2, gray.rows * 2), 0, 0, INTER_LINEAR);
	double  sigma_init = sqrt(sigma * sigma - (SIFT_INIT_SIGMA * 2) * (SIFT_INIT_SIGMA * 2));
	GaussianBlur(up, result, Size(0, 0), sigma_init, sigma_init);
	return result;
}
vector<Mat> get_gaussian_pyramid(const Mat& src, int octaves)
{
	vector<Mat> result(octaves * SIFT_LAYER_PER_OCT);
	int result_index = 0;
	Mat first_img = init_gaussian_pyramid(src);

	vector<double> sigmas(SIFT_LAYER_PER_OCT);
	double k = pow(2.0, 1.0 / SIFT_INTERVALS);
	sigmas[0] = SIFT_SIGMA;
	double sig_prev;
	double sig_total;
	for (int i = 1; i < SIFT_LAYER_PER_OCT; i++)
	{
		sig_prev = pow(k, i - 1) * SIFT_SIGMA;
		sig_total = sig_prev * k;
		sigmas[i] = sqrt(sig_total * sig_total - sig_prev * sig_prev);
	}

	for (int octave = 0; octave < octaves; octave++)
	{
		for (int layer = 0; layer < SIFT_LAYER_PER_OCT; layer++)
		{
			if (octave == 0 && layer == 0)
			{
				first_img.copyTo(result[result_index++]);
			}
			else if (layer == 0)
			{
				int img_col = result[(octave - 1) * SIFT_LAYER_PER_OCT + SIFT_INTERVALS].cols / 2;
				int img_row = result[(octave - 1) * SIFT_LAYER_PER_OCT + SIFT_INTERVALS].rows / 2;
				resize(result[(octave - 1) * SIFT_LAYER_PER_OCT + SIFT_INTERVALS], result[result_index++],Size(img_col,img_row),0,0,INTER_NEAREST);
			}
			else
			{
				GaussianBlur(result[octave * SIFT_LAYER_PER_OCT + layer - 1], result[result_index++], Size(0, 0), sigmas[layer], sigmas[layer]);
			}
		}
	}
	return result;
}
vector<Mat> get_dog_pyramid(const vector<Mat>& gauss_pyr, int octaves, int intervals = SIFT_INTERVALS)
{
	vector<Mat> result(octaves * (SIFT_LAYER_PER_OCT - 1));
	int result_index = 0;
	for (int octave = 0; octave < octaves; octave++)
	{
		for (int layer = 1; layer < SIFT_LAYER_PER_OCT; layer++)
		{
			subtract(gauss_pyr[octave * SIFT_LAYER_PER_OCT + layer], gauss_pyr[octave * SIFT_LAYER_PER_OCT + layer - 1], result[result_index++]);
		}
	}
	return result;
}
bool is_extremum(int x, int y, const vector<Mat>& dogs, int index)
{
	double val = dogs[index].at<double>(y, x);

	if (val > 0)
	{
		for (int i = -1; i <= 1; i++) //prev current next layer
		{
			//3*3範圍
			for (int j = -1; j <= 1; j++)
			{
				for (int k = -1; k <= 1; k++)
				{
					int layer = index + i;
					int row = y + j;
					int col = x + k;
					if (val < dogs[layer].at<double>(row, col))
					{
						return false;
					}
				}
			}
		}
	}
	else
	{
		for (int i = -1; i <= 1; i++) //prev current next layer
		{
			//3*3範圍
			for (int j = -1; j <= 1; j++)
			{
				for (int k = -1; k <= 1; k++)
				{
					int layer = index + i;
					int row = y + j;
					int col = x + k;
					if (val > dogs[layer].at<double>(row, col))
					{
						return false;
					}
				}
			}
		}
	}
	return true;
}
bool on_edge(FeaturePoint fp, const vector<Mat>& dogs, int index) {
	const Mat& img = dogs[index];
	double Dxx, Dyy, Dxy;
	int x = fp.x, y = fp.y;
	Dxx = get_pixel(img, y, x + 1) + get_pixel(img, y, x - 1) - 2 * get_pixel(img, y, x);
	Dyy = get_pixel(img, y + 1, x) + get_pixel(img, y - 1, x) - 2 * get_pixel(img, y, x);
	Dxy = (get_pixel(img, y + 1, x + 1) - get_pixel(img, y - 1, x + 1)
		- get_pixel(img, y + 1, x - 1) + get_pixel(img, y - 1, x - 1)) * 0.25;
	double det_hessian = Dxx * Dyy - Dxy * Dxy;
	double tr_hessian = Dxx + Dyy;
	double edgeness = tr_hessian * tr_hessian / det_hessian;
	if (det_hessian <= 0) return false;
	if (edgeness < ((SIFT_C_EDGE + 1)*(SIFT_C_EDGE + 1) / SIFT_C_EDGE)) return true;
	else return false;
}
array<double,3> derivative_3D(int x, int y, const vector<Mat>& dogs, int index)
{
	array<double, 3> result;
	double Dx = (dogs[index].at<double>(y, x + 1) - dogs[index].at<double>(y, x - 1)) / 2.0;
	double Dy = (dogs[index].at<double>(y + 1, x) - dogs[index].at<double>(y - 1, x)) / 2.0;
	double Ds = (dogs[index + 1].at<double>(y, x) - dogs[index - 1].at<double>(y, x)) / 2.0;
	result[0] = Dx;
	result[1] = Dy;
	result[2] = Ds;
	return result;
}
array<array<double, 3>, 3> hessian_3D(int x, int y, const vector<Mat>& dogs, int index)
{
	array<array<double, 3>, 3> result;
	double val, Dxx, Dyy, Dss, Dxy, Dxs, Dys;

	val = dogs[index].at<double>(y, x);

	Dxx = dogs[index].at<double>(y, x + 1) + dogs[index].at<double>(y, x - 1) - 2 * val;
	Dyy = dogs[index].at<double>(y + 1, x) + dogs[index].at<double>(y - 1, x) - 2 * val;
	Dss = dogs[index + 1].at<double>(y, x) + dogs[index - 1].at<double>(y, x) - 2 * val;

	Dxy = (dogs[index].at<double>(y + 1, x + 1) + dogs[index].at<double>(y - 1, x - 1)
		- dogs[index].at<double>(y - 1, x + 1) - dogs[index].at<double>(y + 1, x - 1)) / 4.0;

	Dxs = (dogs[index + 1].at<double>(y, x + 1) + dogs[index - 1].at<double>(y, x - 1)
		- dogs[index - 1].at<double>(y, x + 1) - dogs[index + 1].at<double>(y, x - 1)) / 4.0;

	Dys = (dogs[index + 1].at<double>(y + 1, x) + dogs[index - 1].at<double>(y - 1, x)
		- dogs[index + 1].at<double>(y - 1, x) - dogs[index - 1].at<double>(y + 1, x)) / 4.0;

	result[0][0] = Dxx;
	result[1][1] = Dyy;
	result[2][2] = Dss;

	result[1][0] = result[0][1] = Dxy;
	result[2][0] = result[0][2] = Dxs;
	result[2][1] = result[1][2] = Dys;
	return result;
}
array<array<double, 3>, 3>  inverse_3D(array<array<double, 3>, 3> H)
{
	array<array<double, 3>, 3> result;

	double A = 
		  H[0][0] * H[1][1] * H[2][2]
		+ H[0][1] * H[1][2] * H[2][0]
		+ H[0][2] * H[1][0] * H[2][1]
		- H[0][0] * H[1][2] * H[2][1]
		- H[0][1] * H[1][0] * H[2][2]
		- H[0][2] * H[1][1] * H[2][0];

	result[0][0] =	(H[1][1] * H[2][2] - H[2][1] * H[1][2]) / A;
	result[0][1] = -(H[0][1] * H[2][2] - H[2][1] * H[0][2]) / A;
	result[0][2] =	(H[0][1] * H[1][2] - H[0][2] * H[1][1]) / A;

	result[1][0] =	(H[1][2] * H[2][0] - H[2][2] * H[1][0]) / A;
	result[1][1] = -(H[0][2] * H[2][0] - H[0][0] * H[2][2]) / A;
	result[1][2] =	(H[0][2] * H[1][0] - H[0][0] * H[1][2]) / A;

	result[2][0] =	(H[1][0] * H[2][1] - H[1][1] * H[2][0]) / A;
	result[2][1] = -(H[0][0] * H[2][1] - H[0][1] * H[2][0]) / A;
	result[2][2] =	(H[0][0] * H[1][1] - H[0][1] * H[1][0]) / A;
	return result;
}
array<double, 3> get_offset(int x, int y, const vector<Mat>& dog_pyr, int index){
	array<array<double, 3>, 3> H = hessian_3D(x, y, dog_pyr, index);
	array<array<double, 3>, 3> H_inv = inverse_3D(H);
	array<double, 3> dx = derivative_3D(x, y, dog_pyr, index);
	array<double, 3> result;
	for (int i = 0; i < 3; i++) {
		result[i] = 0.0f;
		for (int j = 0; j < 3; j++) {
			result[i] -= H_inv[i][j] * dx[j];
		}
	}
	return result;
}

double get_fabs_dx(int x, int y, const vector<Mat>& dogs, int index,const array<double, 3>& offset_x){
	array<double, 3> dx = derivative_3D(x, y, dogs, index);
	double term = 0.0;
	for (int i = 0; i < 3; i++) {
		term += dx[i] * offset_x[i];
	}
	double val = dogs[index].at<double>(y, x);
	return fabs(val + 0.5 * term);
}

FeaturePoint* interploation_extremum(int x, int y, const vector<Mat>& dog_pyr, int index, int octave, int interval){
	array<double, 3> offset;
	const Mat& mat = dog_pyr[index];
	int idx = index;
	int intvl = interval;
	int i = 0;
	while (i < 5){
		offset = get_offset(x, y, dog_pyr, idx);
		if (fabs(offset[0]) < 0.5f && fabs(offset[1]) < 0.5f && fabs(offset[2]) < 0.5f) break;
		x += cvRound(offset[0]);
		y += cvRound(offset[1]);
		interval += cvRound(offset[2]);
		idx = index - intvl + interval;
		if (interval < 1 || interval > 3 || x >= mat.cols - 1 || x < 2 || y >= mat.rows - 1 || y < 2){
			return nullptr;
		}
		i++;
	}
	if (i >= 5) return nullptr;
	if (get_fabs_dx(x, y, dog_pyr, idx, offset) < SIFT_CONTR_THR / 3)return nullptr;
	FeaturePoint* fp = new FeaturePoint(x,y, offset[0], offset[1], interval, offset[2],octave);
	return fp;
}


#define DXTHRESHOLD 0.03
vector<FeaturePoint> detect_local_extrema(const vector<Mat>& dogs, int octaves)
{
	vector<FeaturePoint> result;
	double  thresh = 0.5 * DXTHRESHOLD / SIFT_INTERVALS;
	for (int octave = 0; octave < octaves; octave++)
	{
		for (int layer = 1; layer < SIFT_DOG_LAYER_PER_OCT - 1; layer++)
		{
			int index = octave * SIFT_DOG_LAYER_PER_OCT + layer;

			for (int y = SIFT_IMG_BORDER; y < dogs[index].rows - SIFT_IMG_BORDER; y++)
			{
				for (int x = SIFT_IMG_BORDER; x < dogs[index].cols - SIFT_IMG_BORDER; x++)
				{
					double val = dogs[index].at<double>(y, x);
					if (std::fabs(val) > thresh)
					{
						if (is_extremum(x, y, dogs, index))
						{
							FeaturePoint* extrmum = interploation_extremum(x, y, dogs, index, octave, layer);
							if (extrmum != nullptr)
							{
								if (on_edge(*extrmum, dogs, index))
								{
									extrmum->val = dogs[index].at<double>(extrmum->y, extrmum->x);
									result.push_back(*extrmum);
								}

								delete extrmum;

							}
						}
					}
				}
			}

		}
	}
	return result;
}
void calculate_scale_half_features(vector<FeaturePoint>& features)
{
	double intvl = 0;
	for (int i = 0; i < features.size(); i++)
	{
		intvl = features[i].interval + features[i].offset_interval;
		features[i].scale = SIFT_SIGMA * pow(2.0, features[i].octave + intvl / SIFT_INTERVALS);
		features[i].octave_scale = SIFT_SIGMA * pow(2.0, intvl / SIFT_INTERVALS);
		features[i].dx /= 2;
		features[i].dy /= 2;
		features[i].scale /= 2;
	}

}


array<double, SIFT_N_BINS> get_orientation_histogram(const Mat& gauss, int x, int y, int bins, int radius, double sigma)
{
	array<double, SIFT_N_BINS> hist;
	for (int i = 0; i < bins; i++)
		hist[i] = 0.0f;

	for (int i = -radius; i <= radius; i++)
	{
		for (int j = -radius; j <= radius; j++)
		{
			int row = y + j;
			int col = x + i;
			if (row > 0 && row < gauss.rows - 1 && col > 0 && col < gauss.cols - 1)
			{
				double dx = gauss.at<double>(row, col + 1) - gauss.at<double>(row, col - 1);
				double dy = gauss.at<double>(row + 1, col) - gauss.at<double>(row - 1, col);
				double mag = sqrt(dx * dx + dy * dy);
				double ori = atan2(dy, dx);
				double weight = exp((i * i + j * j) * (-1.0 / (2.0 * sigma * sigma)));
				int bin = cvRound(bins * (CV_PI - ori) / (2.0 * CV_PI));
				bin = bin < bins ? bin : 0;
				hist[bin] += mag * weight;
			}
		}
	}

	return hist;
}

vector<FeaturePoint> get_ori_features(const FeaturePoint& fp, const array<double, SIFT_N_BINS> hist, double mag_thr){
	vector<FeaturePoint> result;
	int n = hist.size();
	for (int i = 0; i < n; i++){
		int l = (i == 0) ? n - 1 : i - 1;
		int r = (i + 1) % n;
		if (hist[i] > hist[l] && hist[i] > hist[r] && hist[i] >= mag_thr){
			double bin = i + parabola_interpolate(hist[l], hist[i], hist[r]);
			bin = (bin < 0) ? (bin + n) : (bin >= n ? (bin - n) : bin);
			FeaturePoint new_fp;
			new_fp = fp;
			new_fp.ori = ((CV_PI * 2.0f * bin) / n) - CV_PI;
			result.push_back(new_fp);
		}
	}
	return result;
}


vector<FeaturePoint> orientation_assignment(vector<FeaturePoint>& extrema, const vector<Mat>& gaussian_pyramid) {
	vector<FeaturePoint> result;
	array<double, SIFT_N_BINS> hist;
	for (int i = 0; i < extrema.size(); i++)
	{
		hist = get_orientation_histogram(gaussian_pyramid[extrema[i].octave * SIFT_LAYER_PER_OCT + extrema[i].interval],extrema[i].x, extrema[i].y, SIFT_N_BINS, cvRound(SIFT_ORI_RADIUS * extrema[i].octave_scale),SIFT_LAMBDA_ORI * extrema[i].octave_scale);

		for (int j = 0; j < SIFT_ORI_SMOOTH_TIMES; j++) {
			double prev = hist[SIFT_N_BINS - 1];
			for (int i = 0; i < SIFT_N_BINS; i++)
			{
				double temp = hist[i];
				hist[i] = 0.25 * prev + 0.5 * hist[i] + 0.25 * (i + 1 >= SIFT_N_BINS ? hist[0] : hist[i + 1]);
				prev = temp;
			}
		}
		double highest_peak = *max_element(hist.begin(), hist.end());
		vector<FeaturePoint> tmp = get_ori_features(extrema[i], hist, highest_peak * SIFT_ORI_PEAK_RATIO);
		result.insert(result.end(),tmp.begin(), tmp.end());
	}
	return result;
}

void interp_hist_entry(double*** hist, double xbin, double ybin, double obin, double mag, int bins, int d)
{
	double d_r, d_c, d_o, v_r, v_c, v_o;
	double** row, * h;
	int r0, c0, o0, rb, cb, ob, r, c, o;
	r0 = cvFloor(ybin);
	c0 = cvFloor(xbin);
	o0 = cvFloor(obin);
	d_r = ybin - r0;
	d_c = xbin - c0;
	d_o = obin - o0;

	for (r = 0; r <= 1; r++)
	{
		rb = r0 + r;
		if (rb >= 0 && rb < d)
		{
			v_r = mag * ((r == 0) ? 1.0 - d_r : d_r);
			row = hist[rb];
			for (c = 0; c <= 1; c++)
			{
				cb = c0 + c;
				if (cb >= 0 && cb < d)
				{
					v_c = v_r * ((c == 0) ? 1.0 - d_c : d_c);
					h = row[cb];
					for (o = 0; o <= 1; o++)
					{
						ob = (o0 + o) % bins;
						v_o = v_c * ((o == 0) ? 1.0 - d_o : d_o);
						h[ob] += v_o;
					}
				}
			}
		}
	}


}
double*** get_descr_hist(const Mat& gauss, int x, int y, double octave_scale, double ori, int bins, int width)
{
	double*** hist = new double** [width];

	for (int i = 0; i < width; i++)
	{
		hist[i] = new double* [width];
		for (int j = 0; j < width; j++)
		{
			hist[i][j] = new double[bins];
		}
	}

	for (int r = 0; r < width; r++)
		for (int c = 0; c < width; c++)
			for (int o = 0; o < bins; o++)
				hist[r][c][o] = 0.0;


	double cos_ori = cos(ori);
	double sin_ori = sin(ori);

	double sigma = 0.5 * width;
	double conste = -1.0 / (2 * sigma * sigma);

	double PI2 = CV_PI * 2;

	double sub_hist_width = SIFT_DESCR_SCALE_ADJUST * octave_scale;
	int    radius = (sub_hist_width * sqrt(2.0) * (width + 1)) / 2.0 + 0.5;
	double grad_ori;
	double grad_mag;

	for (int i = -radius; i <= radius; i++)
	{
		for (int j = -radius; j <= radius; j++)
		{
			double rot_x = (cos_ori * j - sin_ori * i) / sub_hist_width;
			double rot_y = (sin_ori * j + cos_ori * i) / sub_hist_width;

			double xbin = rot_x + width / 2 - 0.5;
			double ybin = rot_y + width / 2 - 0.5;

			int row = y + i;
			int col = x + j;

			if (xbin > -1.0 && xbin < width && ybin > -1.0 && ybin < width)
			{
				if (row > 0 && row < gauss.rows - 1 && col > 0 && col < gauss.cols - 1)
				{
					double dx = gauss.at<double>(row, col + 1) - gauss.at<double>(row, col - 1);
					double dy = gauss.at<double>(row + 1, col) - gauss.at<double>(row - 1, col);
					double grad_mag = sqrt(dx * dx + dy * dy);
					double grad_ori = atan2(dy, dx);

					grad_ori = (CV_PI - grad_ori) - ori;
					while (grad_ori < 0.0)
						grad_ori += PI2;
					while (grad_ori >= PI2)
						grad_ori -= PI2;

					double obin = grad_ori * (bins / PI2);

					double weight = exp(conste * (rot_x * rot_x + rot_y * rot_y));

					interp_hist_entry(hist, xbin, ybin, obin, grad_mag * weight, bins, width);

				}
			}
		}
	}

	return hist;
}

void normalize_descr(FeaturePoint& feat)
{
	double cur, len_inv, len_sq = 0.0;
	int i, d = feat.descr_length;

	for (i = 0; i < d; i++)
	{
		cur = feat.descriptor[i];
		len_sq += cur * cur;
	}
	len_inv = 1.0 / sqrt(len_sq);
	for (i = 0; i < d; i++)
		feat.descriptor[i] *= len_inv;
}

void hist_to_descriptor(double*** hist, int width, int bins, FeaturePoint& feature)
{
	int int_val, i, r, c, o, k = 0;

	for (r = 0; r < width; r++)
		for (c = 0; c < width; c++)
			for (o = 0; o < bins; o++)
			{
				feature.descriptor[k++] = hist[r][c][o];
			}

	feature.descr_length = k;
	normalize_descr(feature);

	for (i = 0; i < k; i++)
		if (feature.descriptor[i] > SIFT_DESCR_MAG_THR)
			feature.descriptor[i] = SIFT_DESCR_MAG_THR;

	normalize_descr(feature);

	for (i = 0; i < k; i++)
	{
		int_val = SIFT_INT_DESCR_FCTR * feature.descriptor[i];
		feature.descriptor[i] = min(255, int_val);
	}
}
void descriptor_representation(vector<FeaturePoint>& features, const vector<Mat>& gauss_pyr, int bins, int width)
{
	double*** hist;

	for (int i = 0; i < features.size(); i++)
	{
		hist = get_descr_hist(gauss_pyr[features[i].octave * SIFT_LAYER_PER_OCT + features[i].interval],
			features[i].x, features[i].y, features[i].octave_scale, features[i].ori, bins, width);

		hist_to_descriptor(hist, width, bins, features[i]);

		for (int j = 0; j < width; j++)
		{
			for (int k = 0; k < width; k++)
			{
				delete[] hist[j][k];
			}
			delete[] hist[j];
		}
		delete[] hist;
	}
}



Mat draw_keypoints(const Mat& target, vector<FeaturePoint>& fps, int size) {
	Mat result = target;
	for (const FeaturePoint& fp : fps) {
		int x = fp.dx;
		int y = fp.dy;
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

Mat draw_matches(const Mat& a, const Mat& b, std::vector<FeaturePoint>& kps_a, std::vector<FeaturePoint>& kps_b, std::vector<std::pair<int, int>> matches)
{
	Mat res = Mat::zeros(std::max(a.rows, b.rows), a.cols + b.cols, CV_8UC3);

	for (int i = 0; i < a.cols; i++) {
		for (int j = 0; j < a.rows; j++) {
			res.at<Vec3b>(j, i)[0] = a.at<Vec3b>(j, i)[0];
			res.at<Vec3b>(j, i)[1] = a.at<Vec3b>(j, i)[1];
			res.at<Vec3b>(j, i)[2] = a.at<Vec3b>(j, i)[2];
		}
	}
	for (int i = 0; i < b.cols; i++) {
		for (int j = 0; j < b.rows; j++) {
			res.at<Vec3b>(j, a.cols + i)[0] = b.at<Vec3b>(j, i)[0];
			res.at<Vec3b>(j, a.cols + i)[1] = b.at<Vec3b>(j, i)[1];
			res.at<Vec3b>(j, a.cols + i)[2] = b.at<Vec3b>(j, i)[2];
		}
	}

	for (auto& m : matches) {
		FeaturePoint& kp_a = kps_a[m.first];
		FeaturePoint& kp_b = kps_b[m.second];
		line(res, Point(kp_a.dx, kp_a.dy), Point(a.cols + kp_b.dx, kp_b.dy), CV_RGB(0, 255, 0));
	}
	return res;
}


vector<FeaturePoint> SIFT(Mat img) {
	vector<FeaturePoint> result;
	int octaves = log((double)min(img.rows, img.cols)) / log(2.0) - 2;
	vector<Mat> gaussian_pyramid = get_gaussian_pyramid(img, octaves);
	vector<Mat> dogs = get_dog_pyramid(gaussian_pyramid, octaves);
	vector<FeaturePoint> extrema = detect_local_extrema(dogs, octaves);
	calculate_scale_half_features(extrema);
	result = orientation_assignment(extrema, gaussian_pyramid);
	descriptor_representation(result, gaussian_pyramid, 8, 4);
	sort(result.begin(), result.end(), [&](const FeaturePoint& f1,const FeaturePoint& f2)->bool	{
			return f1.scale < f2.scale;
	});
	cout << "feature point count:  " << result.size() << endl;
	return result;
}




void match_feature_points(vector<vector<FeaturePoint> >& img_fps_list) {
	int img_count = img_fps_list.size();
	vector<ANNkd_tree*> kd_trees;
	for (int i = 0; i < img_count; i++) {
		int feature_count = img_fps_list[i].size();
		ANNpointArray ann_descriptors = annAllocPts(feature_count, FEATURE_ELEMENT_LENGTH);
		for (int feature_index = 0; feature_index < feature_count; feature_index++) {
			for (int y = 0; y < FEATURE_ELEMENT_LENGTH; y++) {
				ann_descriptors[feature_index][y] = img_fps_list[i][feature_index].descriptor[y];
			}
		}
		kd_trees.push_back(new ANNkd_tree(ann_descriptors, feature_count, FEATURE_ELEMENT_LENGTH));
	}
	ANNidx nn_idx[2];
	ANNdist dists[2];
	for (int i = 0; i < img_count; i++) {
		int feature_num = img_fps_list[i].size();
		for (int feature_index = 0; feature_index < feature_num; feature_index++) {
			img_fps_list[i][feature_index].best_match.assign(img_count, -1);
			for (int j = 0; j < img_count; j++) {
				if (i == j) continue;
				kd_trees[j]->annkSearch(kd_trees[i]->thePoints()[feature_index], 2, nn_idx, dists, 0);
				if (dists[0] < 0.5f * dists[1]) {
					img_fps_list[i][feature_index].best_match[j] = nn_idx[0];
				}
			}
		}
	}
	for (int i = 0; i < img_count; i++) {
		int feature_count = img_fps_list[i].size();
		for (int feature_index = 0; feature_index < feature_count; feature_index++) {
			for (int j = 0; j < img_count; j++) {
				if (i == j) continue;
				if (img_fps_list[i][feature_index].best_match[j] == -1) continue;
				if (img_fps_list[j][img_fps_list[i][feature_index].best_match[j]].best_match[i] != feature_index) {
					img_fps_list[i][feature_index].best_match[j] = -1;
				}
			}
		}
	}
	for (int i = 0; i < img_count; i++) {
		ANNpointArray ann_descriptors = kd_trees[i]->thePoints();
		annDeallocPts(ann_descriptors);
		delete kd_trees[i];
	}
	annClose();
}
