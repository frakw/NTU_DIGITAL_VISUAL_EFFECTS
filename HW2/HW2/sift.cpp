#include "sift.h"
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
vector<Mat> get_dog_pyramid(const vector<Mat>& gauss_pyr, int octaves, int intervals = SIFT_N_SPO)
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
#define Hat(i, j) (*(H+(i)*3 + (j)))
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

#define IMG_BORDER 5
#define DXTHRESHOLD 0.03
vector<FeaturePoint> detect_local_extrema(const vector<Mat>& dogs, int octaves)
{
	vector<FeaturePoint> result;
	double  thresh = 0.5 * DXTHRESHOLD / SIFT_INTERVALS;
	int count = 0;
	for (int octave = 0; octave < octaves; octave++)
	{
		for (int layer = 1; layer < SIFT_DOG_LAYER_PER_OCT - 1; layer++)
		{
			int index = octave * SIFT_DOG_LAYER_PER_OCT + layer;

			for (int y = IMG_BORDER; y < dogs[index].rows - IMG_BORDER; y++)
			{
				for (int x = IMG_BORDER; x < dogs[index].cols - IMG_BORDER; x++)
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
					else {
						count++;
					}
				}
			}

		}
	}
	cout << "count: " << count << endl;
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
				int bin = cvRound(bins * (CV_PI - ori) / (2.0 * CV_PI));     //[6]對一個double行的數進行四捨五入，返回一個整形的數
				bin = bin < bins ? bin : 0;
				hist[bin] += mag * weight;                      //[7]統計梯度的方向直方圖
			}
		}
	}

	return hist;
}
/********************************************************************************************************************************
*模組說明:
*        模組六---步驟5：計算更加精確的關鍵點主方向----拋物插值
*功能說明：
*        1）方向直方圖的峰值則代表了該特徵點的方向，以直方圖中的最大值作為該關鍵點的主方向。為了增強匹配的魯棒性，只保留峰值大於主
*           方向峰值80%的方向作為改關鍵點的輔方向。因此，對於同一梯度值得多個峰值的關鍵點位置，在相同位置和尺度將會有多個關鍵點被
*           建立但方向不同。僅有15%的關鍵點被賦予多個方向，但是可以明顯的提高關鍵點的穩定性。
*        2）在實際程式設計中，就是把該關鍵點複製成多份關鍵點，並將方向值分別賦給這些複製後的關鍵點
*        3）並且，離散的梯度直方圖要進行【插值擬合處理】，來求得更加精確的方向角度值
*********************************************************************************************************************************/
#define Parabola_Interpolate(l, c, r) (0.5*((l)-(r))/((l)-2.0*(c)+(r))) 
vector<FeaturePoint> CalcOriFeatures(const FeaturePoint& fp, const array<double, SIFT_N_BINS> hist, double mag_thr){
	vector<FeaturePoint> result;
	int n = hist.size();
	for (int i = 0; i < n; i++){
		int l = (i == 0) ? n - 1 : i - 1;
		int r = (i + 1) % n;
		if (hist[i] > hist[l] && hist[i] > hist[r] && hist[i] >= mag_thr){
			double bin = i + Parabola_Interpolate(hist[l], hist[i], hist[r]);
			bin = (bin < 0) ? (bin + n) : (bin >= n ? (bin - n) : bin);
			FeaturePoint new_fp;
			new_fp = fp;
			new_fp.ori = ((CV_PI * 2.0f * bin) / n) - CV_PI;
			result.push_back(new_fp);
		}
	}
	return result;
}
/********************************************************************************************************************************
*模組說明:
*        模組六：5 關鍵點方向分配
*功能說明：
*        1）為了使描述符具有旋轉不變性，需要利用影象的區域性特徵為每一個關鍵點分配一個基準方向。使用影象梯度的方法求取區域性結構的穩定
*           方向。
*        2）對於在DOG金字塔中檢測出來的關鍵點，採集其所在高斯金字塔影象3sigma鄰域視窗內畫素的梯度和方向梯度和方向特徵。
*        3）梯度的模和方向如下所示:
*        4) 在完成關鍵點的梯度計算後，使用直方圖統計鄰域內畫素的梯度和方向。梯度直方圖將0~360度的方向範圍分為36個柱，其中每柱10度，
*           如圖5.1所示，直方圖的峰值方向代表了關鍵點的主方向
*********************************************************************************************************************************/

#define ORI_SMOOTH_TIMES 2
vector<FeaturePoint> orientation_assignment(vector<FeaturePoint>& extrema, const vector<Mat>& gaussian_pyramid) {
	vector<FeaturePoint> result;
	array<double, SIFT_N_BINS> hist;
	for (int i = 0; i < extrema.size(); i++)
	{
		hist = get_orientation_histogram(gaussian_pyramid[extrema[i].octave * SIFT_LAYER_PER_OCT + extrema[i].interval],extrema[i].x, extrema[i].y, SIFT_N_BINS, cvRound(SIFT_ORI_RADIUS * extrema[i].octave_scale),SIFT_LAMBDA_ORI * extrema[i].octave_scale);

		for (int j = 0; j < ORI_SMOOTH_TIMES; j++) {
			double prev = hist[SIFT_N_BINS - 1];
			for (int i = 0; i < SIFT_N_BINS; i++)
			{
				double temp = hist[i];
				hist[i] = 0.25 * prev + 0.5 * hist[i] + 0.25 * (i + 1 >= SIFT_N_BINS ? hist[0] : hist[i + 1]);
				prev = temp;
			}
		}
		double highest_peak = *max_element(hist.begin(), hist.end());
		vector<FeaturePoint> tmp = CalcOriFeatures(extrema[i], hist, highest_peak * SIFT_ORI_PEAK_RATIO);
		result.insert(result.end(),tmp.begin(), tmp.end());

	}
	return result;
}

void InterpHistEntry(double*** hist, double xbin, double ybin, double obin, double mag, int bins, int d)
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

	/*
		做插值：
		xbin,ybin,obin:種子點所在子視窗的位置和方向
		所有種子點都將落在4*4的視窗中
		r0,c0取不大於xbin，ybin的正整數
		r0,c0只能取到0,1,2
		xbin,ybin在(-1, 2)

		r0取不大於xbin的正整數時。
		r0+0 <= xbin <= r0+1
		mag在區間[r0,r1]上做插值

		obin同理
	*/

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
#define DESCR_SCALE_ADJUST 3
/********************************************************************************************************************************
*模組說明:
*        模組七--步驟1:計算描述子的直方圖
*功能說明：
*
*********************************************************************************************************************************/
double*** CalculateDescrHist(const Mat& gauss, int x, int y, double octave_scale, double ori, int bins, int width)
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

	//6.1高斯權值，sigma等於描述字視窗寬度的一半
	double sigma = 0.5 * width;
	double conste = -1.0 / (2 * sigma * sigma);

	double PI2 = CV_PI * 2;

	double sub_hist_width = DESCR_SCALE_ADJUST * octave_scale;

	//【1】計算描述子所需的影象領域區域的半徑
	int    radius = (sub_hist_width * sqrt(2.0) * (width + 1)) / 2.0 + 0.5;    //[1]0.5取四捨五入
	double grad_ori;
	double grad_mag;

	for (int i = -radius; i <= radius; i++)
	{
		for (int j = -radius; j <= radius; j++)
		{
			double rot_x = (cos_ori * j - sin_ori * i) / sub_hist_width;
			double rot_y = (sin_ori * j + cos_ori * i) / sub_hist_width;

			double xbin = rot_x + width / 2 - 0.5;                         //[2]xbin,ybin為落在4*4視窗中的下標值
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

					InterpHistEntry(hist, xbin, ybin, obin, grad_mag * weight, bins, width);

				}
			}
		}
	}

	return hist;
}

void NormalizeDescr(FeaturePoint& feat)
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
/********************************************************************************************************************************
*模組說明:
*        模組七--步驟2:直方圖到描述子的轉換
*功能說明：
*
*********************************************************************************************************************************/
#define DESCR_MAG_THR 0.2f
void HistToDescriptor(double*** hist, int width, int bins, FeaturePoint& feature)
{
	int int_val, i, r, c, o, k = 0;

	for (r = 0; r < width; r++)
		for (c = 0; c < width; c++)
			for (o = 0; o < bins; o++)
			{
				feature.descriptor[k++] = hist[r][c][o];
			}

	feature.descr_length = k;
	NormalizeDescr(feature);                           //[1]描述子特徵向量歸一化

	for (i = 0; i < k; i++)                           //[2]描述子向量門限
		if (feature.descriptor[i] > DESCR_MAG_THR)
			feature.descriptor[i] = DESCR_MAG_THR;

	NormalizeDescr(feature);                           //[3]描述子進行最後一次的歸一化操作

	for (i = 0; i < k; i++)                           //[4]將單精度浮點型的描述子轉換為整形的描述子
	{
		int_val = SIFT_INT_DESCR_FCTR * feature.descriptor[i];
		feature.descriptor[i] = min(255, int_val);
	}
}
/********************************************************************************************************************************
*模組說明:
*        模組七:6 關鍵點描述
*功能說明：
*        1）通過以上步驟，對於一個關鍵點，擁有三個資訊：位置、尺度、方向
*        2）接下來就是為每個關鍵點建立一個描述符，用一組向量來將這個關鍵點描述出來，使其不隨各種變化而變化，比如光照、視角變化等等
*        3）這個描述子不但包括關鍵點，也包含關鍵點周圍對其貢獻的畫素點，並且描述符應該有較高的獨特性，以便於特徵點正確的匹配概率
*        1）SIFT描述子----是關鍵點鄰域高斯影象梯度統計結果的一種表示。
*        2）通過對關鍵點周圍影象區域分塊，計算塊內梯度直方圖，生成具有獨特性的向量
*        3）這個向量是該區域影象資訊的一種表述和抽象，具有唯一性。
*Lowe論文：
*    Lowe建議描述子使用在關鍵點尺度空間內4*4的視窗中計算的8個方向的梯度資訊，共4*4*8=128維向量來表徵。具體的步驟如下所示:
*        1)確定計算描述子所需的影象區域
*        2）將座標軸旋轉為關鍵點的方向，以確保旋轉不變性，如CSDN博文中的圖6.2所示；旋轉後鄰域取樣點的新座標可以通過公式(6-2)計算
*        3）將鄰域內的取樣點分配到對應的子區域，將子區域內的梯度值分配到8個方向上，計算其權值
*        4）插值計算每個種子點八個方向的梯度
*        5）如上統計的4*4*8=128個梯度資訊即為該關鍵點的特徵向量。特徵向量形成後，為了去除光照變化的影響，需要對它們進行歸一化處理，
*           對於影象灰度值整體漂移，影象各點的梯度是鄰域畫素相減得到的，所以也能去除。得到的描述子向量為H，歸一化之後的向量為L
*        6）描述子向量門限。非線性光照，相機飽和度變化對造成某些方向的梯度值過大，而對方向的影響微弱。因此，設定門限值（向量歸一化
*           後，一般取0.2）截斷較大的梯度值。然後，在進行一次歸一化處理，提高特徵的鑑別性。
*        7）按特徵點的尺度對特徵描述向量進行排序
*        8）至此，SIFT特徵描述向量生成。
*********************************************************************************************************************************/
void DescriptorRepresentation(vector<FeaturePoint>& features, const vector<Mat>& gauss_pyr, int bins, int width)
{
	double*** hist;

	for (int i = 0; i < features.size(); i++)
	{                                                                       //[1]計算描述子的直方圖
		hist = CalculateDescrHist(gauss_pyr[features[i].octave * (SIFT_N_SPO + 3) + features[i].interval],
			features[i].x, features[i].y, features[i].octave_scale, features[i].ori, bins, width);

		HistToDescriptor(hist, width, bins, features[i]);                   //[2]直方圖到描述子的轉換

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

bool FeatureCmp(FeaturePoint& f1, FeaturePoint& f2)
{
	return f1.scale < f2.scale;
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

/*******************************************************************************************************************
*函式說明:
*        畫出SIFT特徵點的具體函式
********************************************************************************************************************/
void DrawSiftFeature(Mat& src, FeaturePoint& feat, cv::Scalar color)
{
	int len, hlen, blen, start_x, start_y, end_x, end_y, h1_x, h1_y, h2_x, h2_y;
	double scl, ori;
	double scale = 5.0;
	double hscale = 0.75;
	cv::Point start, end, h1, h2;

	/* compute points for an arrow scaled and rotated by feat's scl and ori */
	start_x = cvRound(feat.dx);
	start_y = cvRound(feat.dy);
	scl = feat.scale;
	ori = feat.ori;
	len = cvRound(scl * scale);
	hlen = cvRound(scl * hscale);
	blen = len - hlen;
	end_x = cvRound(len * cos(ori)) + start_x;
	end_y = cvRound(len * -sin(ori)) + start_y;
	h1_x = cvRound(blen * cos(ori + CV_PI / 18.0)) + start_x;
	h1_y = cvRound(blen * -sin(ori + CV_PI / 18.0)) + start_y;
	h2_x = cvRound(blen * cos(ori - CV_PI / 18.0)) + start_x;
	h2_y = cvRound(blen * -sin(ori - CV_PI / 18.0)) + start_y;
	start = cv::Point(start_x, start_y);
	end = cv::Point(end_x, end_y);
	h1 = cv::Point(h1_x, h1_y);
	h2 = cv::Point(h2_x, h2_y);

	line(src, start, end, color, 1, 8, 0);
	line(src, end, h1, color, 1, 8, 0);
	line(src, end, h2, color, 1, 8, 0);
}
/*******************************************************************************************************************
*函式說明:
*         最大的模組3：畫出SIFT特徵點
********************************************************************************************************************/
void DrawSiftFeatures(Mat& src, vector<FeaturePoint>& features)
{
	cv::Scalar color = CV_RGB(0, 255, 0);
	for (int i = 0; i < features.size(); i++)
	{
		DrawSiftFeature(src, features[i], color);
	}
}

vector<FeaturePoint> SIFT(Mat img) {
	vector<FeaturePoint> result;
	int octaves = log((double)min(img.rows, img.cols)) / log(2.0) - 2;
	cout << "octaves: " << octaves << endl;
	vector<Mat> gaussian_pyramid = get_gaussian_pyramid(img, octaves);
	cout << "gaussian_pyramid.octave.size: " << gaussian_pyramid.size() << endl;
	vector<Mat> dogs = get_dog_pyramid(gaussian_pyramid, octaves);

	vector<FeaturePoint> extrema = detect_local_extrema(dogs, octaves);

	cout << "local extrema.size: " << extrema.size() << endl;

	calculate_scale_half_features(extrema);
	//half_features(extrema);

	result = orientation_assignment(extrema, gaussian_pyramid);

	DescriptorRepresentation(result, gaussian_pyramid, 8, 4);
	sort(result.begin(), result.end(), FeatureCmp);
	cout << "result.size: " << result.size() << endl;
	//imshow("result", draw_keypoints(img, result, 3));
	//DrawSiftFeatures(img, result);
	//imshow("result", img);
	return result;

	//vector<Mat> gaussian_pyramid = get_gaussian_pyramid(img);
	//cout << "gaussian_pyramid.octave.size: " << gaussian_pyramid.size() << endl;

	//vector<Mat> dogs = difference_of_gaussian_pyramid(gaussian_pyramid);

	//cout << "dogs.octave.size: " << dogs.size() << endl;



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

	//vector<FeaturePoint> feature_points = find_feature_points(dog_pyr);

	//cout << "feature_points.size: " << feature_points.size() << endl;


	//vector<Mat> gradient_pyramid = generate_gradient_pyramid(gauss_pyr);

	//for (int i = 0; i < feature_points.size(); i++) {
	//	vector<float> orientations = get_orientations(feature_points[i], gradient_pyramid);
	//	for (int j = 0; j < orientations.size(); j++) {
	//		
	//		result.push_back(compute_keypoint_descriptor(feature_points[i], orientations[j], gradient_pyramid));
	//	}
	//}
	//imshow("result",draw_keypoints(img,result,3));
	//return result;
}




void featureMatch(vector< vector<FeaturePoint> >& img_fps_list) {
	int image_num = img_fps_list.size();
	ANNpointArray* descriptors = new ANNpointArray[image_num];
	ANNkd_tree** kdTree = new ANNkd_tree * [image_num];
	for (int i = 0; i < image_num; i++) {
		int feature_num = img_fps_list[i].size();
		//cout << feature_num << endl;
		descriptors[i] = annAllocPts(feature_num, 128);
		for (int x = 0; x < feature_num; x++) {
			for (int y = 0; y < 128; y++) {
				//cout << "x = " << x << " , y = " << y;
				//cout << " , " << (*feature_list)[i][x].x << endl;
				//cout << " , " << (*feature_list)[i][x].descriptorList[y]<< endl;
				descriptors[i][x][y] = img_fps_list[i][x].descriptor[y];
			}
		}
		kdTree[i] = new ANNkd_tree(descriptors[i], feature_num, 128);
	}
	ANNpoint fd = annAllocPt(128);
	ANNidxArray nnIdx = new ANNidx[2];
	ANNdistArray dists = new ANNdist[2];
	for (int i = 0; i < image_num; i++) {
		int feature_num = img_fps_list[i].size();
		for (int x = 0; x < feature_num; x++) {
			img_fps_list[i][x].best_match.assign(image_num, -1);
			for (int y = 0; y < 128; y++)
				fd[y] = img_fps_list[i][x].descriptor[y];

			for (int j = 0; j < image_num; j++) {
				if (i == j) //same image
					continue;
				kdTree[j]->annkSearch(fd, 2, nnIdx, dists, 0);
				if (dists[0] < 0.5 * dists[1])
					img_fps_list[i][x].best_match[j] = nnIdx[0];
			}
		}
	}
	//double check
	for (int i = 0; i < image_num; i++) {
		int feature_num = img_fps_list[i].size();
		for (int x = 0; x < feature_num; x++) {
			for (int j = 0; j < image_num; j++) {
				if (i == j)
					continue;
				if (img_fps_list[i][x].best_match[j] == -1)
					continue;
				if (img_fps_list[j][img_fps_list[i][x].best_match[j]].best_match[i] != x)
					img_fps_list[i][x].best_match[j] = -1;
			}
		}
	}
	delete[] descriptors;
	delete[] kdTree;
	delete[] dists;
	delete[] nnIdx;
	annDeallocPt(fd);
	annClose();
}


//void detectOutliers(const int offset, const int width, const Feat& feat1, const Feat& feat2, const vector<array<int, 2>>& matches, vector<array<int, 2>>& puredMatches)
//{
//	vector<tuple<int, int>> score;
//	vector<tuple<int, int>> moveVector;
//	for (int i = 0; i < matches.size(); i++)
//	{
//		int x1 = get<1>(feat1.keypoints[matches[i][0]]) + offset;
//		int y1 = get<0>(feat1.keypoints[matches[i][0]]);
//		int x2 = get<1>(feat2.keypoints[matches[i][1]]);
//		int y2 = get<0>(feat2.keypoints[matches[i][1]]);
//		moveVector.push_back(make_tuple(x1 - x2, y1 - y2));
//	}
//
//
//	for (int i = 0; i < matches.size(); i++)
//	{
//		int tmp = 0;
//		for (int j = 0; j < matches.size(); j++)
//		{
//			int tmp_a = 0;
//			int tmp_b = 0;
//			tmp_a = abs(get<0>(moveVector[i]) - get<0>(moveVector[j])) * abs(get<0>(moveVector[i]) - get<0>(moveVector[j]));
//			tmp_b = abs(get<1>(moveVector[i]) - get<1>(moveVector[j])) * abs(get<1>(moveVector[i]) - get<1>(moveVector[j]));
//			tmp = (int)sqrt(tmp_a + tmp_b);
//		}
//		//cout << tmp << endl;
//		score.push_back(make_tuple(i, tmp));
//	}
//
//	sort(begin(score), end(score), [](tuple<int, int> const& t1, tuple<int, int> const& t2) {
//		return get<1>(t1) < get<1>(t2);
//		});
//
//	// 0.05 parrington best
//	for (int i = 0; i < matches.size() * PURE_THRESH; i++)
//	{
//		//cout << get<1>(score[i]) << endl;	
//		puredMatches.push_back(matches[get<0>(score[i])]);
//	}
//	cout << "pured matching: " << puredMatches.size() << endl;
//}
//
//


double euclidean_dist(double a[FEATURE_ELEMENT_LENGTH], double b[FEATURE_ELEMENT_LENGTH])
{
	double dist = 0;
	for (int i = 0; i < 128; i++) {
		double di = a[i] - b[i];
		dist += di * di;
	}
	return std::sqrt(dist);
}

std::vector<std::pair<int, int>> find_keypoint_matches(std::vector<FeaturePoint>& a,
	std::vector<FeaturePoint>& b,
	float thresh_relative,
	float thresh_absolute)
{
	assert(a.size() >= 2 && b.size() >= 2);

	std::vector<std::pair<int, int>> matches;

	for (int i = 0; i < a.size(); i++) {
		// find two nearest neighbours in b for current keypoint from a
		int nn1_idx = -1;
		float nn1_dist = 100000000, nn2_dist = 100000000;
		for (int j = 0; j < b.size(); j++) {
			float dist = euclidean_dist(a[i].descriptor, b[j].descriptor);
			if (dist < nn1_dist) {
				nn2_dist = nn1_dist;
				nn1_dist = dist;
				nn1_idx = j;
			}
			else if (nn1_dist <= dist && dist < nn2_dist) {
				nn2_dist = dist;
			}
		}
		if (nn1_dist < thresh_relative * nn2_dist && nn1_dist < thresh_absolute) {
			matches.push_back({ i, nn1_idx });
		}
	}
	return matches;
}


Mat draw_matches(const Mat& a, const Mat& b, std::vector<FeaturePoint>& kps_a,
	std::vector<FeaturePoint>& kps_b, std::vector<std::pair<int, int>> matches)
{
	Mat res = Mat::zeros(std::max(a.rows, b.rows),a.cols + b.cols,CV_8UC3);

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
		//draw_line(res, kp_a.x, kp_a.y, a.width + kp_b.x, kp_b.y);
	}
	return res;
}

Mat draw_matches2(const Mat& a, const Mat& b, std::vector<FeaturePoint>& kps_a,
	std::vector<FeaturePoint>& kps_b)
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

	int count = 0;
	for (auto& fp : kps_a) {
		for (auto& index : fp.best_match) {
			if (index == -1) continue;
			FeaturePoint& kp_b = kps_b[index];
			line(res, Point(fp.dx, fp.dy), Point(a.cols + kp_b.dx, kp_b.dy), CV_RGB(0, 255, 0));
			//draw_line(res, kp_a.x, kp_a.y, a.width + kp_b.x, kp_b.y);

			count++;
		}
	}
	cout << "match count: " << count << endl;
	//for (auto& fp : kps_b) {
	//	for (auto& index : fp.best_match) {
	//		if (index == -1) continue;
	//		FeaturePoint& kp_b = kps_a[index];
	//		line(res, Point(fp.dx, fp.dy), Point(a.cols + kp_b.dx, kp_b.dy), CV_RGB(0, 255, 0));
	//		//draw_line(res, kp_a.x, kp_a.y, a.width + kp_b.x, kp_b.y);

	//		count++;
	//	}
	//}

	cout << "match count: " << count << endl;

	return res;
}


/*

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
	//int threshold = cvFloor(0.5f * SIFT_CONTR_THR / 3.0f * 255);
	float threshold = 0.5f * SIFT_CONTR_THR / 3.0f;
	cout << "threshold " << threshold << endl;
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
					if (fabs(current.at<float>(row, col)) < 0.8 * SIFT_C_DOG) {
					//if (fabs(current.at<float>(row, col)) < threshold) {
						count++;
						continue;
					}
					//if (is_extremum(prev, current, next, row, col)) {
					if (isExtremum(col,row, dogs, i * (SIFT_INTVLS - 1) + j)) {
						
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

bool isExtremum(int x, int y, const vector<Mat>& dog_pyr, int index)
{
	float* data = (float*)dog_pyr[index].data;
	int      step = dog_pyr[index].step / sizeof(data[0]);
	float   val = *(data + y * step + x);

	if (val > 0)
	{
		for (int i = -1; i <= 1; i++)
		{
			int stp = dog_pyr[index + i].step / sizeof(data[0]);
			for (int j = -1; j <= 1; j++)
			{
				for (int k = -1; k <= 1; k++)
				{
					if (val < *((float*)dog_pyr[index + i].data + stp * (y + j) + (x + k)))
					{
						return false;
					}
				}
			}
		}
	}
	else
	{
		for (int i = -1; i <= 1; i++)
		{
			int stp = dog_pyr[index + i].step / sizeof(float);
			for (int j = -1; j <= 1; j++)
			{
				for (int k = -1; k <= 1; k++)
				{
					//檢查最小極值
					if (val > *((float*)dog_pyr[index + i].data + stp * (y + j) + (x + k)))
					{
						return false;
					}
				}
			}
		}
	}
	return true;
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

float euclidean_dist(std::vector<uint8_t>& a, vector<uint8_t>& b)
{
	float dist = 0;
	for (int i = 0; i < 128; i++) {
		int di = (int)a[i] - b[i];
		dist += di * di;
	}
	return std::sqrt(dist);
}


std::vector<std::pair<int, int>> find_keypoint_matches(std::vector<FeaturePoint>& a,std::vector<FeaturePoint>& b,float thresh_relative,float thresh_absolute)
{
	assert(a.size() >= 2 && b.size() >= 2);

	std::vector<std::pair<int, int>> matches;

	for (int i = 0; i < a.size(); i++) {
		// find two nearest neighbours in b for current keypoint from a
		int nn1_idx = -1;
		float nn1_dist = 100000000, nn2_dist = 100000000;
		for (int j = 0; j < b.size(); j++) {
			float dist = euclidean_dist(a[i].descriptor, b[j].descriptor);
			if (dist < nn1_dist) {
				nn2_dist = nn1_dist;
				nn1_dist = dist;
				nn1_idx = j;
			}
			else if (nn1_dist <= dist && dist < nn2_dist) {
				nn2_dist = dist;
			}
		}
		if (nn1_dist < thresh_relative * nn2_dist && nn1_dist < thresh_absolute) {
			matches.push_back({ i, nn1_idx });
		}
	}
	return matches;
}


*/