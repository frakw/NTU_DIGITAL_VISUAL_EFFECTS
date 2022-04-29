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
bool isExtremum(int x, int y, const vector<Mat>& dogs, int index)
{
	double val = dogs[index].at<double>(y, x);

	if (val > 0)
	{
		for (int i = -1; i <= 1; i++) //prev current next layer
		{
			//3*3�d��
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
			//3*3�d��
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
/*************************************************************************************************************************
*�Ҳջ����G
*       �����t���D�ɡH
**************************************************************************************************************************/
#define Hat(i, j) (*(H+(i)*3 + (j)))

double PyrAt(const vector<Mat>& pyr, int index, int x, int y)
{
	pixel_t* data = (pixel_t*)pyr[index].data;
	int      step = pyr[index].step / sizeof(data[0]);
	pixel_t   val = *(data + y * step + x);

	return val;
}
/*************************************************************************************************************************
*�Ҳջ����G
*       �����t���D�ɡH
**************************************************************************************************************************/
#define At(index, x, y) (PyrAt(dog_pyr, (index), (x), (y)))

//3��D(x)�@������,dx�C�V�q
void DerivativeOf3D(int x, int y, const vector<Mat>& dog_pyr, int index, double* dx)
{
	double Dx = (At(index, x + 1, y) - At(index, x - 1, y)) / 2.0;
	double Dy = (At(index, x, y + 1) - At(index, x, y - 1)) / 2.0;
	double Ds = (At(index + 1, x, y) - At(index - 1, x, y)) / 2.0;

	dx[0] = Dx;
	dx[1] = Dy;
	dx[2] = Ds;
}

//3��D(x)�G�����ɡA�YHessian�x�}
void Hessian3D(int x, int y, const vector<Mat>& dog_pyr, int index, double* H)
{
	double val, Dxx, Dyy, Dss, Dxy, Dxs, Dys;

	val = At(index, x, y);

	Dxx = At(index, x + 1, y) + At(index, x - 1, y) - 2 * val;
	Dyy = At(index, x, y + 1) + At(index, x, y - 1) - 2 * val;
	Dss = At(index + 1, x, y) + At(index - 1, x, y) - 2 * val;

	Dxy = (At(index, x + 1, y + 1) + At(index, x - 1, y - 1)
		- At(index, x + 1, y - 1) - At(index, x - 1, y + 1)) / 4.0;

	Dxs = (At(index + 1, x + 1, y) + At(index - 1, x - 1, y)
		- At(index - 1, x + 1, y) - At(index + 1, x - 1, y)) / 4.0;

	Dys = (At(index + 1, x, y + 1) + At(index - 1, x, y - 1)
		- At(index + 1, x, y - 1) - At(index - 1, x, y + 1)) / 4.0;

	Hat(0, 0) = Dxx;
	Hat(1, 1) = Dyy;
	Hat(2, 2) = Dss;

	Hat(1, 0) = Hat(0, 1) = Dxy;
	Hat(2, 0) = Hat(0, 2) = Dxs;
	Hat(2, 1) = Hat(1, 2) = Dys;
}
/*************************************************************************************************************************
*�Ҳջ����G
*       4.4 �T���x�}�D�f
**************************************************************************************************************************/
#define HIat(i, j) (*(H_inve+(i)*3 + (j)))
//3*3���x�}�D�f
bool Inverse3D(const double* H, double* H_inve)
{

	double A = Hat(0, 0) * Hat(1, 1) * Hat(2, 2)
		+ Hat(0, 1) * Hat(1, 2) * Hat(2, 0)
		+ Hat(0, 2) * Hat(1, 0) * Hat(2, 1)
		- Hat(0, 0) * Hat(1, 2) * Hat(2, 1)
		- Hat(0, 1) * Hat(1, 0) * Hat(2, 2)
		- Hat(0, 2) * Hat(1, 1) * Hat(2, 0);

	if (fabs(A) < 1e-10) return false;

	HIat(0, 0) = Hat(1, 1) * Hat(2, 2) - Hat(2, 1) * Hat(1, 2);
	HIat(0, 1) = -(Hat(0, 1) * Hat(2, 2) - Hat(2, 1) * Hat(0, 2));
	HIat(0, 2) = Hat(0, 1) * Hat(1, 2) - Hat(0, 2) * Hat(1, 1);

	HIat(1, 0) = Hat(1, 2) * Hat(2, 0) - Hat(2, 2) * Hat(1, 0);
	HIat(1, 1) = -(Hat(0, 2) * Hat(2, 0) - Hat(0, 0) * Hat(2, 2));
	HIat(1, 2) = Hat(0, 2) * Hat(1, 0) - Hat(0, 0) * Hat(1, 2);

	HIat(2, 0) = Hat(1, 0) * Hat(2, 1) - Hat(1, 1) * Hat(2, 0);
	HIat(2, 1) = -(Hat(0, 0) * Hat(2, 1) - Hat(0, 1) * Hat(2, 0));
	HIat(2, 2) = Hat(0, 0) * Hat(1, 1) - Hat(0, 1) * Hat(1, 0);

	for (int i = 0; i < 9; i++)
	{
		*(H_inve + i) /= A;
	}
	return true;
}
/*************************************************************************************************************************
*�Ҳջ����G
*
**************************************************************************************************************************/
//�p��x^
void GetOffsetX(int x, int y, const vector<Mat>& dog_pyr, int index, double* offset_x)
{
	//x^ = -H^(-1) * dx; dx = (Dx, Dy, Ds)^T
	double H[9], H_inve[9] = { 0 };
	Hessian3D(x, y, dog_pyr, index, H);
	Inverse3D(H, H_inve);
	double dx[3];
	DerivativeOf3D(x, y, dog_pyr, index, dx);

	for (int i = 0; i < 3; i++)
	{
		offset_x[i] = 0.0;
		for (int j = 0; j < 3; j++)
		{
			offset_x[i] += H_inve[i * 3 + j] * dx[j];
		}
		offset_x[i] = -offset_x[i];
	}
}

//�p��|D(x^)|
double GetFabsDx(int x, int y, const vector<Mat>& dog_pyr, int index, const double* offset_x)
{
	//|D(x^)|=D + 0.5 * dx * offset_x; dx=(Dx, Dy, Ds)^T
	double dx[3];
	DerivativeOf3D(x, y, dog_pyr, index, dx);

	double term = 0.0;
	for (int i = 0; i < 3; i++)
		term += dx[i] * offset_x[i];

	pixel_t* data = (pixel_t*)dog_pyr[index].data;
	int step = dog_pyr[index].step / sizeof(data[0]);
	pixel_t val = *(data + y * step + x);

	return fabs(val + 0.5 * term);
}
/*************************************************************************************************************************
*�Ҳջ����G
*       �Ҳե|���ĤG�B:�ץ������I�A�R����í�w���I
*�\�໡��:
*       1--�ھڰ����t���禡���ͪ������I�ä������Oí�w���S�x�I�A�]���Y�Ƿ����I���T�����z�A�ӥBDOG�B�⤸�|���͸��j����t�T��
*       2--�H�W��k�˴��쪺�����I�O�����Ŷ��������I�A�U���q�L���X�T���G���禡�Ӻ�T�w�������I����m�M�ثסA�P�ɥh������
*          �C�M��í�w����t�T���I(�]��DOG�B�⤸�|���͸��j����t�T��)�A�H�W�j�ǰt��í�w�ʡB�����ܾ��n����O�C
*       3--�ץ������I�A�R����í�w�I�A|D(x)| < 0.03 Lowe 2004
**************************************************************************************************************************/
FeaturePoint* InterploationExtremum(int x, int y, const vector<Mat>& dog_pyr, int index, int octave, int interval, double dxthreshold = 0.03)
{
	//�p��x=(x,y,sigma)^T
	//x^ = -H^(-1) * dx; dx = (Dx, Dy, Ds)^T
	double offset_x[3] = { 0 };

	const Mat& mat = dog_pyr[index];

	int idx = index;
	int intvl = interval;
	int i = 0;

	while (i < 5)
	{
		GetOffsetX(x, y, dog_pyr, idx, offset_x);
		//4. Accurate keypoint localization.  Lowe
		//�p�Goffset_x �����@���פj��0.5�Ait means that the extremum lies closer to a different sample point.
		if (fabs(offset_x[0]) < 0.5 && fabs(offset_x[1]) < 0.5 && fabs(offset_x[2]) < 0.5)
			break;

		//�ΩP���I�N��
		x += cvRound(offset_x[0]);
		y += cvRound(offset_x[1]);
		interval += cvRound(offset_x[2]);

		idx = index - intvl + interval;
		//���B�O���˴���� x+1,y+1�Mx-1, y-1����
		if (interval < 1 || interval > 3 || x >= mat.cols - 1 || x < 2 || y >= mat.rows - 1 || y < 2)
		{
			return nullptr;
		}

		i++;
	}

	//«�異��
	if (i >= 5)
		return nullptr;

	//rejecting unstable extrema
	//|D(x^)| < 0.03���g���
	if (GetFabsDx(x, y, dog_pyr, idx, offset_x) < dxthreshold / 3)
	{
		return nullptr;
	}

	FeaturePoint* keypoint = new FeaturePoint;


	keypoint->x = x;
	keypoint->y = y;

	keypoint->offset_x = offset_x[0];
	keypoint->offset_y = offset_x[1];

	keypoint->interval = interval;
	keypoint->offset_interval = offset_x[2];

	keypoint->octave = octave;

	keypoint->dx = (x + offset_x[0]) * pow(2.0, octave);
	keypoint->dy = (y + offset_x[1]) * pow(2.0, octave);

	return keypoint;
}









/*************************************************************************************************************************
*�Ҳջ����G
*       �Ҳե|�G3.5 �Ŷ������I���˴�(�����I����B���d)
*�\�໡���G
*       1--�����I�O��DOG�Ŷ����ϰ�ʷ����I�զ����A�����I����B���d�O�q�L�P�@�դ��UDoG�۾F��h�v�H����������������C���F�M��DoG
*          �禡�������I�A�C�@�ӵe���I���n�M���Ҧ��۾F���I����A�ݨ�O�_�񥦪��v�H��M�ثװ�۾F���I�j�٬O�p�C
*       2--��M�o�˲��ͪ������I�ä������Oí�w���S�x�I�A�]���Y�Ƿ����I�������z�A�ӥBDOG�B�⤸�|���͸��j����t�T���C
**************************************************************************************************************************/

#define IMG_BORDER 5
#define DXTHRESHOLD 0.03
vector<FeaturePoint> detect_local_extrema(const vector<Mat>& dogs, int octaves)
{
	vector<FeaturePoint> result;
	double  thresh = 0.5 * DXTHRESHOLD / SIFT_INTERVALS;
	int count = 0;
	for (int octave = 0; octave < octaves; octave++)
	{
		//�Ĥ@�h�M�̫�@�h���ȩ���
		for (int layer = 1; layer < SIFT_DOG_LAYER_PER_OCT - 1; layer++)
		{
			int index = octave * SIFT_DOG_LAYER_PER_OCT + layer;                              //[1]�Ϥ����ު��w��

			for (int y = IMG_BORDER; y < dogs[index].rows - IMG_BORDER; y++)
			{
				for (int x = IMG_BORDER; x < dogs[index].cols - IMG_BORDER; x++)
				{
					pixel_t val = dogs[index].at<double>(y, x);
					if (std::fabs(val) > thresh)                           //[4]�ư��H�ȹL�p���I
					{
						if (isExtremum(x, y, dogs, index))                //[5]�P�_�O�_�O����
						{
							FeaturePoint* extrmum = InterploationExtremum(x, y, dogs, index, octave, layer);
							if (extrmum != nullptr)
							{
								//if (passEdgeResponse(extrmum->x, extrmum->y, dog_pyr, index))
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
			}//for y

		}
	}
	cout << "count: " << count << endl;
	return result;
}

/*************************************************************************************************************************
*�Ҳջ����G
*       �Ҳդ��G
*�\�໡���G
*
**************************************************************************************************************************/
void CalculateScale(vector<FeaturePoint>& features, double sigma = SIFT_SIGMA, int intervals = SIFT_N_SPO)
{
	double intvl = 0;
	for (int i = 0; i < features.size(); i++)
	{
		intvl = features[i].interval + features[i].offset_interval;
		features[i].scale = sigma * pow(2.0, features[i].octave + intvl / intervals);
		features[i].octave_scale = sigma * pow(2.0, intvl / intervals);
	}

}

//���X�j���v�H�S�x�Y��
void HalfFeatures(vector<FeaturePoint>& features)
{
	for (int i = 0; i < features.size(); i++)
	{
		features[i].dx /= 2;
		features[i].dy /= 2;

		features[i].scale /= 2;
	}
}
/********************************************************************************************************************************
*�Ҳջ���:
*        �Ҳդ�---�B�J2�G�p�������I����שM��פ�V
*�\�໡���G
*        1�^�p�������I(x,y)�B����״T�ȩM��פ�V
*        2�^�N�ҭp��X�Ӫ���״T�ȩM��פ�V�x�s�b�ܼ�mag�Mori��
*********************************************************************************************************************************/
bool CalcGradMagOri(const Mat& gauss, int x, int y, double& mag, double& ori)
{
	if (x > 0 && x < gauss.cols - 1 && y > 0 && y < gauss.rows - 1)
	{
		pixel_t* data = (pixel_t*)gauss.data;
		int step = gauss.step / sizeof(*data);

		double dx = *(data + step * y + (x + 1)) - (*(data + step * y + (x - 1)));           //[1]�Q��X��V�W���t���N���L��dx
		double dy = *(data + step * (y + 1) + x) - (*(data + step * (y - 1) + x));           //[2]�Q��Y��V�W���t���N���L��dy

		mag = sqrt(dx * dx + dy * dy);                                          //[3]�p��������I����״T��
		ori = atan2(dy, dx);                                                //[4]�p��������I����פ�V
		return true;
	}
	else
		return false;
}
/********************************************************************************************************************************
*�Ҳջ���:
*        �Ҳդ�---�B�J1�G�p���ת���V�����
*�\�໡���G
*        1�^����ϥH�C10�׬��@�ӬW�A�@36�ӬW�A�W�N����V�����e���I����פ�V�A�W�����u�N��F��״T�ȡC
*        2�^�ھ�Lowe����ĳ�A����ϲέp�ĥ�3*1.5*sigma
*        3�^�b����ϲέp�ɡA�C�۾F�T�ӵe���I�ĥΰ����[�v�A�ھ�Lowe����ĳ�A�ҪO�ĥ�[0.25,0.5,0.25],�åB�s��[�v�⦸
*��    �סG
*        �v�H�������I�˴�������A�C�������I�N�֦��T�Ӹ�T�G��m�B�ثסB��V�F�P�ɤ]�N�������I��ƥ����B�Y��M���ण�ܩ�
*********************************************************************************************************************************/
double* CalculateOrientationHistogram(const Mat& gauss, int x, int y, int bins, int radius, double sigma)
{
	double* hist = new double[bins];                           //[1]�ʺA���t�@��double���O���}�C
	for (int i = 0; i < bins; i++)                               //[2]���o�Ӱ}�C��l��
		*(hist + i) = 0.0;

	double  mag;                                                //[3]�����I����״T��                                          
	double  ori;                                                //[4]�����I����פ�V
	double  weight;

	int           bin;
	const double PI2 = 2.0 * CV_PI;
	double        econs = -1.0 / (2.0 * sigma * sigma);

	for (int i = -radius; i <= radius; i++)
	{
		for (int j = -radius; j <= radius; j++)
		{
			if (CalcGradMagOri(gauss, x + i, y + j, mag, ori))       //[5]�p��������I����״T�ȩM��V
			{
				weight = exp((i * i + j * j) * econs);
				bin = cvRound(bins * (CV_PI - ori) / PI2);     //[6]��@��double�檺�ƶi��|�ˤ��J�A��^�@�Ӿ�Ϊ���
				bin = bin < bins ? bin : 0;

				hist[bin] += mag * weight;                      //[7]�έp��ת���V�����
			}
		}
	}

	return hist;
}
/********************************************************************************************************************************
*�Ҳջ���:
*        �Ҳդ�---�B�J3�G���פ�V����϶i��s��⦸����������
*�\�໡���G
*        1�^�b����ϲέp�ɡA�C�۾F�T�ӵe���I�ĥΰ����[�v�A�ھ�Lowe����ĳ�A�ҪO�ĥ�[0.25,0.5,0.25],�åB�s��[�v�⦸
*        2�^�諾��϶i��⦸����
*********************************************************************************************************************************/
void GaussSmoothOriHist(double* hist, int n)
{
	double prev = hist[n - 1];
	double temp;
	double h0 = hist[0];

	for (int i = 0; i < n; i++)
	{
		temp = hist[i];
		hist[i] = 0.25 * prev + 0.5 * hist[i] + 0.25 * (i + 1 >= n ? h0 : hist[i + 1]);//���V����϶i�氪������
		prev = temp;
	}
}
/********************************************************************************************************************************
*�Ҳջ���:
*        �Ҳդ�---�B�J4�G�p���V����Ϥ����D��V
*********************************************************************************************************************************/
double DominantDirection(double* hist, int n)
{
	double maxd = hist[0];
	for (int i = 1; i < n; i++)
	{
		if (hist[i] > maxd)                            //�D��36�ӬW�����̤j�p��
			maxd = hist[i];
	}
	return maxd;
}
void CopyKeypoint(const FeaturePoint& src, FeaturePoint& dst)
{
	dst.dx = src.dx;
	dst.dy = src.dy;

	dst.interval = src.interval;
	dst.octave = src.octave;
	dst.octave_scale = src.octave_scale;
	dst.offset_interval = src.offset_interval;

	dst.offset_x = src.offset_x;
	dst.offset_y = src.offset_y;

	dst.ori = src.ori;
	dst.scale = src.scale;
	dst.val = src.val;
	dst.x = src.x;
	dst.y = src.y;
}
/********************************************************************************************************************************
*�Ҳջ���:
*        �Ҳդ�---�B�J5�G�p���[��T�������I�D��V----�ߪ�����
*�\�໡���G
*        1�^��V����Ϫ��p�ȫh�N��F�ӯS�x�I����V�A�H����Ϥ����̤j�ȧ@���������I���D��V�C���F�W�j�ǰt���|�ΩʡA�u�O�d�p�Ȥj��D
*           ��V�p��80%����V�@���������I������V�C�]���A���P�@��׭ȱo�h�Ӯp�Ȫ������I��m�A�b�ۦP��m�M�ثױN�|���h�������I�Q
*           �إߦ���V���P�C�Ȧ�15%�������I�Q�ᤩ�h�Ӥ�V�A���O�i�H���㪺���������I��í�w�ʡC
*        2�^�b��ڵ{���]�p���A�N�O��������I�ƻs���h�������I�A�ñN��V�Ȥ��O�ᵹ�o�ǽƻs�᪺�����I
*        3�^�åB�A��������ת���ϭn�i��i�������X�B�z�j�A�ӨD�o��[��T����V���׭�
*********************************************************************************************************************************/
#define Parabola_Interpolate(l, c, r) (0.5*((l)-(r))/((l)-2.0*(c)+(r))) 
void CalcOriFeatures(const FeaturePoint& keypoint, vector<FeaturePoint>& features, const double* hist, int n, double mag_thr)
{
	double  bin;
	double  PI2 = CV_PI * 2.0;
	int l;
	int r;

	for (int i = 0; i < n; i++)
	{
		l = (i == 0) ? n - 1 : i - 1;
		r = (i + 1) % n;

		//hist[i]�O����
		if (hist[i] > hist[l] && hist[i] > hist[r] && hist[i] >= mag_thr)
		{
			bin = i + Parabola_Interpolate(hist[l], hist[i], hist[r]);
			bin = (bin < 0) ? (bin + n) : (bin >= n ? (bin - n) : bin);

			FeaturePoint new_key;

			CopyKeypoint(keypoint, new_key);

			new_key.ori = ((PI2 * bin) / n) - CV_PI;
			features.push_back(new_key);
		}
	}
}
/********************************************************************************************************************************
*�Ҳջ���:
*        �Ҳդ��G5 �����I��V���t
*�\�໡���G
*        1�^���F�ϴy�z�Ũ㦳���ण�ܩʡA�ݭn�Q�μv�H���ϰ�ʯS�x���C�@�������I���t�@�Ӱ�Ǥ�V�C�ϥμv�H��ת���k�D���ϰ�ʵ��c��í�w
*           ��V�C
*        2�^���bDOG���r���˴��X�Ӫ������I�A�Ķ���Ҧb�������r��v�H3sigma�F��������e������שM��V��שM��V�S�x�C
*        3�^��ת��ҩM��V�p�U�ҥ�:
*        4) �b���������I����׭p���A�ϥΪ���ϲέp�F�줺�e������שM��V�C��ת���ϱN0~360�ת���V�d�����36�ӬW�A�䤤�C�W10�סA
*           �p��5.1�ҥܡA����Ϫ��p�Ȥ�V�N��F�����I���D��V
*********************************************************************************************************************************/

#define ORI_SMOOTH_TIMES 2
void OrientationAssignment(vector<FeaturePoint>& extrema, vector<FeaturePoint>& features, const vector<Mat>& gauss_pyr)
{
	int n = extrema.size();
	double* hist;

	for (int i = 0; i < n; i++)
	{

		hist = CalculateOrientationHistogram(gauss_pyr[extrema[i].octave * (SIFT_N_SPO + 3) + extrema[i].interval],
			extrema[i].x, extrema[i].y, SIFT_N_BINS, cvRound(SIFT_ORI_RADIUS * extrema[i].octave_scale),
			SIFT_LAMBDA_ORI * extrema[i].octave_scale);                             //[1]�p���ת���V�����

		for (int j = 0; j < ORI_SMOOTH_TIMES; j++)
			GaussSmoothOriHist(hist, SIFT_N_BINS);                              //[2]���V����϶i�氪������
		double highest_peak = DominantDirection(hist, SIFT_N_BINS);            //[3]�D����V����Ϥ����p��
																				  //[4]�p���[��T�������I�D��V
		CalcOriFeatures(extrema[i], features, hist, SIFT_N_BINS, highest_peak * SIFT_ORI_PEAK_RATIO);

		delete[] hist;

	}
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
		�����ȡG
		xbin,ybin,obin:�ؤl�I�Ҧb�l��������m�M��V
		�Ҧ��ؤl�I���N���b4*4��������
		r0,c0�����j��xbin�Aybin�������
		r0,c0�u�����0,1,2
		xbin,ybin�b(-1, 2)

		r0�����j��xbin������ƮɡC
		r0+0 <= xbin <= r0+1
		mag�b�϶�[r0,r1]�W������

		obin�P�z
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
*�Ҳջ���:
*        �ҲդC--�B�J1:�p��y�z�l�������
*�\�໡���G
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

	//6.1�����v�ȡAsigma����y�z�r�����e�ת��@�b
	double sigma = 0.5 * width;
	double conste = -1.0 / (2 * sigma * sigma);

	double PI2 = CV_PI * 2;

	double sub_hist_width = DESCR_SCALE_ADJUST * octave_scale;

	//�i1�j�p��y�z�l�һݪ��v�H���ϰ쪺�b�|
	int    radius = (sub_hist_width * sqrt(2.0) * (width + 1)) / 2.0 + 0.5;    //[1]0.5���|�ˤ��J
	double grad_ori;
	double grad_mag;

	for (int i = -radius; i <= radius; i++)
	{
		for (int j = -radius; j <= radius; j++)
		{
			double rot_x = (cos_ori * j - sin_ori * i) / sub_hist_width;
			double rot_y = (sin_ori * j + cos_ori * i) / sub_hist_width;

			double xbin = rot_x + width / 2 - 0.5;                         //[2]xbin,ybin�����b4*4���������U�Э�
			double ybin = rot_y + width / 2 - 0.5;

			if (xbin > -1.0 && xbin < width && ybin > -1.0 && ybin < width)
			{
				if (CalcGradMagOri(gauss, x + j, y + i, grad_mag, grad_ori)) //[3]�p�������I����פ�V
				{
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
*�Ҳջ���:
*        �ҲդC--�B�J2:����Ϩ�y�z�l���ഫ
*�\�໡���G
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
	NormalizeDescr(feature);                           //[1]�y�z�l�S�x�V�q�k�@��

	for (i = 0; i < k; i++)                           //[2]�y�z�l�V�q����
		if (feature.descriptor[i] > DESCR_MAG_THR)
			feature.descriptor[i] = DESCR_MAG_THR;

	NormalizeDescr(feature);                           //[3]�y�z�l�i��̫�@�����k�@�ƾާ@

	for (i = 0; i < k; i++)                           //[4]�N���ׯB�I�����y�z�l�ഫ����Ϊ��y�z�l
	{
		int_val = SIFT_INT_DESCR_FCTR * feature.descriptor[i];
		feature.descriptor[i] = min(255, int_val);
	}
}
/********************************************************************************************************************************
*�Ҳջ���:
*        �ҲդC:6 �����I�y�z
*�\�໡���G
*        1�^�q�L�H�W�B�J�A���@�������I�A�֦��T�Ӹ�T�G��m�B�ثסB��V
*        2�^���U�ӴN�O���C�������I�إߤ@�Ӵy�z�šA�Τ@�զV�q�ӱN�o�������I�y�z�X�ӡA�Ϩ䤣�H�U���ܤƦ��ܤơA��p���ӡB�����ܤƵ���
*        3�^�o�Ӵy�z�l�����]�A�����I�A�]�]�t�����I�P����^�m���e���I�A�åB�y�z�����Ӧ��������W�S�ʡA�H�K��S�x�I���T���ǰt���v
*        1�^SIFT�y�z�l----�O�����I�F�찪���v�H��ײέp���G���@�ت�ܡC
*        2�^�q�L�������I�P��v�H�ϰ�����A�p�������ת���ϡA�ͦ��㦳�W�S�ʪ��V�q
*        3�^�o�ӦV�q�O�Ӱϰ�v�H��T���@�ت�z�M��H�A�㦳�ߤ@�ʡC
*Lowe�פ�G
*    Lowe��ĳ�y�z�l�ϥΦb�����I�ثתŶ���4*4���������p�⪺8�Ӥ�V����׸�T�A�@4*4*8=128���V�q�Ӫ�x�C���骺�B�J�p�U�ҥ�:
*        1)�T�w�p��y�z�l�һݪ��v�H�ϰ�
*        2�^�N�y�жb���ର�����I����V�A�H�T�O���ण�ܩʡA�pCSDN�դ夤����6.2�ҥܡF�����F������I���s�y�Хi�H�q�L����(6-2)�p��
*        3�^�N�F�줺�������I���t��������l�ϰ�A�N�l�ϰ줺����׭Ȥ��t��8�Ӥ�V�W�A�p����v��
*        4�^���ȭp��C�Ӻؤl�I�K�Ӥ�V�����
*        5�^�p�W�έp��4*4*8=128�ӱ�׸�T�Y���������I���S�x�V�q�C�S�x�V�q�Φ���A���F�h�������ܤƪ��v�T�A�ݭn�復�̶i���k�@�ƳB�z�A
*           ���v�H�ǫ׭Ⱦ���}���A�v�H�U�I����׬O�F��e���۴�o�쪺�A�ҥH�]��h���C�o�쪺�y�z�l�V�q��H�A�k�@�Ƥ��᪺�V�q��L
*        6�^�y�z�l�V�q�����C�D�u�ʥ��ӡA�۾����M���ܤƹ�y���Y�Ǥ�V����׭ȹL�j�A�ӹ��V���v�T�L�z�C�]���A�]�w�����ȡ]�V�q�k�@��
*           ��A�@���0.2�^�I�_���j����׭ȡC�M��A�b�i��@���k�@�ƳB�z�A�����S�x��Ų�O�ʡC
*        7�^���S�x�I���ث׹�S�x�y�z�V�q�i��Ƨ�
*        8�^�ܦ��ASIFT�S�x�y�z�V�q�ͦ��C
*********************************************************************************************************************************/
void DescriptorRepresentation(vector<FeaturePoint>& features, const vector<Mat>& gauss_pyr, int bins, int width)
{
	double*** hist;

	for (int i = 0; i < features.size(); i++)
	{                                                                       //[1]�p��y�z�l�������
		hist = CalculateDescrHist(gauss_pyr[features[i].octave * (SIFT_N_SPO + 3) + features[i].interval],
			features[i].x, features[i].y, features[i].octave_scale, features[i].ori, bins, width);

		HistToDescriptor(hist, width, bins, features[i]);                   //[2]����Ϩ�y�z�l���ഫ

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
*�禡����:
*        �e�XSIFT�S�x�I������禡
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
*�禡����:
*         �̤j���Ҳ�3�G�e�XSIFT�S�x�I
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

	CalculateScale(extrema, SIFT_SIGMA, SIFT_N_SPO);
	HalfFeatures(extrema);

	OrientationAssignment(extrema, result, gaussian_pyramid);

	DescriptorRepresentation(result, gaussian_pyramid, 8, 4);
	sort(result.begin(), result.end(), FeatureCmp);
	cout << "result.size: " << result.size() << endl;
	imshow("result", draw_keypoints(img, result, 3));
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


	//vector<Mat> result(SIFT_N_OCTAVE * SIFT_INTVLS); //�p�Ⱚ�����r���`layer��(�X��octave * �C��octave�X��layer)
	//vector<double> sigmas(SIFT_INTVLS);
	//int result_index = 0;
	//double k = pow(2.0f, 1.0f / (double)(SIFT_INTVLS - 3));//-3�����A���G�O�t�X�U����sigma�p��
	//int img_row = img.rows * 2;
	//int img_col = img.cols * 2;
	//
	//for (int i = 0; i < SIFT_N_OCTAVE;i++) {
	//	double total_sigma = SIFT_SIGMA_MIN;
	//	double pre_total_sigma = total_sigma;
	//	Mat octave_base;
	//	if (i != 0) {//�C�hoctave���Ĥ@�i�Ϥ���GaussianBlur�A�]���Y�p�N�㦳�ҽk���ĪG
	//		img_row /= 2.0f;
	//		img_col /= 2.0f;
	//		resize(img, octave_base, Size(img_col, img_row), 0.0f, 0.0f, INTER_LINEAR);
	//		result[result_index] = octave_base;
	//		result_index++;
	//	}
	//	else { //first image �Ĥ@�hoctave���Ĥ@�i��
	//		resize(img, octave_base, Size(img_col, img_row), 0.0f, 0.0f, INTER_LINEAR);
	//		double sigma_diff = sqrt((total_sigma * total_sigma) / (0.25f) - 1.0f);//����
	//		GaussianBlur(octave_base, octave_base, Size(3, 3), sigma_diff, sigma_diff);
	//		result[result_index] = octave_base;
	//		result_index++;
	//	}

	//	for (int j = 1; j < SIFT_INTVLS; j++) { //�C�hoctave�Ĥ@�i�᪺��
	//		total_sigma *= k;
	//		double sigma_diff = sqrt(total_sigma * total_sigma - pre_total_sigma * pre_total_sigma);//����
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
	vector<Mat> result(SIFT_N_OCTAVE * (SIFT_INTVLS - 1));//�p��DOG���r���`layer��(�X��octave * (��Ӱ����Ϭ۴�A�]���|�֤@��))	
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
		for (int j = 1; j < (SIFT_INTVLS - 1) - 1; j++) {//�|�ݭn�e����layer����T�A�]���Ĥ@�Ӹ�̫�@��layer���]
			Mat& prev = dogs[i * (SIFT_INTVLS - 1) + j - 1];
			Mat& current = dogs[i * (SIFT_INTVLS - 1) + j];
			Mat& next = dogs[i * (SIFT_INTVLS - 1) + j + 1];
			int img_row = current.rows;
			int img_col = current.cols;
			//�]���n3*3�ϰ�A�]���Ĥ@�ӻP�̫�@�ӹ�������
			for (int row = 1; row < img_row - 1; row++) {
				//�]���n3*3�ϰ�A�]���Ĥ@�ӻP�̫�@�ӹ�������
				for (int col = 1; col < img_col - 1; col++) {
					//cout << current.at<float>(row, col) << endl;
					if (fabs(current.at<float>(row, col)) < 0.8 * SIFT_C_DOG) {
					//if (fabs(current.at<float>(row, col)) < threshold) {
						count++;
						continue;
					}
					//if (is_extremum(prev, current, next, row, col)) {
					if (isExtremum(col,row, dogs, i * (SIFT_INTVLS - 1) + j)) {
						
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
	cout << "count: " << count << endl;
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
					//�ˬd�̤p����
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




/// �ݭק�//////////////////////////////////////////////
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


/// �ݭק�//////////////////////////////////////////////
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