#define _USE_MATH_DEFINES
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseQR>

#include "HDR.h"


#define Zfunc(img,COLOR,row,col) ((img).at<Vec3b>((row),(col))[(COLOR)])
#define Wfunc(val) (1.0f - ((float)abs(val-127)/127.0f))


//#define opencv_SVD
//#define eigen_jacobi
#define eigen_SparseQR

#define fixed_sample

Mat HDR_recover(vector<Mat> images,vector<float> exposure_times) {
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    int image_count = images.size();
    vector<Mat> images_sample(images.size());
    for (int i = 0; i < image_count; i++) {
#ifdef fixed_sample
        resize(images[i], images_sample[i], Size(12, 16));
#else
        int scale_down = 100;
        resize(images[i], images_sample[i], Size(images[i].cols / scale_down, images[i].rows / scale_down));
#endif // fixed_sample
    }
    int origin_pixel_count = images[0].rows * images[0].cols;
    int pixel_count = images_sample[0].rows * images_sample[0].cols;// N
    int row_count = images_sample[0].rows;
    int col_count = images_sample[0].cols;

    int n = 256;
    float l = 30;

#ifdef opencv_SVD
    Mat x[3];
#elif defined(eigen_jacobi)
    Eigen::MatrixXf A(pixel_count * image_count + n + 1, n + pixel_count);
    Eigen::VectorXf b(A.rows());
    Eigen::VectorXf x[3];
#elif defined(eigen_SparseQR)
    Eigen::SparseMatrix<float> A(pixel_count * image_count + n + 1, n + pixel_count);
    Eigen::VectorXf b(A.rows());
    Eigen::VectorXf x[3];
#endif // !opencv_SVD

    float Gfunc[3][256];
    float logG[3][256];
    Mat HDR = Mat::zeros(images[0].size(), CV_32FC3);
    Mat HDR_radiance = Mat::zeros(images[0].size(), CV_32FC3);
    auto axes = CvPlot::makePlotAxes();

    bool show_log = true;
    for (int color = 0; color < 3; color++) {
#ifdef opencv_SVD
        Mat A = Mat::zeros(pixel_count * image_count + n + 1, n + pixel_count, CV_32F);
        Mat b = Mat::zeros(A.rows, 1, CV_32F);
#else
        A.setZero();
        b.setZero();
#endif

        int k = 0;
        int i = 0;

        for (int row = 0; row < row_count; row++) {
            for (int col = 0; col < col_count; col++) {
                for (int img_index = 0; img_index < image_count; img_index++) {
                    int z_val = Zfunc(images_sample[img_index], color, row, col);
                    float wij = Wfunc(z_val);

#ifdef opencv_SVD
                    A.at<float>(k, z_val) = wij;
                    A.at<float>(k, n + i) = -wij;
                    b.at<float>(k, 0) = wij * log(exposure_times[img_index]) / log(2.718281828);
#else
                    A.coeffRef(k, z_val) = wij;
                    A.coeffRef(k, n + i) = -wij;
                    b.coeffRef(k) = wij * log(exposure_times[img_index]);
#endif // !opencv_SVD

                    k = k + 1;
                }
                i++;
            }
        }

#ifdef opencv_SVD
        A.at<float>(k, 128) = 1.0f;
#else
        A.coeffRef(k, 128) = 1.0f;
#endif // !opencv_SVD

        k = k + 1;
        for (int i = 0; i < n - 2; i++) {
            float lW = l * Wfunc(i + 1);

#ifdef opencv_SVD
            A.at<float>(k, i) = lW;
            A.at<float>(k, i + 1) = -2 * lW;
            A.at<float>(k, i + 2) = lW;
#else
            A.coeffRef(k, i) = lW;
            A.coeffRef(k, i + 1) = -2 * lW;
            A.coeffRef(k, i + 2) = lW;
#endif // !opencv_SVD

            k = k + 1;
        }



        std::cout << "Ax=b solving..." << std::endl;

#ifdef opencv_SVD
        solve(A, b, x[color], DECOMP_SVD); // Pseudo Inverse
#elif defined(eigen_jacobi)
        x[color] = A.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b);
#elif defined(eigen_SparseQR)
        Eigen::LeastSquaresConjugateGradient<Eigen::SparseMatrix<float>> lscg;
        lscg.compute(A);
        x[color] = lscg.solve(b);
#endif // !opencv_SVD


        std::vector<float> x_index;
        for (int i = 0; i < 256; i++) {
#ifdef opencv_SVD
            logG[color][i] = (x[color].at<float>(i));
            Gfunc[color][i] = exp(x[color].at<float>(i));
#else
            logG[color][i] = (x[color].coeff(i));
            Gfunc[color][i] = exp(x[color].coeff(i));
#endif // !opencv_SVD

            x_index.push_back(i);
        }


        if (!show_log) {
            switch (color)
            {
            case 0:axes.create<CvPlot::Series>(x_index, std::vector<float>(std::begin(Gfunc[color]), std::end(Gfunc[color])), "-b"); break;
            case 1:axes.create<CvPlot::Series>(x_index, std::vector<float>(std::begin(Gfunc[color]), std::end(Gfunc[color])), "-g"); break;
            case 2:axes.create<CvPlot::Series>(x_index, std::vector<float>(std::begin(Gfunc[color]), std::end(Gfunc[color])), "-r"); break;
            default:
                break;
            }
        }
        else {
            switch (color)
            {
            case 0:axes.create<CvPlot::Series>(x_index, std::vector<float>(std::begin(logG[color]), std::end(logG[color])), "-b"); break;
            case 1:axes.create<CvPlot::Series>(x_index, std::vector<float>(std::begin(logG[color]), std::end(logG[color])), "-g"); break;
            case 2:axes.create<CvPlot::Series>(x_index, std::vector<float>(std::begin(logG[color]), std::end(logG[color])), "-r"); break;
            default:
                break;
            }
        }


    }
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << std::endl;

    imshow("response curve", axes.render());
    for (int color = 0; color < 3; color++)
    {
        for (int row = 0; row < images[0].rows; row++)
        {
            for (int col = 0; col < images[0].cols; col++)
            {
                float log_sum = 0, weight_sum = 0;

                for (int img_index = 0; img_index < image_count; img_index++)
                {
                    int pixel_val = images[img_index].at<Vec3b>(row, col)(color);

                    log_sum += Wfunc(pixel_val) * log(Gfunc[color][pixel_val] / exposure_times[img_index]);
                    weight_sum += Wfunc(pixel_val);
                }
                float result = exp(log_sum / weight_sum);
                if (isinf(result) || isnan(result)) {
                    HDR.at<Vec3f>(row, col)(color) = 0.0f;
                }
                else {
                    HDR.at<Vec3f>(row, col)(color) = result;
                }
                //HDR_radiance.at<Vec3f>(row, col)(color) = log_sum / weight_sum;
                //std::cout << HDR.at<Vec3f>(row, col)(color) << "   " << HDR_radiance.at<Vec3f>(row, col)(color) <<'\n';
                //std::cout << HDR.at<Vec3f>(row, col)(color) << std::endl;
            }
        }
    }
    return HDR;
}