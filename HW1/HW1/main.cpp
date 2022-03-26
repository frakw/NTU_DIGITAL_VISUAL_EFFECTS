#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cmath>
#include <chrono>
#include <limits>
#include <algorithm>
#include <iterator>

#include <opencv2/opencv.hpp>

#define CVUI_IMPLEMENTATION
#include "cvui.h"
#include "tinyexpr.h"
#include "alignment.h"
#include "MTB.h"
#include "HDR.h"
#include "tonemapping.h"

using namespace std;
using namespace cv;


int main() {
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    
    std::string folder = "./exposures/";
    int image_count = 13;
    std::string* image_paths = new std::string[image_count];
    float* exposure_times = new float[image_count];
    std::fstream time_file(folder + "time.data");
    for (int i = 1; i <= image_count; i++) {
        std::string path = folder;
        std::string time;
        if (i < 10) path += "img0";
        else path += "img";

        path += std::to_string(i) + ".jpg";
        image_paths[i - 1] = path;
        time_file >> time;
        //std::cout << time << '\n';
        exposure_times[i - 1] = te_interp(time.c_str(),0);
        std::cout << exposure_times[i - 1] << '\n';

    }
    std::vector<Mat> images;
    //std::vector<tuple<Mat,double>> images_t;
   
    for (int i = 0; i < image_count; i++) {
        images.push_back(imread(image_paths[i], 1));
        //images_t.push_back(make_tuple(imread(image_paths[i], IMREAD_COLOR), exposure_times[i]));
    }
    bool MTB_open = false;
    if (MTB_open) {
        //MTBA(images, images);
        images = MTB(images);
        //images_t = alignment(images_t,false);
        //for (int i = 0; i < images_t.size(); i++) {
        //    //cout << get<0>(images_t[i]).type() << endl;
        //    cvtColor(get<0>(images_t[i]),images[i],COLOR_BGRA2BGR);
        //    //images[i] = get<0>(images_t[i]);
        //}

    }
    //Mat HDR = Debevec_HDR_recover(images, vector<float>(exposure_times, exposure_times + image_count));
    Mat HDR = Robertson_HDR_recover(images, vector<float>(exposure_times, exposure_times + image_count),30);
    Mat HDR_radiance;
    imshow("HDR", HDR);
   
    cvtColor(HDR, HDR_radiance, COLOR_BGR2GRAY);
    HDR_radiance.convertTo(HDR_radiance,CV_8U,255.0f);
    applyColorMap(HDR_radiance, HDR_radiance, COLORMAP_JET);
    imshow("HDR radiance", HDR_radiance);
    
    imwrite("recovered_HDR.exr", HDR);

    //tonemapping/////////
    
    auto AfterTonemapping = global_operator(HDR);
    //auto AfterTonemapping = local_operator(HDR);
    //auto AfterTonemapping = logarithmic_operator(HDR);
    //auto AfterTonemapping = bilateral_operator(HDR,true);

    //Mat ldr;
    //Ptr<Tonemap> tonemap = createTonemap(2.2f);
    //tonemap->process(HDR, ldr);
    //imshow("opencv tonemapping", ldr);


    cout << "final img rows cols\n";

    cout << AfterTonemapping.first.rows << ' ' << AfterTonemapping.first.cols << endl;
    imshow("After tonemapping", AfterTonemapping.first);

    imwrite("tonemapping.png", AfterTonemapping.second);
    std::cout << "finish";
    cv::waitKey();
    return 0;
}