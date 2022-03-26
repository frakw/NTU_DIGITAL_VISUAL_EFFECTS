#include <iostream>
#include <fstream>
#include <string>
#include <cstring>
#include <vector>
#include <cmath>
#include <chrono>
#include <limits>
#include <algorithm>
#include <iterator>
#include <set>

#include <opencv2/opencv.hpp>

#include "tinyexpr.h"
#include "MTB.h"
#include "HDR.h"
#include "tonemapping.h"

using namespace std;
using namespace cv;


int main(int argc, char* argv[]) {
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    if (argc < 4) {
        cout << "command error\n";
        return 0;
    }
    int image_count = atoi(argv[1]);
    string folder = argv[2];
    string file_extension = argv[3];
    if (folder.back() != '/') folder.push_back('/');
    vector<string> image_paths(image_count);
    vector<float> exposure_times(image_count);
    fstream time_file(folder + "time.data");
    for (int i = 1; i <= image_count; i++) {
        string path = folder;
        string time;
        if (i < 10) path += "img0";
        else path += "img";
        path += std::to_string(i) + '.' + file_extension;
        image_paths[i - 1] = path;
        time_file >> time;
        exposure_times[i - 1] = te_interp(time.c_str(), 0);
        std::cout << path << "\t" << exposure_times[i - 1] << '\n';
    }
    set<string> parameter;
    for (int i = 4; i < argc; i++) {
        parameter.insert(argv[i]);
    }
    std::vector<Mat> images;
    for (int i = 0; i < image_count; i++) {
        images.push_back(imread(image_paths[i], 1));
    }
    if (parameter.find("-MTB") != parameter.end()) {
        cout << "running MTB\n";
        images = MTB(images);
    }
    Mat HDR;
    if (parameter.find("-Debevec") != parameter.end()) {
        cout << "running Debevec HDR recover\n";
        HDR = Debevec_HDR_recover(images, exposure_times);
    }
    else if (parameter.find("-Robertson") != parameter.end()) {
        cout << "running Robertson HDR recover\n";
        HDR = Robertson_HDR_recover(images, exposure_times, 30);
    }
    else { //default
        cout << "running Debevec HDR recover\n";
        HDR = Debevec_HDR_recover(images, exposure_times);
    }

    if (parameter.find("-show-hdr") != parameter.end()) {
        imshow("HDR", HDR);
    }

    if (parameter.find("-show-hdr-radiance") != parameter.end()) {
        Mat HDR_radiance;
        cvtColor(HDR, HDR_radiance, COLOR_BGR2GRAY);
        HDR_radiance.convertTo(HDR_radiance, CV_8U, 255.0f);
        applyColorMap(HDR_radiance, HDR_radiance, COLORMAP_JET);
        imshow("HDR radiance map", HDR_radiance);
    }
    imwrite("recovered_HDR.exr", HDR);

    pair<Mat, Mat> after_tonemapping;
    
    if (parameter.find("-global") != parameter.end()) {
        cout << "running global tonemapping\n";
        after_tonemapping = global_operator(HDR);
    }
    else if (parameter.find("-local") != parameter.end()) {
        cout << "running local tonemapping\n";
        after_tonemapping = local_operator(HDR);
    }
    else if (parameter.find("-bilateral") != parameter.end()) {
        cout << "running bilateral tonemapping\n";
        after_tonemapping = bilateral_operator(HDR, true);
    }
    else if (parameter.find("-logarithmic") != parameter.end()) {
        cout << "running logarithmic tonemapping\n";
        after_tonemapping = logarithmic_operator(HDR);
    }
    else {
        cout << "running global tonemapping\n";
        after_tonemapping = global_operator(HDR);
    }
    imshow("After tonemapping", after_tonemapping.first);
    imwrite("result.png", after_tonemapping.second);
    cv::waitKey();
    return 0;
}