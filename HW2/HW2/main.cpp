#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include <string>
#include <fstream>
#include <opencv2/opencv.hpp>


#include "ImGuiFileDialog.h"

#include "image_stitch.h"

#include "texture.h"

#include "gui.h"

using namespace std;
using namespace cv;



int main(int argc,char* argv[]) {
	if (argc > 1) {
		vector<string> filenames;
		Mat result;
		fstream img_list_file(argv[1]);
		string line;
		while (getline(img_list_file, line)) {
			if (line.empty()) continue;
			filenames.push_back(line);
		}
		result = image_stitch(filenames);
		cv::imshow("panorama result", result);
		cv::imwrite("result.png", result);
		return 0;
	}
	else {
		run_gui();
	}	
	return 0;
}