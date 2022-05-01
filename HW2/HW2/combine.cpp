#include "combine.h"
//假設圖片由左而右
int get_right_img_index(const Mat& img, const vector<FeaturePoint>& img_fps,int img_count) {
	vector<int> macth_count(img_count);
	for (int i = 0; i < img_fps.size(); i++) {
		for (int j = 0; j < img_fps[i].best_match.size(); j++) {
			if (img_fps[i].best_match[j] != -1 && (double)img_fps[i].dx > img.cols * 0.5) {
				macth_count[j]++;
			}
		}
	}
	int max_count = numeric_limits<int>::min();
	int index = 0;
	for (int i = 0; i < macth_count.size(); i++) {
		if (macth_count[i] > max_count) {
			max_count = macth_count[i];
			index = i;
		}
	}
	return index;
}

vector<int> get_image_order(const vector<Mat>& imgs, const vector<vector<FeaturePoint>>& img_fps_list) {
	vector<int> result;
	vector<int> img_right_indexs;
	int img_count = imgs.size();
	for (int i = 0; i < img_count; i++) {
		img_right_indexs.push_back(get_right_img_index(imgs[i], img_fps_list[i],img_count));
	}
	/* calculate the serial of all image */

	

	vector<bool> notfirst;
	notfirst.resize(img_count,false);
	int first_index = 0;
	for (int i = 0; i < img_right_indexs.size(); i++) {
		int index = img_right_indexs[i];
		notfirst[index] = true;
	}
	//第一張圖，沒有其他圖片設定它為右邊
	for (int i = 0; i < img_right_indexs.size(); i++) {
		if (notfirst[i] == false) {
			first_index = i;
		}
	}
	int count = 1;
	result.push_back(first_index);
	int index = first_index;
	while (count < img_right_indexs.size()) {
		index = img_right_indexs[index];
		result.push_back(index);
		count++;
	}

	return result;
}