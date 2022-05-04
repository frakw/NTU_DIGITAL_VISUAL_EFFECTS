#include "combine.h"
using namespace std;
using namespace cv;
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


double calc_error(const vector<Mat>& warp_imgs, const vector<vector<FeaturePoint>>& img_fps_list, int a_index, int b_index, int f_num) {
	int cur_x = img_fps_list[a_index][f_num].dx;
	int cur_y = img_fps_list[a_index][f_num].dy;
	int pre_f_num = img_fps_list[a_index][f_num].best_match[b_index];
	int pre_x = img_fps_list[b_index][pre_f_num].dx;
	int pre_y = img_fps_list[b_index][pre_f_num].dy;
	int m_x = warp_imgs[b_index].cols + cur_x - pre_x;
	int m_y = cur_y - pre_y;
	double total_diff = 0.0;
	for (int i = 0; i < img_fps_list[a_index].size(); i++) {
		if (img_fps_list[a_index][i].best_match[b_index] == -1 || (double)img_fps_list[a_index][i].dx > warp_imgs[a_index].cols * 0.6)
			continue;
		int index = img_fps_list[a_index][i].best_match[b_index];
		int c_x = img_fps_list[a_index][i].dx;
		c_x = c_x + warp_imgs[b_index].cols;
		int c_y = img_fps_list[a_index][i].dy;
		int p_x = img_fps_list[b_index][index].dx;
		int p_y = img_fps_list[b_index][index].dy;
		int diff_x = (c_x - m_x) - p_x;
		int diff_y = (c_y - m_y) - p_y;
		double diff = sqrt(diff_x * diff_x + diff_y * diff_y);
		total_diff += diff;
	}
	return total_diff;
}


pair<int, int> get_two_img_move(const vector<Mat>& warp_imgs, const vector<vector<FeaturePoint>>& img_fps_list, int a_index, int b_index) {
	double min_error = 0.0;
	double min_index;
	bool first = true;
	int count = 0;
	for (int i = 0; i < img_fps_list[a_index].size(); i++) {
		if (img_fps_list[a_index][i].best_match[b_index] == -1 || (double)img_fps_list[a_index][i].dx > warp_imgs[a_index].cols * 0.6)
			continue;
		double error = calc_error(warp_imgs, img_fps_list,a_index, b_index, i);
		if (first) {
			min_error = error;
			min_index = i;
			first = false;
		}
		if (error < min_error) {
			min_error = error;
			min_index = i;
		}
		count++;
	}
	int index = img_fps_list[a_index][min_index].best_match[b_index];
	int c_x = img_fps_list[a_index][min_index].dx;
	int c_y = img_fps_list[a_index][min_index].dy;
	int p_x = img_fps_list[b_index][index].dx;
	int p_y = img_fps_list[b_index][index].dy;

	int move_x = warp_imgs[b_index].cols + c_x - p_x;
	int move_y = c_y - p_y;
	return make_pair(move_x, move_y);
}

Mat get_panorama(vector<Mat>& warp_imgs, vector<int> img_order, vector<pair<int,int>> img_moves) {
	int result_width = warp_imgs[img_order[0]].cols;
	int y_bound = 100;
	int max_bound = numeric_limits<int>::min();
	int min_bound = numeric_limits<int>::max();
	int bound = 0;
	for (int i = 1; i < warp_imgs.size(); i++) {
		result_width += warp_imgs[img_order[i]].cols - round(img_moves[i].first);
		bound += round(img_moves[i].second);
		if (bound > max_bound)
			max_bound = bound;
		if (bound < min_bound)
			min_bound = bound;
	}
	if (abs(max_bound) > abs(min_bound))
		y_bound = abs(max_bound);
	else
		y_bound = abs(min_bound);
	int result_height = warp_imgs[img_order[0]].rows + 2 * y_bound + 10;

	Mat result = Mat::zeros(result_height, result_width, CV_8UC3);

	for (int x = 0; x < warp_imgs[img_order[0]].cols; x++) {
		for (int y = 0; y < warp_imgs[img_order[0]].rows; y++) {
			int resultx = x;
			int resulty = y + y_bound;
			result.at<Vec3b>(resulty, resultx) = warp_imgs[img_order[0]].at<Vec3b>(y, x);
		}
	}
	int startx = 0;
	int starty = 0 + y_bound;
	int endx = startx + warp_imgs[img_order[0]].cols - 1;
	int endy = starty + warp_imgs[img_order[0]].rows - 1;
	Vec3b black(0, 0, 0);
	for (int i = 1; i < warp_imgs.size(); i++) {
		for (int x = 0; x < warp_imgs[img_order[i]].cols; x++) {
			for (int y = 0; y < warp_imgs[img_order[i]].rows; y++) {

				int resultx = endx + x + 1 - round(img_moves[i].first);
				int resulty = starty + y + 1 - round(img_moves[i].second);
				if (x >= 0 && resultx <= endx) {
					int l_x = endx + 1 - round(img_moves[i].first);
					int r_x = endx;
					double len_x = (r_x - l_x);
					double a = (double)(resultx - l_x) / len_x;
					double b = (double)(r_x - resultx) / len_x;
					Vec3b pre = result.at<Vec3b>(resulty, resultx);
					Vec3b current = warp_imgs[img_order[i]].at<Vec3b>(y, x);
					if (pre == black) {
						result.at<Vec3b>(resulty, resultx) = current;
					}
					else if (current == black) {
						result.at<Vec3b>(resulty, resultx) = pre;
					}
					else {
						for (int i = 0; i < 3; i++) {
							result.at<Vec3b>(resulty, resultx)[i] = pre[i] * b + current[i] * a;
						}
					}
				}
				else {
					result.at<Vec3b>(resulty, resultx) = warp_imgs[img_order[i]].at<Vec3b>(y, x);
				}
			}
		}
		endx = endx + warp_imgs[img_order[i]].cols - round(img_moves[i].first);
		starty = starty - round(img_moves[i].second);
	}
	return result;
}