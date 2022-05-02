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


double calculateError(const vector<Mat>& warp_imgs, const vector<vector<FeaturePoint>>& img_fps_list, int a_index, int b_index, int f_num) {
	int cur_x = img_fps_list[a_index][f_num].dx;
	int cur_y = img_fps_list[a_index][f_num].dy;
	int pre_f_num = img_fps_list[a_index][f_num].best_match[b_index];
	int pre_x = img_fps_list[b_index][pre_f_num].dx;
	int pre_y = img_fps_list[b_index][pre_f_num].dy;
	int m_x = warp_imgs[b_index].cols + cur_x - pre_x;
	//m_x = - m_x ;
	int m_y = cur_y - pre_y;
	//m_y = - m_y ; 

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
		double error = calculateError(warp_imgs, img_fps_list,a_index, b_index, i);
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

	//cout << "calculateMove :: min_error = " << min_error ;
	//cout << " , match point num = " << count << endl ;
	int move_x = warp_imgs[b_index].cols + c_x - p_x;
	int move_y = c_y - p_y;
	return make_pair(move_x, move_y);
}

Mat generateNewImage(vector<Mat>& warp_imgs, vector<int> img_order, vector<pair<int,int>> img_moves) {
	/* calculate image width & height */
	int img_width = warp_imgs[img_order[0]].cols;
	int y_bound = 100;
	int max_bound = numeric_limits<int>::min();
	int min_bound = numeric_limits<int>::max();
	int bound = 0;
	for (int i = 1; i < warp_imgs.size(); i++) {
		img_width += warp_imgs[img_order[i]].cols - round(img_moves[i].first);
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
	int img_height = warp_imgs[img_order[0]].rows + 2 * y_bound + 10;

	cout << "height width: " << img_height << " " << img_width << endl;
	/* create image */
	Mat result = Mat::zeros(img_height, img_width, CV_8UC3);
	/* first image , no need to move */

	for (int x = 0; x < warp_imgs[img_order[0]].cols; x++) {
		for (int y = 0; y < warp_imgs[img_order[0]].rows; y++) {
			int put_x = x;
			int put_y = y + y_bound;
			result.at<Vec3b>(put_y, put_x) = warp_imgs[img_order[0]].at<Vec3b>(y, x);
		}
	}
	int img_start_x = 0;
	int img_start_y = 0 + y_bound;
	int img_end_x = img_start_x + warp_imgs[img_order[0]].cols - 1;
	int img_end_y = img_start_y + warp_imgs[img_order[0]].rows - 1;
	/* generate the rest new image */
	cout << "move x y:" << endl;
	for (int i = 1; i < warp_imgs.size(); i++) {
		cout << img_moves[i].first << " ";
		cout << img_moves[i].second << " " << endl;
		for (int x = 0; x < warp_imgs[img_order[i]].cols; x++) {
			for (int y = 0; y < warp_imgs[img_order[i]].rows; y++) {

				int put_x = img_end_x + x + 1 - round(img_moves[i].first);
				int put_y = img_start_y + y + 1 - round(img_moves[i].second);
				if (x >= 0 && put_x <= img_end_x) {
					/* blending */
					int l_x = img_end_x + 1 - round(img_moves[i].first);
					int r_x = img_end_x;
					double len_x = (r_x - l_x);
					double a = (double)(put_x - l_x) / len_x;
					double b = (double)(r_x - put_x) / len_x;
					Vec3b pre_s = result.at<Vec3b>(put_y, put_x);
					Vec3b tmp_s = warp_imgs[img_order[i]].at<Vec3b>(y, x);
					Vec3b new_s;
					if (pre_s.val[0] == 0 && pre_s.val[1] == 0 && pre_s.val[2] == 0 && pre_s.val[3] == 0) {
						result.at<Vec3b>(put_y, put_x) = tmp_s;
					}
					else if (tmp_s.val[0] == 0 && tmp_s.val[1] == 0 && tmp_s.val[2] == 0 && tmp_s.val[3] == 0) {
						result.at<Vec3b>(put_y, put_x) = pre_s;
					}
					else {
						for (int i = 0; i < 3; i++) {
							new_s.val[i] = pre_s.val[i] * b + tmp_s.val[i] * a;
						}
						result.at<Vec3b>(put_y, put_x) = new_s;
					}
				}
				else {
					/* no need to blend */
					Vec3b s = warp_imgs[img_order[i]].at<Vec3b>(y, x);
					result.at<Vec3b>(put_y, put_x) = s;
				}
			}
		}
		img_end_x = img_end_x + warp_imgs[img_order[i]].cols - round(img_moves[i].first);
		img_start_y = img_start_y - round(img_moves[i].second);
	}
	return result;
}