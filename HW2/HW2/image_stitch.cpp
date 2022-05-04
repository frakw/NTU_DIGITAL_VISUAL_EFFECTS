#include "image_stitch.h"

using namespace std;
using namespace cv;

Mat image_stitch(vector<string> filenames, int limit_size) {
	Mat result;
	int img_count = filenames.size();
	if (img_count == 0) return result;
	else if (img_count == 1) return imread(filenames[0]);

	vector<Mat> imgs(img_count);
	for (int i = 0; i < filenames.size(); i++) {
		imgs[i] = imread(filenames[i]);
		int new_rows, new_cols;
		if (imgs[i].cols > imgs[i].rows) {
			new_rows = limit_size * imgs[i].rows / imgs[i].cols;
			new_cols = limit_size;
		}
		else if (imgs[i].cols < imgs[i].rows) {
			new_cols = limit_size * imgs[i].cols / imgs[i].rows;
			new_rows = limit_size;
		}
		else {
			new_rows = limit_size;
			new_cols = limit_size;
		}
		resize(imgs[i], imgs[i], Size(new_cols, new_rows));
	}
	vector<vector<FeaturePoint>> img_fps_list(img_count);
	for (int i = 0; i < img_count; i++) {
		cout << "running image" << i << " SIFT...\n";
		img_fps_list[i] = SIFT(imgs[i]);
		cout << "find " << img_fps_list[i].size() << " feature points\n";
	}
	cout << "feature point matching...\n";
	match_feature_points(img_fps_list);
	cout << "match finished\n";
	vector<Mat> warp_imgs(img_count);
	cout << "warping image...\n";
	for (int i = 0; i < filenames.size(); i++) {
		warp_imgs[i] = cylindrical_warping(imgs[i], img_fps_list[i]);
	}
	cout << "warping finished\n";
	cout << "combine image...\n";
	vector<int> img_order = get_image_order(warp_imgs, img_fps_list);
	vector<pair<int, int>> img_moves(img_count);
	for (int i = 1; i < warp_imgs.size(); i++) {
		img_moves[i] = get_two_img_move(warp_imgs, img_fps_list, img_order[i], img_order[i - 1]);
	}
	cout << "combine finished\n";

	result = get_panorama(warp_imgs, img_order, img_moves);
	return result;
}