#include "image_stitch.h"

Mat image_stitch(vector<string> filenames) {
	int img_count = filenames.size();
	Mat result;
	vector<Mat> imgs(img_count);
	for (int i = 0; i < filenames.size(); i++) {
		imgs[i] = imread(filenames[i]);
	}
	vector<vector<FeaturePoint>> img_fps_list(img_count);
	for (int i = 0; i < img_count; i++) {
		img_fps_list[i] = SIFT(imgs[i]);
	}
	featureMatch(img_fps_list);
	//for (int i = 0; i < img_fps_list.size(); i++) {
	//	for (int j = 0; j < img_fps_list[i].size(); j++) {
	//		cout << "size:" << img_fps_list[i][j].best_match.size() << " [0]: "<< img_fps_list[i][j].best_match[0] << " [1]: " << img_fps_list[i][j].best_match[1] <<endl;
	//	}
	//	cout << "next img:" << endl;
	//}
	vector<Mat> warp_imgs(img_count);
	for (int i = 0; i < filenames.size(); i++) {
		warp_imgs[i] = cylindrical_warping2(imgs[i], img_fps_list[i]);
	}

	//圖片由左而右
	//cout << "get_left_img_index" << get_right_img_index(imgs[0], img_fps_list[0],img_count);
	vector<int> img_order = get_image_order(warp_imgs, img_fps_list);



	cout << "img_order" << endl;
	for (int i : img_order) cout << i << " ";
	int x = 0;

	cout << "img_move" << endl;

	vector<pair<int, int>> img_moves(img_count);
	for (int i = 1; i < warp_imgs.size(); i++) {
		img_moves[i] = get_two_img_move(warp_imgs, img_fps_list, img_order[i], img_order[i - 1]);
		cout << img_moves[i].first << " " << img_moves[i].second << endl;
	}
	cout << endl;

	generateNewImage(warp_imgs, img_order,img_moves);
	//Mat match_result = draw_matches2(warp_imgs[0], warp_imgs[1], img_fps_list[0], img_fps_list[1]);
	//imshow("match", match_result);
	//imshow("feature point", draw_keypoints(imgs[1], img_fps_list[1],3));
	return result;
}