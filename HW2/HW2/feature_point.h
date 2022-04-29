#ifndef _FEATURE_POINT_H_
#define _FEATURE_POINT_H_
#include <vector>
using namespace std;
//class FeaturePoint {
//	
//public:
//	int row;
//	int col;
//	int octave; //哪個octave
//	int layer_index;//在該octave中的位置(index)
//
//	int x;
//	int y;
//	double sigma;
//	double extremum_val;
//	vector<uint8_t> descriptor;
//
//
//
//	bool valid = false;
//	FeaturePoint() {}
//	FeaturePoint(int _row, int _col, int _octave, int _layer_index):row(_row),col(_col),octave(_octave),layer_index(_layer_index) {
//		descriptor.reserve(128);
//	}
//};
#define FEATURE_ELEMENT_LENGTH 128
class FeaturePoint {
public:
	int    octave;                                        //【1】關鍵點所在組
	int    interval;                                      //【2】關鍵點所在層
	double offset_interval;                               //【3】調整後的層的增量

	int    x;                                             //【4】x,y座標,根據octave和interval可取的層內影象
	int    y;
	double scale;                                         //【5】空間尺度座標scale = sigma0*pow(2.0, o+s/S)

	double dx;                                            //【6】特徵點座標，該座標被縮放成原影象大小 
	double dy;

	double offset_x;
	double offset_y;

	//============================================================
	//1---高斯金字塔組內各層尺度座標，不同組的相同層的sigma值相同
	//2---關鍵點所在組的組內尺度
	//============================================================
	double octave_scale;                                  //【1】offset_i;
	double ori;                                           //【2】方向
	int    descr_length;
	double descriptor[FEATURE_ELEMENT_LENGTH];            //【3】特徵點描述符            
	double val;
};

#endif