#ifndef _FEATURE_POINT_H_
#define _FEATURE_POINT_H_
#include <vector>
using namespace std;
//class FeaturePoint {
//	
//public:
//	int row;
//	int col;
//	int octave; //����octave
//	int layer_index;//�b��octave������m(index)
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
	int    octave;                                        //�i1�j�����I�Ҧb��
	int    interval;                                      //�i2�j�����I�Ҧb�h
	double offset_interval;                               //�i3�j�վ�᪺�h���W�q

	int    x;                                             //�i4�jx,y�y��,�ھ�octave�Minterval�i�����h���v�H
	int    y;
	double scale;                                         //�i5�j�Ŷ��ث׮y��scale = sigma0*pow(2.0, o+s/S)

	double dx;                                            //�i6�j�S�x�I�y�СA�Ӯy�гQ�Y�񦨭�v�H�j�p 
	double dy;

	double offset_x;
	double offset_y;

	//============================================================
	//1---�������r��դ��U�h�ث׮y�СA���P�ժ��ۦP�h��sigma�ȬۦP
	//2---�����I�Ҧb�ժ��դ��ث�
	//============================================================
	double octave_scale;                                  //�i1�joffset_i;
	double ori;                                           //�i2�j��V
	int    descr_length;
	double descriptor[FEATURE_ELEMENT_LENGTH];            //�i3�j�S�x�I�y�z��            
	double val;
};

#endif