#include "stdafx.h"
#include "SquareLoss.h"


SquareLoss::SquareLoss()
{

}


SquareLoss::~SquareLoss()
{

}

double SquareLoss::forward(double** y, double** t, int out_num, int batch_num_t) {

	if (loss != NULL) {
		delete[] loss;
	}
	batch_num = batch_num_t;
	loss = new double*[batch_num];
	for (int i = 0; i < batch_num; i++) {
		loss[i] = new double[out_num];
		for (int j = 0; j < out_num; j++) {
			loss[i][j] = y[i][j] - t[i][j];
		}
	}
	double loss_t = 0;
	for (int i = 0; i < batch_num; i++) {
		for (int j = 0; j < out_num; j++) {
			loss_t += (y[i][j] - t[i][j]) * (y[i][j] - t[i][j]); 
		}
	}
	return loss_t/ out_num / batch_num/2;
}

double** SquareLoss::backward() {

	return loss;

}