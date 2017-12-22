#include "stdafx.h"
#include "SquareLoss.h"


SquareLoss::SquareLoss()
{

}


SquareLoss::~SquareLoss()
{

}

float* SquareLoss::forward(float** y, float** t, int out_num, int batch_num_t) {

	if (loss != NULL) {
		delete[] loss;
	}
	batch_num = batch_num_t;
	loss = new float[batch_num] {0};
	for (int i = 0; i < batch_num; i++) {
		float loss_t = 0;
		for (int j = 0; j < out_num; j++) {
			loss_t += (y[i][j] - t[i][j]) * (y[i][j] - t[i][j]);
		}
		loss[i] = loss_t / out_num / 2;
	}
	return loss;
}

float* SquareLoss::backward() {

	return loss;

}