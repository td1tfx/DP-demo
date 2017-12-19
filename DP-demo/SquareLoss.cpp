#include "stdafx.h"
#include "SquareLoss.h"


SquareLoss::SquareLoss(int batch_num_t)
{
	batch_num = batch_num_t;
	loss = new float[batch_num] {0};
}


SquareLoss::~SquareLoss()
{

}

float* SquareLoss::forward(float** y, float** t, int out_num) {

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