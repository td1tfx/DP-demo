#pragma once
class SquareLoss
{
public:
	float* loss;
	int batch_num;

	SquareLoss(int batch_num_t);
	~SquareLoss();
	
	float* forward(float** y, float** t, int out_num);
	float* backward();

};

