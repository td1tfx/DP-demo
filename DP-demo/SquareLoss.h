#pragma once
class SquareLoss
{
public:
	float* loss;
	int batch_num;

	SquareLoss();
	~SquareLoss();
	
	float* forward(float** y, float** t, int out_num, int batch_num_t);
	float* backward();

};

