#pragma once
class SquareLoss
{
public:
	double** loss;
	int batch_num;

	SquareLoss();
	~SquareLoss();
	
	double forward(double** y, double** t, int out_num, int batch_num_t);
	double** backward();

};

