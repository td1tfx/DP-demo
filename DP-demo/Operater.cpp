#include "stdafx.h"
#include "Operater.h"
#include <iostream>


Operater::Operater()
{
}


Operater::~Operater()
{
}

float Operater::gradientDescent(float x_start, float step, functype f, functype g, int epoch, float acc, float mom_dis) {
	float x = x_start;
	float y = 0;
	float grad = 0;
	float pre_grad = 0;
	for (int i = 0; i < epoch; i++) {
		grad = g(x);
		pre_grad = pre_grad * mom_dis + grad * step;
		x -= pre_grad;
		y = f(x);
		std::cout << "Epoch=" << i << ";grad=" << grad << ";x=" << x << ";y=" << y << std::endl;
		if (abs(grad) < acc) {
			break;
		}
	}
	return x;
}
