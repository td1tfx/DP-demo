#include "stdafx.h"
#include "Operater.h"
#include <iostream>


Operater::Operater()
{
}


Operater::~Operater()
{
}

double Operater::gradientDescent(double x_start, double step, functype f, functype g, int epoch, double acc, double mom_dis) {
	double x = x_start;
	double y = 0;
	double grad = 0;
	double pre_grad = 0;
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
