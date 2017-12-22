#pragma once
class Operater
{
public:
	Operater();
	~Operater();

	typedef double functype(double);

	double gradientDescent(double x_start, double step, functype f, functype g, int epoch, double acc, double mom_dis);
};

