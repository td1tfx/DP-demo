#pragma once
class Operater
{
public:
	Operater();
	~Operater();

	typedef float functype(float);

	float gradientDescent(float x_start, float step, functype f, functype g, int epoch, float acc);
};

