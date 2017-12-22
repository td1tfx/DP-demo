#include "stdafx.h"
#include "Function.h"


Function::Function()
{
}


Function::~Function()
{
}

double Function::func1(double x) {
	return x*x - 2 * x + 1;
}
double Function::grad1(double x) {
	return 2 * x - 2;
}
