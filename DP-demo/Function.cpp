#include "stdafx.h"
#include "Function.h"


Function::Function()
{
}


Function::~Function()
{
}

float Function::func1(float x) {
	return x*x - 2 * x + 1;
}
float Function::grad1(float x) {
	return 2 * x - 2;
}
