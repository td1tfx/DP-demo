// DP-demo.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "Operater.h"
#include "Function.h"
#include <stdlib.h>


int main()
{

	Operater op;
	op.gradientDescent(-5, 0.2, Function::func1, Function::grad1, 20, 0.0001,0.0);

	system("pause");
    return 0;
}

