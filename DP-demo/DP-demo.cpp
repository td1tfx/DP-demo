// DP-demo.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "Operater.h"
#include "Function.h"
#include <stdlib.h>
#include "FullConnection.h"
#include <iostream>


int main()
{
	//gradient descent
	//Operater op;
	//op.gradientDescent(-5, 0.2, Function::func1, Function::grad1, 20, 0.0001,0.0);

	//forward
	int in_num = 3;
	int out_num = 2;
	float* in_data = new float[in_num] {3,5,8};
	std::cout << "input:" << std::endl;
	for (int i = 0; i < in_num; i++) {
		std::cout << in_data[i] << " ";
	}
	std::cout << std::endl;
	float* out_data;
	FullConnection fc(in_num,out_num);
	out_data = fc.forward(in_data);
	FullConnection fc1(out_num, out_num);
	out_data = fc1.forward(out_data);
	FullConnection fc2(out_num, out_num);
	out_data = fc2.forward(out_data);
	std::cout << "output:" << std::endl;
	for (int i = 0; i < out_num; i++) {
		std::cout << out_data[i] << " ";
	}
	std::cout << std::endl;
	delete in_data;

	system("pause");
    return 0;
}

