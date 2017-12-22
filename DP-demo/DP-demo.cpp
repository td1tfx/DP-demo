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
	int batch_num = 3;
	float** in_data = new float*[batch_num];
	in_data[0] = new float[in_num] {3, 5, 8};
	in_data[1] = new float[in_num] {2, 4, 3};
	in_data[2] = new float[in_num] {1, 3, 2};
	std::cout << "input:" << std::endl;
	for (int i = 0; i < batch_num; i++) {
		for (int j = 0; j < in_num; j++) {
			std::cout << in_data[i][j] << " ";
		}
		std::cout << "; ";
	}
	std::cout << std::endl;
	float** out_data;
	FullConnection fc(in_num,out_num, batch_num);
	out_data = fc.forward(in_data);
	FullConnection fc1(out_num, out_num,batch_num);
	out_data = fc1.forward(out_data);
	FullConnection fc2(out_num, out_num,batch_num);
	out_data = fc2.forward(out_data);
	std::cout << "output:" << std::endl;
	for (int i = 0; i < batch_num; i++) {
		for (int j = 0; j < out_num; j++) {
			std::cout << out_data[i][j] << " ";
		}
		std::cout << ";";
	}
	std::cout << std::endl;


	//delete in_data;

	system("pause");
    return 0;
}

