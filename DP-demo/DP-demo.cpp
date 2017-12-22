// DP-demo.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "Operater.h"
#include "Function.h"
#include <stdlib.h>
#include "FullConnection.h"
#include <iostream>
#include "SquareLoss.h"


int main()
{
	//gradient descent
	//Operater op;
	//op.gradientDescent(-5, 0.2, Function::func1, Function::grad1, 20, 0.0001,0.0);

	//forward
	int in_num = 3;
	int out_num = 2;
	int batch_num = 3;
	FullConnection fc(in_num, out_num, batch_num);
	FullConnection fc1(out_num, out_num, batch_num);
	FullConnection fc2(out_num, out_num, batch_num);
	double** in_data = new double*[batch_num];
	in_data[0] = new double[in_num] {3, 5, 8};
	in_data[1] = new double[in_num] {2, 4, 3};
	in_data[2] = new double[in_num] {1, 3, 2};
	double** bench_data = new double*[batch_num];
	bench_data[0] = new double[out_num] {0.30, 0.50};
	bench_data[1] = new double[out_num] {0.50, 0.80};
	bench_data[2] = new double[out_num] {0.60, 0.90};
	std::cout << "input:" << std::endl;
	for (int i = 0; i < batch_num; i++) {
		for (int j = 0; j < in_num; j++) {
			std::cout << in_data[i][j] << " ";
		}
		std::cout << "; ";
	}
	std::cout << std::endl;
	std::cout << "benchput:" << std::endl;
	for (int i = 0; i < batch_num; i++) {
		for (int j = 0; j < out_num; j++) {
			std::cout << bench_data[i][j] << " ";
		}
		std::cout << ";";
	}
	std::cout << std::endl;
	SquareLoss loss;
	double mse = 1;
	int run_num = 0;
	double** out_data;
	double** loss_data;
	while (mse >= 0.001 && run_num < 3) {
		out_data = fc.forward(in_data);
		out_data = fc1.forward(out_data);
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
		mse = loss.forward(out_data, bench_data, out_num, batch_num);
		std::cout << "mse = " << mse << ";" << std::endl;
		loss_data = loss.backward();
		fc2.forward(loss_data);
		loss_data = loss.backward();
		fc1.forward(loss_data);
		loss_data = loss.backward();
		fc.forward(loss_data);
		loss_data = loss.backward();
		run_num++;
	}
	system("pause");
    return 0;
}

