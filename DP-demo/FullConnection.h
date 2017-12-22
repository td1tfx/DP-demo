#pragma once
class FullConnection
{
private:
	int m_in_num;
	int m_out_num;
	int m_batch_num;
	double m_lr;
	double** m_in_data;
	double** m_out_data;
	double** w;
	double** m_grad_w;
	double* b;
	double* m_grad_b;
	double** m_residual_z;
	double** m_residual_x;

	double** __sigmoid(double** in_data_t);

public:
	FullConnection();
	FullConnection(int in_num_t, int out_num_t, int batch_num_t, double lr = 0.01);
	~FullConnection();

	double** forward(double** in_data_t);
	double** backward(double* loss_t);
};

