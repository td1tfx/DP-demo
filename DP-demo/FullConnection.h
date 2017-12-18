#pragma once
class FullConnection
{
private:
	int m_in_num;
	int m_out_num;
	float* m_in_data;
	float* m_out_data;
	float** w;
	float* b;

	float* __sigmoid(float* in_data_t);

public:
	FullConnection();
	FullConnection(int in_num_t, int out_num_t, float lr = 0.01);
	~FullConnection();

	float* forward(float* in_data_t);
};

