#include "stdafx.h"
#include "FullConnection.h"
#include <stdlib.h>
#include <math.h>


FullConnection::FullConnection()
{
	FullConnection(3, 2);
}

FullConnection::FullConnection(int in_num_t, int out_num_t, float lr) {
	m_in_num = in_num_t;
	m_out_num = out_num_t;
	w = new float*[in_num_t];
	for (int i = 0; i < in_num_t; i++) {
		w[i] = new float[out_num_t];
		for (int j = 0; j < out_num_t; j++) {
			w[i][j] = rand() % 100 / 100.00;
		}
	}
	b = new float[out_num_t];
	for (int j = 0; j < out_num_t; j++) {
		b[j] = 0;
	}

	m_out_data = new float[m_out_num] {0};
}


FullConnection::~FullConnection()
{
	if (w != NULL) {
		for (int i = 0; i < m_in_num; i++) {
			delete w[i];
			w[i] = NULL;
		}
		delete[m_in_num] w;
		w = NULL;
	}
	if (b != NULL) {
		delete[m_out_num] b;
		b = NULL;
	}
	if (m_out_data != NULL) {
		delete[m_out_num] m_out_data;
	}
}

float* FullConnection::__sigmoid(float* in_data_t) {
	for (int i = 0; i < m_in_num; i++) {
		in_data_t[i] = 1 / (1 + exp(-in_data_t[i]));
	}
	return in_data_t;
}

float* FullConnection::forward(float* in_data_t) {
	for (int i = 0; i < m_out_num; i++) {
		for (int j = 0; j < m_in_num; j++) {
			float test = w[j][i];
			float test1 = w[j][i] * in_data_t[j];
			m_out_data[i] = m_out_data[i] += w[j][i] * in_data_t[j];
		}
		m_out_data[i] += b[i];
	}
	return __sigmoid(m_out_data);
}