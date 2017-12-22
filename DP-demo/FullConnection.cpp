#include "stdafx.h"
#include "FullConnection.h"
#include <stdlib.h>
#include <math.h>


FullConnection::FullConnection()
{
	FullConnection(3, 2, 3);
}

FullConnection::FullConnection(int in_num_t, int out_num_t,int batch_num_t, float lr) {
	m_in_num = in_num_t;
	m_out_num = out_num_t;
	m_batch_num = batch_num_t;
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

	m_in_data = new float*[m_batch_num];
	for (int i = 0; i < m_batch_num; i++) {
		m_in_data[i] = new float[m_in_num];
		for (int j = 0; j < m_in_num; j++) {
			m_in_data[i][j] = 0;
		}
	}
	m_out_data = new float*[m_batch_num];
	for (int i = 0; i < m_batch_num; i++) {
		m_out_data[i] = new float[m_out_num];
		for (int j = 0; j < m_out_num; j++) {
			m_out_data[i][j] = 0;
		}
	}
	m_residual_z = new float*[m_batch_num];
	for (int i = 0; i < m_batch_num; i++) {
		m_residual_z[i] = new float[m_out_num];
		for (int j = 0; j < m_out_num; j++) {
			m_residual_z[i][j] = 0;
		}
	}

	m_residual_x = new float*[m_batch_num];
	for (int i = 0; i < m_batch_num; i++) {
		m_residual_x[i] = new float[m_in_num];
		for (int j = 0; j < m_in_num; j++) {
			m_residual_x[i][j] = 0;
		}
	}

	m_grad_b = new float[m_out_num] {0};
	m_grad_w = new float*[in_num_t];
	for (int i = 0; i < in_num_t; i++) {
		m_grad_w[i] = new float[out_num_t];
		for (int j = 0; j < out_num_t; j++) {
			m_grad_w[i][j] = 0;
		}
	}
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
	if (m_grad_w != NULL) {
		for (int i = 0; i < m_in_num; i++) {
			delete m_grad_w[i];
			m_grad_w[i] = NULL;
		}
		delete[m_in_num] m_grad_w;
		m_grad_w = NULL;
	}
	if (b != NULL) {
		delete[m_out_num] b;
		b = NULL;
	}
	if (m_grad_b != NULL) {
		delete[m_out_num] m_grad_b;
		m_grad_b = NULL;
	}
	if (m_in_data != NULL) {
		for (int i = 0; i < m_batch_num; i++) {
			delete m_in_data[i];
			m_in_data[i] = NULL;
		}
		delete[m_in_num] m_in_data;
		m_in_data = NULL;
	}
	if (m_residual_z != NULL) {
		for (int i = 0; i < m_batch_num; i++) {
			delete m_residual_z[i];
			m_residual_z[i] = NULL;
		}
		delete[m_out_num] m_in_data;
		m_in_data = NULL;
	}
// 	if (m_out_data != NULL) {
// 		delete[m_out_num] m_out_data;
// 	}
}

float** FullConnection::__sigmoid(float** in_data_t) {
	for (int i = 0; i < m_batch_num; i++) {
		for(int j = 0; j<m_out_num;j++)
		in_data_t[i][j] = 1 / (1 + exp(-in_data_t[i][j]));
	}
	return in_data_t;
}

float** FullConnection::forward(float** in_data_t) {
	memcpy(m_in_data, in_data_t, m_in_num * m_batch_num * sizeof(float));
	for (int i = 0; i < m_batch_num; i++) {
		for (int j = 0; j < m_out_num; j++) {
			//float test = w[j][i];
			//float test1 = w[j][i] * in_data_t[j];
			for (int h = 0; h < m_in_num; h++) {
				for (int f = 0; f < m_out_num; f++) {
					m_out_data[i][j] = m_out_data[i][j] += w[h][f] * in_data_t[i][h];
				}
			}
			m_out_data[i][j] += b[j];
		}
	}
	__sigmoid(m_out_data);
	return m_out_data;
}

float* FullConnection::backward(float* loss_t) {
// 	for (int i = 0; i < m_out_num; i++) {
// 		m_residual_z[i] = loss_t[i] * m_out_data[i] * (1 - m_out_data[i]);
// 		m_grad_b[i] = m_residual_z[i];
// 		b[i] -= m_grad_b[i];
// 		m_residual_x[i] = 0;
// 		for (int j = 0; j < m_in_num; j++) {
// 			m_grad_w[i][j] = m_in_data[j] * m_residual_z[i];
// 			w[i][j] -= m_grad_w[i][j];
// 			//m_residual_x[j] += w[i][j] * m_residual_z[i];
// 		}
// 	}

	return NULL;

}