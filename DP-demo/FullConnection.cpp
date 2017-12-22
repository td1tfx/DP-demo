#include "stdafx.h"
#include "FullConnection.h"
#include <stdlib.h>
#include <math.h>


FullConnection::FullConnection()
{
	FullConnection(3, 2, 3, 0.1);
}

FullConnection::FullConnection(int in_num_t, int out_num_t,int batch_num_t, double lr) {
	m_in_num = in_num_t;
	m_out_num = out_num_t;
	m_batch_num = batch_num_t;
	m_lr = lr;
	w = new double*[in_num_t];
	for (int i = 0; i < in_num_t; i++) {
		w[i] = new double[out_num_t];
		for (int j = 0; j < out_num_t; j++) {
			w[i][j] = rand() % 100 / 100.00;
		}
	}
	b = new double[out_num_t];
	for (int j = 0; j < out_num_t; j++) {
		b[j] = 0;
	}

	m_in_data = new double*[m_batch_num];
	for (int i = 0; i < m_batch_num; i++) {
		m_in_data[i] = new double[m_in_num];
		for (int j = 0; j < m_in_num; j++) {
			m_in_data[i][j] = 0;
		}
	}
	m_out_data = new double*[m_batch_num];
	for (int i = 0; i < m_batch_num; i++) {
		m_out_data[i] = new double[m_out_num];
		for (int j = 0; j < m_out_num; j++) {
			m_out_data[i][j] = 0;
		}
	}
	m_residual_z = new double*[m_batch_num];
	for (int i = 0; i < m_batch_num; i++) {
		m_residual_z[i] = new double[m_out_num];
		for (int j = 0; j < m_out_num; j++) {
			m_residual_z[i][j] = 0;
		}
	}

	m_residual_x = new double*[m_batch_num];
	for (int i = 0; i < m_batch_num; i++) {
		m_residual_x[i] = new double[m_in_num];
		for (int j = 0; j < m_in_num; j++) {
			m_residual_x[i][j] = 0;
		}
	}

	m_grad_b = new double[m_out_num];
	for (int i = 0; i < m_out_num; i++) {
		m_grad_b[i] = 0;
	}
	m_grad_w = new double*[in_num_t];
	for (int i = 0; i < in_num_t; i++) {
		m_grad_w[i] = new double[out_num_t];
		for (int j = 0; j < out_num_t; j++) {
			m_grad_w[i][j] = 0;
		}
	}
}


FullConnection::~FullConnection()
{
	if (w != NULL) {
		for (int i = 0; i < m_in_num; i++) {
			delete[] w[i];
			w[i] = NULL;
		}
		delete[] w;
		w = NULL;
	}
	if (m_grad_w != NULL) {
		for (int i = 0; i < m_in_num; i++) {
			delete[] m_grad_w[i];
			m_grad_w[i] = NULL;
		}
		delete[] m_grad_w;
		m_grad_w = NULL;
	}
	if (b != NULL) {
		delete[] b;
		b = NULL;
	}
	if (m_grad_b != NULL) {
		delete[] m_grad_b;
		m_grad_b = NULL;
	}
	if (m_in_data != NULL) {
		for (int i = 0; i < m_batch_num; i++) {
			delete[] m_in_data[i];
			m_in_data[i] = NULL;
		}
		delete[] m_in_data;
		m_in_data = NULL;
	}
	if (m_residual_z != NULL) {
		for (int i = 0; i < m_batch_num; i++) {
			delete[] m_residual_z[i];
			m_residual_z[i] = NULL;
		}
		delete[] m_residual_z;
		m_residual_z = NULL;
	}
	if (m_residual_x != NULL) {
		for (int i = 0; i < m_batch_num; i++) {
			delete[] m_residual_x[i];
			m_residual_x[i] = NULL;
		}
		delete[] m_residual_x;
		m_residual_x = NULL;
	}
	if (m_out_data != NULL) {
		for (int i = 0; i < m_batch_num; i++) {
			delete[] m_out_data[i];
			m_out_data[i] = NULL;
		}
		delete[] m_out_data;
		m_out_data = NULL;
	}
}

double** FullConnection::__sigmoid(double** in_data_t) {
	for (int i = 0; i < m_batch_num; i++) {
		for(int j = 0; j<m_out_num;j++)
		in_data_t[i][j] = 1 / (1 + exp(-in_data_t[i][j]));
	}
	return in_data_t;
}

double** FullConnection::forward(double** in_data_t) {
	memcpy(m_in_data, in_data_t, m_in_num * m_batch_num * sizeof(double));
	for (int i = 0; i < m_batch_num; i++) {
		for (int j = 0; j < m_out_num; j++) {
			//double test = w[j][i];
			//double test1 = w[j][i] * in_data_t[j];
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

double** FullConnection::backward(double** loss_t) {
	for (int i = 0; i < m_batch_num; i++) {
		for (int j = 0; j < m_out_num; j++) {
			double test0 = m_out_data[i][j];
			double test1 = loss_t[i][j];
			m_residual_z[i][j] = loss_t[i][j] * m_out_data[i][j] * (1 - m_out_data[i][j]);
			m_grad_b[i] += m_residual_z[i][j];
			for (int h = 0; h < m_in_num; h++) {
				double test2 = m_in_data[h][i];
				m_grad_w[h][j] = m_in_data[h][i] * m_residual_z[i][j];
				double test3 = w[h][j] -= m_lr*m_grad_w[h][j];
				m_residual_x[i][h] += m_residual_z[i][j] * w[h][j];
			}
		}
		b[i] -= m_lr*m_grad_b[i];
	}
	return m_residual_x;

}