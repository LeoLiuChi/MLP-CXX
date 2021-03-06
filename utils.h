/*
* @Author: kmrocki
* @Date:   2016-02-24 10:47:03
* @Last Modified by:   kmrocki
* @Last Modified time: 2016-02-24 15:25:40
*/

#ifndef __UTILS_H__
#define __UTILS_H__

//set Matrix & Vector implementation
#include <Eigen/Dense>
typedef Eigen::MatrixXf Matrix;
typedef Eigen::VectorXf Vector;

#include <iostream>
#include <random>

//f(x) = sigm(x)
inline float __logistic(const float x) {
	return 1.0 / (1.0 +::expf(-x));
}

inline float __exponential(const float x) {
	return exp(x);
}

#ifdef USE_BLAS
#include <cblas.h>
void BLAS_mmul( Eigen::MatrixXf& __restrict c, Eigen::MatrixXf& __restrict a,
                Eigen::MatrixXf& __restrict b, bool aT = false, bool bT = false );

#endif

Matrix rectify(Matrix& x) {

	Matrix y(x.rows(), x.cols());

	for (int i = 0; i < x.rows(); i++) {
		for (int j = 0; j < x.cols(); j++) {

			y(i, j) = x(i, j) > 0 ? x(i, j) : 0;
		}
	}

	return y;

}

Matrix derivative_ReLU(Matrix& x) {

	Matrix y(x.rows(), x.cols());

	for (int i = 0; i < x.rows(); i++) {
		for (int j = 0; j < x.cols(); j++) {

			y(i, j) = x(i, j) > 0 ? 1 : 0;
		}
	}

	return y;

}

Matrix logistic(Matrix& x) {

	Matrix y(x.rows(), x.cols());

	for (int i = 0; i < x.rows(); i++) {
		for (int j = 0; j < x.cols(); j++) {

			y(i, j) = __logistic(x(i, j));
		}
	}

	return y;
}

Matrix softmax(Matrix& x) {

	Matrix y(x.rows(), x.cols());

	//probs(class) = exp(x, class)/sum(exp(x, class))

	Matrix e = x.unaryExpr(std::ptr_fun(::expf));

	Vector sum = e.colwise().sum();

	for (int i = 0; i < e.rows(); i++) {
		for (int j = 0; j < e.cols(); j++) {

			y(i, j) = e(i, j) / sum(j);
		}
	}

	return y;
}

float cross_entropy(Matrix& predictions, Matrix& targets) {

	float ce = 0.0;
	Matrix error(predictions.rows(), predictions.cols());

	//check what has happened and get information content for that event
	error.array() = -predictions.unaryExpr(std::ptr_fun(::logf)).array() * targets.array();
	ce = error.sum();

	return ce;
}

//generate an array of random numbers in range
void randi(Eigen::VectorXi& m, int range_min, int range_max) {

	std::random_device rd;
	std::mt19937 mt(rd());
	std::uniform_int_distribution<> dis(range_min, range_max);

	for (int i = 0; i < m.rows(); i++) {
		m(i) = (float)dis(mt);
	}

}

void randn(Matrix& m, float mean, float stddev) {

	std::random_device rd;
	std::mt19937 mt(rd());
	std::normal_distribution<> randn(mean, stddev);

	for (int i = 0; i < m.rows(); i++) {
		for (int j = 0; j < m.cols(); j++) {
			m(i, j) = randn(mt);
		}
	}

}

void linspace(Eigen::VectorXi& m, int range_min, int range_max) {

	for (int i = 0; i < m.rows(); i++) {
		m(i) = (float)(range_min + i);
	}

}

Matrix make_batch(std::deque<datapoint>& data, Eigen::VectorXi& random_numbers) {

	size_t batch_size = random_numbers.rows();
	Matrix batch(data[0].x.rows(), batch_size);

	for (size_t i = 0; i < batch_size; i++) {

		batch.col(i) = data[random_numbers(i)].x;

	}

	return batch;
}

Matrix make_targets(std::deque<datapoint>& data, Eigen::VectorXi& random_numbers, size_t classes) {

	size_t batch_size = random_numbers.rows();
	Matrix encoding = Matrix::Identity(classes, classes);
	Matrix batch(classes, batch_size);

	for (size_t i = 0; i < batch_size; i++) {

		batch.col(i) = encoding.col(data[random_numbers(i)].y);

	}

	return batch;
}

Eigen::VectorXi colwise_max_index(Matrix& m) {

	Eigen::VectorXi indices(m.cols());

	for (size_t i = 0; i < m.cols(); i++) {

		float current_max_val;
		int index;

		for (size_t j = 0; j < m.rows(); j++) {

			if (j == 0 || m(j, i) > current_max_val) {

				index = j;
				current_max_val = m(j, i);
			}

			indices(i) = index;

		}
	}

	return indices;
}

size_t count_zeros(Eigen::VectorXi m) {

	size_t zeros = 0;

	for (int i = 0; i < m.rows(); i++) {

		bool isZero = m(i) == 0;

		zeros += isZero;
	}

	return zeros;

}

size_t count_correct_predictions(Matrix& p, Matrix& t) {

	Eigen::VectorXi predicted_classes = colwise_max_index(p);
	Eigen::VectorXi target_classes = colwise_max_index(t);
	Eigen::VectorXi correct = (target_classes - predicted_classes);

	return count_zeros(correct);
}

#ifdef USE_BLAS
// c = a * b
void BLAS_mmul( Eigen::MatrixXf& __restrict c, Eigen::MatrixXf& __restrict a,
                Eigen::MatrixXf& __restrict b, bool aT, bool bT ) {

	enum CBLAS_TRANSPOSE transA = aT ? CblasTrans : CblasNoTrans;
	enum CBLAS_TRANSPOSE transB = bT ? CblasTrans : CblasNoTrans;

	size_t M = c.rows();
	size_t N = c.cols();
	size_t K = aT ? a.rows() : a.cols();

	float alpha = 1.0f;
	float beta = 1.0f;

	size_t lda = aT ? K : M;
	size_t ldb = bT ? N : K;
	size_t ldc = M;

	cblas_sgemm( CblasColMajor, transA, transB, M, N, K, alpha,
	             a.data(), lda,
	             b.data(), ldb, beta, c.data(), ldc );


}
#endif /* USE_BLAS */

#endif
