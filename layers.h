/*
* @Author: kmrocki
* @Date:   2016-02-24 15:26:54
* @Last Modified by:   kmrocki
* @Last Modified time: 2016-02-24 15:27:38
*/

#ifndef __LAYERS_H__
#define __LAYERS_H__

#include <utils.h>

//abstract
class Layer {

public:

	//used in forward pass
	Matrix x; //inputs
	Matrix y; //outputs

	//grads, used in backward pass
	Matrix dx;
	Matrix dy;

	Layer(size_t inputs, size_t outputs, size_t batch_size) {

		x = Matrix(inputs, batch_size);
		y = Matrix(outputs, batch_size);
		dx = Matrix(inputs, batch_size);
		dy = Matrix(outputs, batch_size);

	};

	//need to override these
	virtual void forward() = 0;
	virtual void backward() = 0;
	virtual void resetGrads() {};
	virtual void applyGrads(float alpha) {};

	virtual ~Layer() {};

};

class Linear : public Layer {

public:

	Matrix W;
	Matrix b;

	Matrix dW;
	Matrix db;

	void forward() {

		y = b.replicate ( 1, x.cols() );
		BLAS_mmul ( y, W, x );

	}

	void backward() {

		dW.setZero();
		BLAS_mmul ( dW, dy, x, false, true );
		db = dy.rowwise().sum();
		dx.setZero();
		BLAS_mmul ( dx, W, dy, true, false );


	}

	Linear(size_t inputs, size_t outputs, size_t batch_size) : Layer(inputs, outputs, batch_size) {

		W = Matrix(outputs, inputs);
		b = Vector::Zero(outputs);
		randn(W, 0, 0.1);

	};

	void resetGrads() {

		dW = Matrix::Zero(W.rows(), W.cols());
		db = Vector::Zero(b.rows());
	}

	void applyGrads(float alpha) {

		b += alpha * db;
		W += alpha * dW;

	}

	~Linear() {};

};

class Sigmoid : public Layer {

public:

	void forward() {

		y = logistic(x);

	}

	void backward() {

		dx.array() = dy.array() * y.array() * (1.0 - y.array()).array();

	}

	Sigmoid(size_t inputs, size_t outputs, size_t batch_size) : Layer(inputs, outputs, batch_size) {};
	~Sigmoid() {};

};

class ReLU : public Layer {

public:

	void forward() {

		y = rectify(x);

	}

	void backward() {

		dx.array() = derivative_ReLU(y).array() * dy.array();

	}

	ReLU(size_t inputs, size_t outputs, size_t batch_size) : Layer(inputs, outputs, batch_size) {};
	~ReLU() {};

};

class Softmax : public Layer {

public:

	void forward() {

		y = softmax(x);

	}

	void backward() {

		dx = dy - y;
	}


	Softmax(size_t inputs, size_t outputs, size_t batch_size) : Layer(inputs, outputs, batch_size) {};
	~Softmax() {};

};

#endif
