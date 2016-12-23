/*
* @Author: kmrocki
* @Date:   2016-02-24 09:43:05
* @Last Modified by:   kmrocki
* @Last Modified time: 2016-02-24 16:36:27
*/

#include <importer.h>
#include <utils.h>
#include <layers.h>
#include <nn.h>
#include <iostream>


int main() {

	size_t epochs = 10;
	size_t batch_size = 250;
	double learning_rate = 1e-3;

	NN nn(batch_size);

	//a simple NN for classification (should give around 97.6% accuracy)
	//MNIST - 28x28 -> 400 -> 256 -> 100 -> 10

	nn.layers.push_back(new Linear(28 * 28, 400, batch_size));
	nn.layers.push_back(new ReLU(400, 400, batch_size));
	nn.layers.push_back(new Linear(400, 256, batch_size));
	nn.layers.push_back(new ReLU(256, 256, batch_size));
	nn.layers.push_back(new Linear(256, 100, batch_size));
	nn.layers.push_back(new ReLU(100, 100, batch_size));
	nn.layers.push_back(new Linear(100, 10, batch_size));
	nn.layers.push_back(new Softmax(10, 10, batch_size));

	//[60000, 784]
	std::deque<datapoint> train_data =
		MNISTImporter::importFromFile("data/mnist/train-images-idx3-ubyte",
									  "data/mnist/train-labels-idx1-ubyte");
	//[10000, 784]
	std::deque<datapoint> test_data =
		MNISTImporter::importFromFile("data/mnist/t10k-images-idx3-ubyte",
									  "data/mnist/t10k-labels-idx1-ubyte");

	for (size_t e = 0; e < epochs; e++) {

		std::cout << "Epoch " << e + 1 << std::endl << std::endl;
		nn.train(train_data, learning_rate, train_data.size() / batch_size);
		nn.test(test_data);

	}

}