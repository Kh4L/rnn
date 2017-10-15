#pragma once
#include <vector>
#include <cmath>
#include <Eigen/Dense>

using Matrix = Eigen::MatrixXd;

class RNN
{
public:
	RNN(unsigned lexicSize, unsigned hiddenSize, unsigned seqLength);
	void forward(std::vector<unsigned> &inputs);
	void backProp(std::vector<unsigned> &targets);
	void update();

	static double tanh(double x)
	{
		return std::tanh(x);
	}

	static double exp(double x)
	{
		return std::exp(x);
	}

private:
	// hyperparameters
	unsigned lexicSize = 26;
	unsigned hiddenSize = 100;  // size of hidden layer of neuron
	unsigned seqLength = 30;  // number of steps to unroll the RNN for
	double learningRate = 1e-1;

	// parameters
	Matrix Wxh;
	Matrix Whh;
	Matrix Why;
	Matrix bh;
	Matrix by;

	Matrix hprev;

	// gradients
	Matrix dWxh;
	Matrix dWhh;
	Matrix dWhy;
	Matrix dbh;
	Matrix dby;

	// rnn pipeline
	std::vector<Matrix> hs; // hiddenstate
	std::vector<Matrix> xs; // one hot vectors
	std::vector<Matrix> ys; // unormalized log prob for next
	std::vector<Matrix> ps; // prob for next
};
