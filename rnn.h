#pragma once
#include <vector>
#include <cmath>
#include <Eigen/Dense>
#include <random>

using Matrix = Eigen::MatrixXd;

class RNN
{
public:
	RNN(unsigned lexicSize, unsigned hiddenSize, unsigned seqLength);
	void forward(std::vector<unsigned> &inputs);
	void backProp(std::vector<unsigned> &targets);
	void update();
    std::vector<unsigned> generate(unsigned seed, unsigned int iter);

	static double tanh(double x)
	{
		return std::tanh(x);
	}

	static double dtanh(double x)
	{
		double t = std::tanh(x);
		return 1 - t * t;
	}

	static double exp(double x)
	{
		return std::exp(x);
	}

    static double adagradInv(double x)
    {
        return 1 / std::sqrt(x + 1e-8);
    }
	static double clip(double x)
	{
		if (x > 5)
			return 5;
		else if (x < -5)
			return -5;
		else
			return x;
	}

private:
	void adagrad(Matrix& param, Matrix& dparam, Matrix& mem);

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

	// memory for adagrad
	Matrix mWxh;
	Matrix mWhh;
	Matrix mWhy;
	Matrix mbh;
	Matrix mby;

  std::random_device rd;
  std::mt19937 gen;

};
