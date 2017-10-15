#include "rnn.h"
#include <iostream>

RNN::RNN(unsigned lexicSize, unsigned hiddenSize, unsigned seqLength)
	:lexicSize(lexicSize),
		hiddenSize(hiddenSize),
		seqLength(seqLength),
		learningRate(1e-1),
		Wxh(Matrix::Random(hiddenSize,lexicSize) * 0.01),
		Whh(Matrix::Random(hiddenSize, hiddenSize) * 0.01),
		Why(Matrix::Random(lexicSize, hiddenSize) * 0.01),
		bh(Matrix::Zero(hiddenSize, 1)),
		by(Matrix::Zero(lexicSize, 1)),
		dWxh(Matrix::Zero(hiddenSize, lexicSize)),
		dWhh(Matrix::Zero(hiddenSize, hiddenSize)),
		dWhy(Matrix::Zero(lexicSize, hiddenSize)),
		dbh(Matrix::Zero(hiddenSize, 1)), 
		dby(Matrix::Zero(lexicSize, 1))
{
	std::cout << Wxh.unaryExpr(&RNN::tanh) << std::endl;
	std::cout << std::endl;
	std::cout << bh << std::endl;
}

void RNN::forward(std::vector<unsigned> &inputs) 
{
	hs.push_back(hprev);
	for (unsigned i = 0; i < inputs.size(); ++i) {
		xs.push_back(Matrix::Zero(lexicSize, 1));
		xs.back()(inputs[i], 0) = 1;
		hs.push_back((Wxh * xs.back() + Whh * hs.back() + bh).unaryExpr(&RNN::tanh));
		ys.push_back(Why * hs.back() + by);
		Matrix ysExp = ys.back().unaryExpr(&RNN::exp);
		ps.push_back(ysExp / ysExp.sum());
	}

	// save hprev for next forward pass
	hprev = hs.back();
}

void RNN::backProp(std::vector<unsigned> &targets)
{
	dWxh.setZero();
	dWhh.setZero(); 
	dWhy.setZero(); 
	dbh.setZero(); 
	dby.setZero(); 
	
}

void RNN::update()
{

	xs.clear();
	hs.clear();
	ys.clear();
	ps.clear();
}
