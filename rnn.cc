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
        hprev(Matrix::Zero(hiddenSize, 1)),
		dWxh(Matrix::Zero(hiddenSize, lexicSize)),
		dWhh(Matrix::Zero(hiddenSize, hiddenSize)),
		dWhy(Matrix::Zero(lexicSize, hiddenSize)),
		dbh(Matrix::Zero(hiddenSize, 1)),
		dby(Matrix::Zero(lexicSize, 1)),
		mWxh(Matrix::Zero(hiddenSize,lexicSize)),
		mWhh(Matrix::Zero(hiddenSize, hiddenSize)),
		mWhy(Matrix::Zero(lexicSize, hiddenSize)),
		mbh(Matrix::Zero(hiddenSize, 1)),
		mby(Matrix::Zero(lexicSize, 1))
{
	std::cout << Wxh.unaryExpr(&RNN::tanh) << std::endl;
	std::cout << std::endl;
	std::cout << bh << std::endl;
}

void RNN::forward(std::vector<unsigned> &inputs)
{
	hs.push_back(hprev);
	for (int i = 0; i < static_cast<int>(inputs.size()); ++i) {
        //Matrix x = Matrix::Zero(lexicSize, 1);

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

	Matrix dhnext = Matrix::Zero(hiddenSize, 1);
	for (int i = targets.size() - 1; i >= 0; --i) {
		Matrix dy = ps[i];
		dy(targets[i], 0) -= 1;
		dWhy += dy * hs[i + 1].transpose();
		dby += dy;
		Matrix dh = Why.transpose() * dy + dhnext;
		Matrix dhraw = hs[i + 1].unaryExpr(&RNN::dtanh).cwiseProduct(dh);
		dbh += dhraw;
		dWxh += dhraw * xs[i].transpose();
		dWhh += dhraw * hs[i].transpose();
		dhnext = Whh.transpose() * dhraw;
	}

	dWxh.unaryExpr(&RNN::clip);
	dWhh.unaryExpr(&RNN::clip);
	dWhy.unaryExpr(&RNN::clip);
	dbh.unaryExpr(&RNN::clip);
	dby.unaryExpr(&RNN::clip);
}

void RNN::adagrad(Matrix& param, Matrix& dparam, Matrix& mem)
{
	mem += dparam.cwiseProduct(dparam);
    // alpha * dparam / (mem + 1e-8)
	param += (-learningRate * dparam).cwiseProduct(mem.unaryExpr(&RNN::adagradInv));
}

void RNN::update()
{
	adagrad(Wxh, dWxh, mWxh);
	adagrad(Whh, dWhh, mWhh);
	adagrad(Why, dWhy, mWhy);
	adagrad(bh, dbh, mbh);
	adagrad(by, dby, mby);

	xs.clear();
	hs.clear();
	ys.clear();
	ps.clear();
}

std::vector<unsigned> RNN::generate(unsigned seed, unsigned iter)
{
    std::vector<unsigned> out;
    Matrix x = Matrix::Zero(lexicSize, 1);
    x(seed, 0) = 1;
    Matrix h = hprev;
    for (unsigned i = 0; i < iter; ++i) {
        h = Wxh * x + Whh * h + bh;
        h = h.unaryExpr(&RNN::tanh);
        Matrix y = Why * h + by;
        Matrix yexp = y.unaryExpr(&RNN::exp);
        Matrix p = yexp / yexp.sum();
        if (i == 0)
            std::cout << p;
        //ix = np.random.choice(range(vocab_size), p=p.ravel())
        //x = np.zeros((vocab_size, 1))
        //x[ix] = 1
        //ixes.append(ix)
    }

    return out;
}

