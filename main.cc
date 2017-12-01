#include <iostream>
#include <fstream>
#include <set>
#include <map>
#include "rnn.h"

std::vector<char> load(std::ifstream &datafs, unsigned count, bool training)
{
    char *cinput = new char[count];
	datafs.read(cinput, count);
    //std::cout << datafs.gcount() << " chars read :\n";
    //std::cout << cinput << '\n';
    //std::cin.get();

    if (training && datafs.gcount() < count) {
  		datafs.clear();
		datafs.seekg(0, std::ios::beg);
    }

	std::vector<char> input;
    for (int i = 0; i < datafs.gcount(); ++i) {
       input.push_back(cinput[i]);
    }
    delete[] cinput;
	return input;
}

void addchars(std::set<char> &vocab, std::vector<char> &input)
{
	for (auto &c : input) {
		vocab.insert(c);
	}
}

void buildDicts(std::set<char> &vocab,
				std::vector<char> &intToChar,
				std::map<char, unsigned> &charToInt)
{
	intToChar.clear();
	charToInt.clear();
	unsigned i = 0;
	for (auto &c : vocab) {
		intToChar.push_back(c);
		charToInt.insert(std::pair<char,unsigned>(c, i));
		++i;
	}
}

int main(int argc, const char ** argv)
{
    const unsigned seqLen = 31;

	std::set<char> vocab;
	std::vector<char> intToChar;
	std::map<char, unsigned> charToInt;

	std::ifstream datafs("input.txt");
	while (!datafs.eof()) {
		std::vector<char> input = load(datafs, seqLen, false);
		addchars(vocab, input);
	}
	buildDicts(vocab, intToChar, charToInt);
	for (unsigned i = 0; i < intToChar.size(); ++i) {
		std::cout << i << " => " << intToChar[i] << std::endl;
		std::cout << intToChar[i] << " => " << charToInt[intToChar[i]] << std::endl;
	}

	RNN rnn(vocab.size(), 40, seqLen - 1);

    unsigned long long iter = 0;
	datafs.seekg(0, std::ios::beg);
	while (!datafs.eof()) {
		std::vector<char> seq = load(datafs, seqLen, true);
        if (seq.size() < seqLen)
            continue;

        if (iter % 500 == 0) {
            std::cout << "Iteration " << iter << std::endl;
            std::cout << std::string(seq.begin(), seq.end()) << std::endl;
            rnn.generate(0, 30);
        }

        // Building inputs and targets
		std::vector<unsigned> inputs;
		for (auto &c: seq){
			inputs.push_back(charToInt[c]);
		}
        std::vector<unsigned> targets = inputs;
        inputs.pop_back();
        targets.erase(targets.begin());

        rnn.forward(inputs);
        rnn.backProp(targets);
        rnn.update();
        iter++;
	}

	return 0;
}
