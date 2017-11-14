#include <iostream>
#include <fstream>
#include <set>
#include <map>
#include "rnn.h"

std::vector<char> load(std::ifstream &datafs, unsigned count)
{
	if (datafs.eof()) {
  		datafs.clear();
		datafs.seekg(0, std::ios::beg);
	}
	std::vector<char> input (count);
	datafs.read(&input[0], count);
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
	RNN rnn(10, 40, 30);

	std::set<char> vocab;
	std::vector<char> intToChar;
	std::map<char, unsigned> charToInt;

	std::ifstream datafs("input.txt");
	while (!datafs.eof()) {
		std::vector<char> input = load(datafs, 31);
		for (auto &c : input) {
			std::cout << c;
		}
		addchars(vocab, input);
	}
	buildDicts(vocab, intToChar, charToInt);
	for (unsigned i = 0; i < intToChar.size(); ++i) {
		std::cout << i << " => " << intToChar[i] << std::endl;
		std::cout << intToChar[i] << " => " << charToInt[intToChar[i]] << std::endl;
	}

	datafs.seekg(0, std::ios::beg);
	while (!datafs.eof()) {
		std::vector<char> seq = load(datafs, 31);
		std::vector<unsigned> inputs;
		for (auto &c: seq){
			inputs.push_back(charToInt[c]);
		}
        std::vector<unsigned> targets = inputs;
        inputs.pop_back();
        targets.erase(targets.begin());

        rnn.forward(inputs);
        rnn.backProp(targets);
	}

	return 0;
}
