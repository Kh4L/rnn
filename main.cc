#include <iostream>
#include <fstream>
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

int main(int argc, const char ** argv)
{
	RNN rnn(10, 40, 30);
	std::ifstream datafs("input.txt");
	while (true) {
		std::vector<char> input = load(datafs, 31);
		for (char &c: input) {
			std::cout << c;
		}
	}
	return 0;
}
