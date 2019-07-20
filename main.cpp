#include <iostream>
#include <random>
#include <Eigen/Dense>

#include "Network.hpp"

int main()
{
	constexpr std::size_t input_size = 2;
	constexpr std::size_t output_size = 1;

	std::array<std::size_t, 4> arr = { input_size, 2, 2, output_size };
	BNN::Network network(arr);
	std::unique_ptr<Eigen::Matrix<double, -1, 1>[]> input, output;

	constexpr std::size_t testcase = 4;

	input = std::make_unique<Eigen::Matrix<double, -1, 1>[]>(testcase);
	output = std::make_unique<Eigen::Matrix<double, -1, 1>[]>(testcase);
	/*
	std::random_device rd;
	std::mt19937_64 e2(rd());
	std::uniform_real_distribution<double> dist(-1, 1);
	*/
	for (std::size_t i = 0; i < testcase; i++)
	{
		input[i] = Eigen::Matrix<double, -1, 1>(input_size);
		input[i](0) = i % 2;
		input[i](1) = i / 2;
		/*
		for (std::size_t j = 0; j < input_size; j++)
		{
			input[i](j) = dist(e2);
		}
		*/
		output[i] = Eigen::Matrix<double, -1, 1>(output_size);
		std::cout << (output[i](0) = ((i / 2) + 4 * (i % 2))) << std::endl;
	}
	std::size_t cnt = 0;
A:
	cnt++;
	for (std::size_t i = 0; i < testcase; i++)
	{
		//std::cout << "input #" << i << std::endl;
		//std::cout << network.forwardprop(input[i]) << std::endl;
		network.forwardprop(input[i]);
		//std::cout << "network " << i << ":\n";
		//std::cout << network << '\n' << std::endl;
		network.backprop(output[i]);
	}
	if (cnt == 30000) goto B;
	goto A;
B:
	for (std::size_t i = 0; i < testcase; i++)
	{
		std::cout << "input #" << i << std::endl;
		std::cout << network.forwardprop(input[i]) << std::endl;
		std::cout << network << '\n' << std::endl;
	}
}