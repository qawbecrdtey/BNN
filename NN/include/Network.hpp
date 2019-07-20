#ifndef BNN_Network_hpp
#define BNN_Network_hpp

#include <array>
#include <cassert>
#include <cmath>
#include <functional>
#include <iostream>
#include <memory>
#include <variant>

#include <AdaGrad.hpp>
#include <Adam.hpp>
#include <Momentum.hpp>
#include <NesterovMomentum.hpp>
#include <RMSProp.hpp>
#include <RMSProp_NesterovMomentum.hpp>

#include <Eigen/Dense>

namespace BNN {
	/*
	enum class Opt_enum
	{
		AdaGrad,
		Adam,
		Momentum,
		NesterovMomentum,
		RMSProp,
		RMSProp_NesterovMomentum,
	};

	using Optimizer = std::variant<AdaGrad, Adam, Momentum, NesterovMomentum, RMSProp, RMSProp_NesterovMomentum>;
	*/
	// template<Opt_enum optimizer_type>
	class Network {
		std::size_t depth;
		std::unique_ptr<Eigen::Matrix<double, -1, 1>[]> layer;
		std::unique_ptr<Eigen::Matrix<double, -1, 1>[]> z;
		std::unique_ptr<Eigen::Matrix<double, -1, -1>[]> weight;
		std::unique_ptr<Eigen::Matrix<double, -1, 1>[]> bias;
		std::function<Eigen::Matrix<double, -1, 1>(Eigen::Matrix<double, -1, 1>)> outactfunc;
		std::function<Eigen::Matrix<double, -1, 1>(Eigen::Matrix<double, -1, 1>)> hidactfunc;

		std::unique_ptr<Adam[]> momentum_bias;
		std::unique_ptr<Adam[]> momentum_weight;

		static constexpr double alpha_constant = 0.001;

#define ELU [](double x) -> double { return x > 0 ? x : (std::exp(x) - 1); }
#define ID [](double x) -> double { return x; }
#define RELU [](double x) -> double { return x < 0 ? 0 : x; }
#define SIGMOID [](double x) -> double { return 1 / (1 + std::exp(-x)); }
#define TANH [](double x) -> double { return std::tanh(x); }

#define ELUMAP [](Eigen::Matrix<double, -1, 1> const &x) -> Eigen::Matrix<double, -1, 1> { Eigen::Matrix<double, -1, 1> o(x.rows()); for(std::size_t i = 0; i < x.rows(); i++) o(i) = ELU(x(i)); return o; }
#define IDMAP [](Eigen::Matrix<double, -1, 1> const &x) -> Eigen::Matrix<double, -1, 1> { return x; }
#define RELUMAP [](Eigen::Matrix<double, -1, 1> const &x) -> Eigen::Matrix<double, -1, 1> { Eigen::Matrix<double, -1, 1> o(x); for (std::size_t i = 0; i < x.rows(); i++) if(o(i) < 0) o(i) = 0; return o; }
#define SIGMOIDMAP [](Eigen::Matrix<double, -1, 1> const &x) -> Eigen::Matrix<double, -1, 1> { Eigen::Matrix<double, -1, 1> o(x); for (std::size_t i = 0; i < x.rows(); i++) o(i) = SIGMOID(o(i)); return o; }
#define TANHMAP [](Eigen::Matrix<double,-1,1> const &x) -> Eigen::Matrix<double, -1, 1> { Eigen::Matrix<double, -1, 1> o(x); for(std::size_t i = 0; i < x.rows(); i++) o(i) = std::tanh(o(i)); return o; }

#define D_ELUMAP [](Eigen::Matrix<double, -1, 1> const &x) -> Eigen::Matrix<double, -1, 1> { Eigen::Matrix<double, -1, 1> o(x.rows()); for(std::size_t i = 0; i < x.rows(); i++) o(i) = (x(i) < 0 ? std::exp(x(i)) : 1); return o; }
#define D_IDMAP [](Eigen::Matrix<double, -1, 1> const &x) -> Eigen::Matrix<double, -1, 1> { Eigen::Matrix<double, -1, 1> o(x.rows()); for (std::size_t i = 0; i < x.rows(); i++) o(i) = 1; return o; }
#define D_RELUMAP [](Eigen::Matrix<double, -1, 1> const &x) -> Eigen::Matrix<double, -1, 1> { Eigen::Matrix<double, -1, 1> o(x); for (std::size_t i = 0; i < x.rows(); i++) o(i) = (o(i) > 0 ? 1 : (o(i) < 0 ? 0 : 0.5)); return o; }
#define D_SIGMOIDMAP [](Eigen::Matrix<double, -1, 1> const &x) -> Eigen::Matrix<double, -1, 1> { Eigen::Matrix<double, -1, 1> o(x); for(std::size_t i = 0; i < x.rows(); i++) { double const t = std::exp(-o(i)); o(i) = t / ((1 + t) * (1 + t)); } return o; }
#define D_TANHMAP [](Eigen::Matrix<double, -1, 1> const &x) -> Eigen::Matrix<double, -1, 1> { Eigen::Matrix<double, -1, 1> o(x); for(std::size_t i = 0; i < x.rows(); i++) { double const t = std::tanh(o(i)); o(i) = 1 - t * t; } return o; }


		double loss(Eigen::Matrix<double, -1, 1> const &result, Eigen::Matrix<double, -1, 1> const &expected_result)
		{
			
			assert(result.rows() == expected_result.rows());
			/*
			double l = 0;
			for (std::size_t i = 0; i < result.rows(); i++)
			{
				l -= expected_result[i] * std::log(result[i]) + (1 - expected_result[i]) * std::log(1 - result[i]);
			}
			return l;
			*/
			double l = 0;
			for (std::size_t i = 0; i < result.rows(); i++)
			{
				l += (expected_result(i) - result(i)) * (expected_result(i) - result(i));
			}
			l /= result.rows();
			return l;
		}
		Eigen::Matrix<double, -1, 1> d_loss(Eigen::Matrix<double, -1, 1> const &result, Eigen::Matrix<double, -1, 1> const &expected_result)
		{
			assert(result.rows() == expected_result.rows());
			/*
			Eigen::Matrix<double, -1, 1> dl(result.rows());
			for (std::size_t i = 0; i < result.rows(); i++)
			{
				dl[i] = (result[i] - expected_result[i]) / ((1 - result[i]) * result[i]);
			}
			return dl;
			*/
			Eigen::Matrix<double, -1, 1> dl(result.rows());
			for (std::size_t i = 0; i < result.rows(); i++)
			{
				dl[i] = 2 * (result(i) - expected_result(i)) / result.rows();
			}
			return dl;
		}

		void fp(Eigen::Matrix<double, -1, 1> const &input)
		{
			layer[0] = input;
			//std::cout << "input =\n" << input << std::endl;
			for (std::size_t i = 0; i < depth; i++)
			{
				layer[i + 1] = hidactfunc(z[i] = weight[i].transpose() * layer[i] + bias[i]);
			}
			layer[depth] = outactfunc(z[depth - 1] = weight[depth - 1].transpose() * layer[depth - 1] + bias[depth - 1]);
		}

		void bp(Eigen::Matrix<double, -1, 1> const &result, Eigen::Matrix<double, -1, 1> const &expected_result)
		{
			//constexpr double lambda = 0.09;
			auto delta = std::make_unique<Eigen::Matrix<double, -1, 1>[]>(depth);
			delta[depth - 1] = d_loss(result, expected_result).cwiseProduct(D_IDMAP(z[depth - 1]));
			for (std::size_t i = depth - 2; i < depth; i--)
			{
				delta[i] = weight[i + 1] * delta[i + 1].cwiseProduct(D_ELUMAP(z[i + 1]));
			}
			for (std::size_t i = depth - 1; i < depth; i--)
			{
				constexpr double lambda = 0.01;
				bias[i] += momentum_bias[i](delta[i] + lambda * bias[i]);
				weight[i] += momentum_weight[i](layer[i] * static_cast<Eigen::Matrix<double, -1, -1>>(delta[i]).transpose() + lambda * weight[i]);
				//bias[i] -= delta[i] + lambda * bias[i];
				//std::cout << static_cast<Eigen::Matrix<double, -1, -1> >(delta[i]).transpose() << std::endl;
				//weight[i] -= layer[i] * static_cast<Eigen::Matrix<double, -1, -1> >(delta[i]).transpose() + lambda * weight[i];
			}
		}

	public:
		Network()
			: depth(0),
			layer(nullptr),
			z(nullptr),
			weight(nullptr),
			bias(nullptr),
			outactfunc(IDMAP),
			hidactfunc(IDMAP),
			momentum_bias(nullptr),
			momentum_weight(nullptr)
		{}
		template<std::size_t DEPTHP>
		Network(std::array<std::size_t, DEPTHP> info)
			: depth(DEPTHP - 1),
			layer(std::make_unique<Eigen::Matrix<double, -1, 1>[]>(DEPTHP)),
			z(std::make_unique<Eigen::Matrix<double, -1, 1>[]>(DEPTHP - 1)),
			weight(std::make_unique<Eigen::Matrix<double, -1, -1>[]>(DEPTHP - 1)),
			bias(std::make_unique<Eigen::Matrix<double, -1, 1>[]>(DEPTHP - 1)),
			outactfunc(IDMAP),
			hidactfunc(ELUMAP),
			momentum_bias(std::make_unique<Adam[]>(DEPTHP - 1)),
			momentum_weight(std::make_unique<Adam[]>(DEPTHP - 1))
		{
			for (std::size_t i = 0; i <= depth; i++)
			{
				layer[i] = Eigen::Matrix<double, -1, 1>(info[i]);
			}
			for (std::size_t i = 0; i < depth; i++)
			{
				weight[i] = Eigen::Matrix<double, -1, -1>::Random(info[i], info[i + 1]) * std::sqrt(6 / (info[i] + info[i + 1]));
				bias[i] = Eigen::Matrix<double, -1, 1>::Random(info[i + 1]);
				momentum_weight[i] = Adam(info[i], info[i + 1], 0.001, 0.5);
				momentum_bias[i] = Adam(info[i + 1], 1, 0.001, 0.5);
			}
		}

		Eigen::Matrix<double, -1, 1> const &forwardprop(Eigen::Matrix<double, -1, 1> const &input)
		{
			assert(input.rows() == layer[0].rows());
			fp(input);
			return layer[depth];
		}

		void backprop(Eigen::Matrix<double, -1, 1> const &expected_result)
		{
			//std::cout << "loss = " << loss(layer[depth], expected_result) << std::endl;
			bp(layer[depth], expected_result);
		}

		friend std::ostream &operator<<(std::ostream &os, Network const &network)
		{
			std::cout << network.layer[0] << std::endl;
			for (std::size_t i = 0; i < network.depth; i++)
			{
				std::cout << "weight =\n" << network.weight[i] << std::endl;
				std::cout << "bias =\n" << network.bias[i] << '\n' << std::endl;
				std::cout << "layer =\n" << network.layer[i + 1] << std::endl;
			}
			return os;
		}

#undef ID
#undef RELU
#undef SIGMOID
#undef TANH

#undef IDMAP
#undef RELUMAP
#undef SIGMOIDMAP
#undef TANHMAP

#undef D_IDMAP
#undef D_RELUMAP
#undef D_SIGMOIDMAP
#undef D_TANHMAP
	};
}

#endif