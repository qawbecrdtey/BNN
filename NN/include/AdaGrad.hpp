#ifndef BNN_AdaGrad_hpp
#define BNN_AdaGrad_hpp

#include <Eigen/Dense>

namespace BNN
{
	class AdaGrad
	{
		double epsilon, delta;
		Eigen::Matrix<double, -1, -1> r;

		Eigen::Matrix<double, -1, -1> _calculate()
		{
			Eigen::Matrix<double, -1, -1> o(r.rows(), r.cols());
			for (std::size_t i = 0; i < r.rows(); i++)
			{
				for (std::size_t j = 0; j < r.cols(); j++)
				{
					o(i, j) = -epsilon / (delta + std::sqrt(r(i, j)));
				}
			}
			return o;
		}

	public:
		AdaGrad(std::size_t row = 0, std::size_t col = 0, double epsilon = 1, double delta = 0.0000001) : epsilon(epsilon), delta(delta), r(Eigen::Matrix<double, -1, -1>::Zero(row, col)) {}
		Eigen::Matrix<double, -1, -1> operator()(Eigen::Matrix<double, -1, -1> const &g)
		{
			assert(r.rows() == g.rows() && r.cols() == g.cols());
			r += g.cwiseProduct(g);
			return g.cwiseProduct(_calculate());
		}
	};
}

#endif