#ifndef BNN_Momentum_hpp
#define BNN_Momentum_hpp

#include <Eigen/Dense>

namespace BNN
{
	class Momentum
	{
		Eigen::Matrix<double, -1, -1> velocity;
		double alpha, epsilon;

	public:
		Momentum(std::size_t row = 0, std::size_t col = 0, double alpha = 1, double epsilon = 1) : alpha(alpha), epsilon(epsilon), velocity(Eigen::Matrix<double, -1, -1>::Zero(row, col)) {}

		Eigen::Matrix<double, -1, -1> const &operator()(Eigen::Matrix<double, -1, -1> const &g)
		{
			assert(velocity.rows() == g.rows() && velocity.cols() == g.cols());
			return velocity = alpha * velocity - epsilon * g;
		}
	};
}

#endif