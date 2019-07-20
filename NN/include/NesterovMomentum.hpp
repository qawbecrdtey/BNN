#ifndef BNN_NesterovMomentum_hpp
#define BNN_NesterovMomentum_hpp

#include <Eigen/Dense>

namespace BNN
{
	class NesterovMomentum
	{
		Eigen::Matrix<double, -1, -1> velocity;
		double alpha, epsilon;

	public:
		NesterovMomentum(std::size_t row = 0, std::size_t col = 0, double alpha = 1, double epsilon = 0) : alpha(alpha), epsilon(epsilon), velocity(Eigen::Matrix<double, -1, -1>::Zero(row, col)) {}
		Eigen::Matrix<double, -1, -1> get_interm_param(Eigen::Matrix<double, -1, -1> const &param) const
		{
			assert(velocity.rows() == param.rows() && velocity.cols() == param.cols());
			return param + alpha * velocity;
		}
		Eigen::Matrix<double, -1, -1> const &update(Eigen::Matrix<double, -1, -1> const &g)
		{
			assert(velocity.rows() == g.rows() && velocity.cols() == g.cols());
			return velocity = alpha * velocity - epsilon * g;
		}
	};
}

#endif