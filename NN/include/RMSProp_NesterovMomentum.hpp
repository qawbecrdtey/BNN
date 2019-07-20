#ifndef BNN_RMSProp_NesterovMomentum_hpp
#define BNN_RMSProp_NesterovMomentum_hpp

#include <Eigen/Dense>

namespace BNN
{
	class RMSProp_NesterovMomentum
	{
		double epsilon, ro, alpha;
		Eigen::Matrix<double, -1, -1> velocity, r;

		Eigen::Matrix<double, -1, -1> _compute() const
		{
			Eigen::Matrix<double, -1, -1> o(r.rows(), r.cols());
			for (std::size_t i = 0; i < r.rows(); i++)
			{
				for (std::size_t j = 0; j < r.cols(); j++)
				{
					o(i, j) = -epsilon / std::sqrt(r(i, j));
				}
			}
			return o;
		}

	public:
		RMSProp_NesterovMomentum(std::size_t row = 0, std::size_t col = 0, double epsilon = 1, double ro = 1, double alpha = 1) : epsilon(epsilon), ro(ro), alpha(alpha), velocity(Eigen::Matrix<double, -1, -1>::Zero(row, col)), r(Eigen::Matrix<double, -1, -1>::Zero(row, col)) {}

		Eigen::Matrix<double, -1, -1> get_interm_param(Eigen::Matrix<double, -1, -1> const &param) const
		{
			assert(velocity.rows() == param.rows() && velocity.cols() == param.cols());
			return param + alpha * velocity;
		}

		Eigen::Matrix<double, -1, -1> const &update(Eigen::Matrix<double, -1, -1> const &g)
		{
			r = ro * r + (1 - ro) * g.cwiseProduct(g);
			return velocity = alpha * velocity + g.cwiseProduct(_compute());
		}
	};
}

#endif