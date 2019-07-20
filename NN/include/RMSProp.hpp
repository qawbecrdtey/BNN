#ifndef BNN_RMSProp_hpp
#define BNN_RMSProp_hpp

#include <Eigen/Dense>

namespace BNN
{
	class RMSProp
	{
		double epsilon, ro, delta;
		Eigen::Matrix<double, -1, -1> r;

		Eigen::Matrix<double, -1, -1> _compute() const
		{
			Eigen::Matrix<double, -1, -1> o(r.rows(), r.cols());
			for (std::size_t i = 0; i < r.rows(); i++)
			{
				for (std::size_t j = 0; j < r.cols(); j++)
				{
					o(i, j) = -epsilon / std::sqrt(delta + r(i, j));
				}
			}
			return o;
		}

	public:
		RMSProp(std::size_t row = 0, std::size_t col = 0, double epsilon = 1, double ro = 1, double delta = 0.0000001) : epsilon(epsilon), ro(ro), delta(delta), r(Eigen::Matrix<double, -1, -1>::Zero(row, col)) {}

		Eigen::Matrix<double, -1, -1> operator()(Eigen::Matrix<double, -1, -1> const &g)
		{
			assert(r.rows() == g.rows() && r.cols() == g.cols());
			r = ro * r + (1 - ro) * g.cwiseProduct(g);
			return g.cwiseProduct(_compute());
		}
	};
}

#endif