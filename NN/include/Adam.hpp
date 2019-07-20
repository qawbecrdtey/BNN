#ifndef BNN_Adam_hpp
#define BNN_Adam_hpp

#include <Eigen/Dense>

namespace BNN
{
	class Adam
	{
		double epsilon, ro1, ro2, delta;
		double pro1, pro2;
		Eigen::Matrix<double, -1, -1> s, r;

		Eigen::Matrix<double, -1, -1> _compute()
		{
			Eigen::Matrix<double, -1, -1> os = s / (1 - (pro1 = pro1 * ro1));
			Eigen::Matrix<double, -1, -1> or = r / (1 - (pro2 = pro2 * ro2));
			Eigen::Matrix<double, -1, -1> o(r.rows(), r.cols());
			
			for (std::size_t i = 0; i < r.rows(); i++)
			{
				for (std::size_t j = 0; j < r.cols(); j++)
				{
					o(i, j) = -epsilon * s(i, j) / std::sqrt(r(i, j) + delta);
				}
			}

			return o;
		}

	public:
		Adam(std::size_t row = 0, std::size_t col = 0, double epsilon = 0.001, double ro1 = 0.9, double ro2 = 0.999, double delta = 0.0000001)
			: epsilon(epsilon), ro1(ro1), ro2(ro2), delta(delta), pro1(1), pro2(1),
			s(Eigen::Matrix<double, -1, -1>::Zero(row, col)),
			r(Eigen::Matrix<double, -1, -1>::Zero(row, col))
		{}

		Eigen::Matrix<double, -1, -1> operator()(Eigen::Matrix<double, -1, -1> const &g)
		{
			s = ro1 * s + (1 - ro1) * g;
			r = ro2 * r + (1 - ro2) * g.cwiseProduct(g);
			return _compute();
		}
	};
}

#endif