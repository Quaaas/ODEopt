#include <functional>
#include <eigen3/Eigen/Dense>
#include <vector>
#include <iostream>
#include "util.hh"
#include <stack>

class ODEopt {
public:
	ODEopt(
			std::function<double(const ::Eigen::VectorXd&, const ::Eigen::VectorXd&)> g,
			std::function<::Eigen::VectorXd(const ::Eigen::VectorXd&, const ::Eigen::VectorXd&)> dyg,
			std::function<::Eigen::VectorXd(const ::Eigen::VectorXd&, const ::Eigen::VectorXd&)> dug,
			std::function<::Eigen::MatrixXd(const ::Eigen::VectorXd&, const ::Eigen::VectorXd&)> dyyg,
			std::function<::Eigen::MatrixXd(const ::Eigen::VectorXd&, const ::Eigen::VectorXd&)> duug,
			std::function<::Eigen::MatrixXd(const ::Eigen::VectorXd&, const ::Eigen::VectorXd&)> dyug,
			std::function<::Eigen::VectorXd(const ::Eigen::VectorXd&, const ::Eigen::VectorXd&)> f,
			std::function<::Eigen::MatrixXd(const ::Eigen::VectorXd&, const ::Eigen::VectorXd&)> duf,
			std::function<::Eigen::MatrixXd(const ::Eigen::VectorXd&, const ::Eigen::VectorXd&)> dyf,
			std::function<::Eigen::MatrixXd(const ::Eigen::VectorXd&, const ::Eigen::VectorXd&, const ::Eigen::VectorXd&)> pduuf,
			std::function<::Eigen::MatrixXd(const ::Eigen::VectorXd&, const ::Eigen::VectorXd&, const ::Eigen::VectorXd&)> pduyf,
			std::function<::Eigen::MatrixXd(const ::Eigen::VectorXd&, const ::Eigen::VectorXd&, const ::Eigen::VectorXd&)> pdyyf,
			std::function<::Eigen::VectorXd(const ::Eigen::VectorXd&, const ::Eigen::VectorXd&)> r,
			std::function<::Eigen::MatrixXd(const ::Eigen::VectorXd&, const ::Eigen::VectorXd&)> dar,
			std::function<::Eigen::MatrixXd(const ::Eigen::VectorXd&, const ::Eigen::VectorXd&)> dbr,
			Eigen::VectorXd c,
			std::vector<double> grid,
			int dim_y,
			int dim_u,
			int dim_r
	);


	// collocation points
	Eigen::VectorXd c_;

	// grid
	std::vector<double> grid_;

	// number of gridpoints
	int N_grid_;

	// order of ODE
	int dim_y_;

	// dimension of control variable
	int dim_u_;

	// number of collocation points
	int N_col_;

	// number of boundary conditions
	int dim_r_;



	// functions for OC problem with ODEs

	// cost function
	std::function<double(const ::Eigen::VectorXd&, const ::Eigen::VectorXd&)> g_;
	std::function<::Eigen::VectorXd(const ::Eigen::VectorXd&, const ::Eigen::VectorXd&)> dyg_;
	std::function<::Eigen::VectorXd(const ::Eigen::VectorXd&, const ::Eigen::VectorXd&)> dug_;
	std::function<::Eigen::MatrixXd(const ::Eigen::VectorXd&, const ::Eigen::VectorXd&)> dyyg_;
	std::function<::Eigen::MatrixXd(const ::Eigen::VectorXd&, const ::Eigen::VectorXd&)> duug_;
	std::function<::Eigen::MatrixXd(const ::Eigen::VectorXd&, const ::Eigen::VectorXd&)> dyug_;

	// ODE
	std::function<::Eigen::VectorXd(const ::Eigen::VectorXd&, const ::Eigen::VectorXd&)> f_;
	std::function<::Eigen::MatrixXd(const ::Eigen::VectorXd&, const ::Eigen::VectorXd&)> duf_;
	std::function<::Eigen::MatrixXd(const ::Eigen::VectorXd&, const ::Eigen::VectorXd&)> dyf_;
	std::function<::Eigen::MatrixXd(const ::Eigen::VectorXd&, const ::Eigen::VectorXd&, const ::Eigen::VectorXd&)> pduuf_;
	std::function<::Eigen::MatrixXd(const ::Eigen::VectorXd&, const ::Eigen::VectorXd&, const ::Eigen::VectorXd&)> pduyf_;
	std::function<::Eigen::MatrixXd(const ::Eigen::VectorXd&, const ::Eigen::VectorXd&, const ::Eigen::VectorXd&)> pdyyf_;

	// boundary conditions
	std::function<::Eigen::VectorXd(const ::Eigen::VectorXd&, const ::Eigen::VectorXd&)> r_;
	std::function<::Eigen::MatrixXd(const ::Eigen::VectorXd&, const ::Eigen::VectorXd&)> dar_;
	std::function<::Eigen::MatrixXd(const ::Eigen::VectorXd&, const ::Eigen::VectorXd&)> dbr_;


	// memberfunctions
	// x=(y,u)

	// functions for composite step solver
	double cs_f(const Eigen::VectorXd& x);
	Eigen::VectorXd cs_f_derivative(const Eigen::VectorXd& x);
	Eigen::VectorXd cs_f_derivative_test(const Eigen::VectorXd& x);
	Eigen::MatrixXd cs_f_secDerivative(const Eigen::VectorXd& x);
	Eigen::VectorXd cs_c(const Eigen::VectorXd& x);
	Eigen::MatrixXd cs_c_derivative(const Eigen::VectorXd& x);
	Eigen::MatrixXd cs_c_secDerivative(const Eigen::VectorXd& x, const Eigen::VectorXd& p);
	Eigen::MatrixXd cs_gramian(const Eigen::VectorXd& x);
	Eigen::MatrixXd cs_M(const Eigen:: VectorXd& x);
	// auxiliary functions

	// get y_ij, both indices starting at 0
	Eigen::VectorXd get_y_ij(int i, int j, const Eigen::VectorXd& x);
	// get u_ij, both indices starting at 0
	Eigen::VectorXd get_u_ij(int i, int j, const Eigen::VectorXd& x);

	Eigen::VectorXd cs_f_y(const Eigen::VectorXd& x);
	Eigen::VectorXd cs_f_u(const Eigen::VectorXd& x);
	Eigen::MatrixXd cs_J_uu(const Eigen::VectorXd& x);
	Eigen::MatrixXd cs_J_yy(const Eigen::VectorXd& x);
	Eigen::MatrixXd cs_J_uy(const Eigen::VectorXd& x);
	Eigen::MatrixXd cs_c_y(const Eigen::VectorXd& x);
	Eigen::MatrixXd cs_c_u(const Eigen::VectorXd& x);
	Eigen::MatrixXd cs_M_y(const Eigen::VectorXd& x);
	Eigen::MatrixXd cs_M_u(const Eigen::VectorXd& x);

};

