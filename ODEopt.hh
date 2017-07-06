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
			int dim_r,
			Eigen::VectorXd y,
			Eigen::VectorXd u,
			Eigen::VectorXd p
	);


	//Vektor der Kollokationspunkte
	Eigen::VectorXd c_;

	//Gitter
	std::vector<double> grid_;

	//Anzahl der Gitterpunkte
	int N_grid_;

	//Dimension der Differentialgleichung
	int dim_y_;

	// Dimension der Kontrollfunktion
	int dim_u_;

	// Anzahl der Kollokationspunkte
	int N_col_;

	//Dimension der Randbedingungen
	int dim_r_;

	//Zustandsvector
	Eigen::VectorXd y_;

	//Kontrollfunktion
	Eigen::VectorXd u_;

	//Lagrange Multiplikator
	Eigen::VectorXd p_;

	// functions for OC problem with ODEs

	//Kostenfunktion g
	std::function<double(const ::Eigen::VectorXd&, const ::Eigen::VectorXd&)> g_;
	std::function<::Eigen::VectorXd(const ::Eigen::VectorXd&, const ::Eigen::VectorXd&)> dyg_;
	std::function<::Eigen::VectorXd(const ::Eigen::VectorXd&, const ::Eigen::VectorXd&)> dug_;
	std::function<::Eigen::MatrixXd(const ::Eigen::VectorXd&, const ::Eigen::VectorXd&)> dyyg_;
	std::function<::Eigen::MatrixXd(const ::Eigen::VectorXd&, const ::Eigen::VectorXd&)> duug_;
	std::function<::Eigen::MatrixXd(const ::Eigen::VectorXd&, const ::Eigen::VectorXd&)> dyug_;

	//Rechte Seite der Differentialgleichung
	std::function<::Eigen::VectorXd(const ::Eigen::VectorXd&, const ::Eigen::VectorXd&)> f_;
	std::function<::Eigen::MatrixXd(const ::Eigen::VectorXd&, const ::Eigen::VectorXd&)> duf_;
	std::function<::Eigen::MatrixXd(const ::Eigen::VectorXd&, const ::Eigen::VectorXd&)> dyf_;
	std::function<::Eigen::MatrixXd(const ::Eigen::VectorXd&, const ::Eigen::VectorXd&, const ::Eigen::VectorXd&)> pduuf_;
	std::function<::Eigen::MatrixXd(const ::Eigen::VectorXd&, const ::Eigen::VectorXd&, const ::Eigen::VectorXd&)> pduyf_;
	std::function<::Eigen::MatrixXd(const ::Eigen::VectorXd&, const ::Eigen::VectorXd&, const ::Eigen::VectorXd&)> pdyyf_;

	//Randbedingung
	std::function<::Eigen::VectorXd(const ::Eigen::VectorXd&, const ::Eigen::VectorXd&)> r_;
	std::function<::Eigen::MatrixXd(const ::Eigen::VectorXd&, const ::Eigen::VectorXd&)> dar_;
	std::function<::Eigen::MatrixXd(const ::Eigen::VectorXd&, const ::Eigen::VectorXd&)> dbr_;

	//



	//Memberfunktionen
	// x=(y,u)
	double cs_f(const Eigen::VectorXd& x);
	Eigen::VectorXd cs_f_derivative(const Eigen::VectorXd& x);
	Eigen::VectorXd cs_f_derivative_test(const Eigen::VectorXd& x);
	Eigen::MatrixXd cs_f_secDerivative(const Eigen::VectorXd& x);
	Eigen::VectorXd cs_c(const Eigen::VectorXd& x);
	Eigen::MatrixXd cs_c_derivative(const Eigen::VectorXd& x);
	Eigen::MatrixXd cs_c_secDerivative(const Eigen::VectorXd& x, const Eigen::VectorXd& p);
	Eigen::MatrixXd cs_gramian(const Eigen::VectorXd& x);


	Eigen::VectorXd get_y_i(int i, const Eigen::VectorXd& x);
	Eigen::VectorXd get_y_ij(int i, int j, const Eigen::VectorXd& x);
	Eigen::VectorXd get_u_ij(int i, int j, const Eigen::VectorXd& x);
	Eigen::VectorXd cs_f_y(const Eigen::VectorXd& x);
	Eigen::VectorXd cs_f_u(const Eigen::VectorXd& x);
	Eigen::MatrixXd cs_J_uu(const Eigen::VectorXd& x);
	Eigen::MatrixXd cs_J_yy(const Eigen::VectorXd& x);
	Eigen::MatrixXd cs_J_yy_2(const Eigen::VectorXd& x);
	Eigen::MatrixXd cs_J_uy(const Eigen::VectorXd& x);
	Eigen::MatrixXd cs_c_y(const Eigen::VectorXd& x);
	Eigen::MatrixXd cs_c_u(const Eigen::VectorXd& x);
};

