/*
 * util.hh
 *
 *  Created on: May 31, 2017
 *      Author: sebastian
 */
#include <eigen3/Eigen/Dense>
#include <vector>
#include <iostream>

namespace Polynomial{

//Auswertung eines Lagrangepolynoms L_i(t), das durch den Vektor c definiert ist
double laPol(double t, int i,const Eigen::VectorXd &c);

//Ableitung
double laPolDer(double t, int i,const Eigen::VectorXd &c);

//Erzeugt Element im Dualraum, um ein Polynom auszuwerten
Eigen::VectorXd evalOperator(double t, const Eigen::VectorXd &c);


Eigen::MatrixXd collocationLocal(const std::vector<Eigen::MatrixXd>& F, const Eigen::VectorXd &c, int dim_y);

//Erzeugt Differentialoperator
Eigen::MatrixXd diffLocalOperator(const Eigen::VectorXd &c, int dim_y);

Eigen::MatrixXd diffOperator(const std::vector<double> &grid, const Eigen::VectorXd &c, int dim_y);

//Hilfsfunktion für Integraloperatoren
Eigen::MatrixXd intW_ij(int i, int j, double tau, const std::vector<Eigen::MatrixXd>& G, const Eigen::VectorXd &c);

Eigen::MatrixXd intLocalOperator(double tau, const std::vector<Eigen::MatrixXd>& G, const Eigen::VectorXd &c);

Eigen::VectorXd intW_j(int j, double tau, const std::vector<Eigen::VectorXd>& g, const Eigen::VectorXd& c);

Eigen::VectorXd intLocalOperator_lin(double tau, const std::vector<Eigen::VectorXd>& g , const Eigen::VectorXd &c);
}
