#include <Spacy/Adapter/eigen.hh>
#include <Spacy/Adapter/Eigen/compositeStepFunctional.hh>
#include <Spacy/Adapter/Eigen/util.hh>
#include <Spacy/Spaces/realSpace.hh>
#include <Spacy/Util/Base/FunctionalBase.hh>
#include <Spacy/Util/cast.hh>
#include <Spacy/Util/Base/VectorBase.hh>
#include <Spacy/spacy.h>
#include <eigen3/Eigen/Dense>
#include <functional>
#include <iostream>
#include <Spacy/Algorithm/CompositeStep/affineCovariantSolver.hh>
#include <Spacy/zeroVectorCreator.hh>
#include "ODEopt.hh"
#include "util.hh"
#include <fstream>
#include <ctime>


using Vector = Eigen::VectorXd;
using Matrix = Eigen::MatrixXd;

std::function<double(Vector,Vector)> g = [] (const Vector& x, const Vector& u)
		{
			return u(0)*u(0);
		};

std::function<Vector(Vector,Vector)> dxg = [] (const Vector& x, const Vector& u)
		{
			Vector d(2);
			d(0) = 0;
			d(1) = 0;
			return d;
		};

std::function<Vector(Vector,Vector)> dug = [] (const Vector& x, const Vector& u)
		{
			Vector d(1);
			d(0) = 2*u(0);
			return d;
		};

std::function<Matrix(Vector,Vector)> dxxg = [] (const Vector& x, const Vector& u)
		{
			Matrix D(2,2);
			D(0,0) = 0;
			D(1,0) = 0;
			D(0,1) = 0;
			D(1,1) = 0;
			return D;
		};

std::function<Matrix(Vector,Vector)> duug = [] (const Vector& x, const Vector& u)
		{
			Matrix D(1,1);
			D(0,0) = 2;
			return D;
		};

std::function<Matrix(Vector,Vector)> dxug = [] (const Vector& x, const Vector& u)
		{
			Matrix D(2,1);
			D(0,0) = 0;
			D(1,0) = 0;
			return D;
		};

std::function<Vector(Vector,Vector)> f = [] (const Vector& x, const Vector& u)
		{
			Vector d(2);
			d(0) = x(1);
			d(1) = u(0);
			return d;
		};

std::function<Matrix(Vector,Vector)> duf = [] (const Vector& x, const Vector& u)
		{
			Matrix D(2,1);
			D(0,0) = 0;
			D(1,0) = 1;
			return D;
		};

std::function<Matrix(Vector,Vector)> dxf = [] (const Vector& x, const Vector& u)
		{
			Matrix D(2,2);
			D(0,0) = 0;
			D(1,0) = 0;
			D(0,1) = 1;
			D(1,1) = 0;
			return D;
		};

std::function<Matrix(Vector,Vector,Vector)> pduuf = [] (const Vector& p, const Vector& x, const Vector& u)
		{
			Matrix D(2,2);
			D(0,0) = 0;
			D(1,0) = 0;
			D(0,1) = 0;
			D(1,1) = 0;
			return D;
		};

std::function<Matrix(Vector,Vector,Vector)> pduxf = [] (const Vector& p, const Vector& x, const Vector& u)
		{
			Matrix D(2,2);
			D(0,0) = 0;
			D(1,0) = 0;
			D(0,1) = 0;
			D(1,1) = 0;
			return D;
		};

std::function<Matrix(Vector,Vector,Vector)> pdxxf = [] (const Vector& p, const Vector& x, const Vector& u)
		{
			Matrix D(2,2);
			D(0,0) = 0;
			D(1,0) = 0;
			D(0,1) = 0;
			D(1,1) = 0;
			return D;
		};

std::function<Vector(Vector,Vector)> r = [] (const Vector& x, const Vector& y)
		{
			Vector d(4);
			d(0) = x(0)-1;
			d(1) = x(1)-1;
			d(2) = y(0);
			d(3) = y(1);
			return d;
		};

std::function<Matrix(Vector,Vector)> dxr = [] (const Vector& x, const Vector& y)
		{
			Matrix D = Matrix::Zero(4,2);
			D(0,0)=1;
			D(1,1)=1;
			return D;
		};

std::function<Matrix(Vector,Vector)> dyr = [] (const Vector& x, const Vector& y)
		{
			Matrix D = Matrix::Zero(4,2);
			D(2,0)=1;
			D(3,1)=1;
			return D;
		};


int main() {
	//Kollokationspunkte
	Vector c(3);
	c << 0,(1.0/2-sqrt(3.0)/6),(1.0/2+sqrt(3.0)/6);

	//Gitter
	int N = 10;
	std::vector<double> grid(N,0);
	for(int i = 0;i<N;i++){
		grid[i] = (double) i*1/(N-1);
	}

	//Dimensionen
	int dim_y = 2;
	int dim_u = 1;
	int dim_r = 4;

	//Vector y0(N*dim_y*c.size());

	//Startwert
	Vector y0 = Vector::Zero((N-1)*dim_y*c.size() + dim_y);
	Vector u0 = Vector::Zero((N-1)*dim_u*c.size());
	Vector p0 = Vector::Zero((N-1)*dim_y*c.size() + dim_r);

	for(int i = 0; i<y0.size();i++)
	{
		y0(i) = 1;
	}
	for(int i = 0; i<u0.size();i++)
	{
		u0(i) = 1;
	}

	for(int i = 0; i<p0.size();i++)
	{
		p0(i) = 1;
	}

	Vector x(y0.size() + u0.size() + p0.size());
	x << y0 , u0, p0;



	auto odeopt = ODEopt(g,dxg,dug,dxxg,duug,dxug,f,duf,dxf,pduuf,pduxf,pdxxf,
						 r,dxr,dyr,c,grid,dim_y,dim_u,dim_r,y0,u0,p0);


	//std::cout << odeopt.cs_J_yy(x) << std::endl;


	//Linear Test
	Eigen::MatrixXd MAT((N-1)*(dim_y*2+dim_u)*c.size() + dim_y + dim_r,(N-1)*(dim_y*2+dim_u)*c.size() + dim_y + dim_r);
	Eigen::VectorXd RHS(x.size());
	Eigen::MatrixXd J = odeopt.cs_f_secDerivative(x);
	Eigen::MatrixXd C = odeopt.cs_c_derivative(x);


	Eigen::VectorXd u_temp(y0.size() + u0.size());
	u_temp << Eigen::VectorXd::Zero(y0.size()), u0;

	//std::cout << odeopt.cs_f_derivative(x) << std::endl;

	//odeopt.cs_f_u(x);
	//std::cout << odeopt.cs_f_secDerivative(x)*u_temp;

//	MAT<< J, C.transpose(), C, Eigen::MatrixXd::Zero((N-1)*dim_y*c.size() + dim_r,(N-1)*dim_y*c.size() + dim_r);
//	RHS << odeopt.cs_f_derivative(x) - C.transpose()*p0, -odeopt.cs_c(x);
//
//	clock_t begin = clock();
//
//
//	Eigen::VectorXd res = MAT.householderQr().solve(RHS);
//
//	clock_t end = clock();
//
//	double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
//	//std::cout << elapsed_secs << std::endl;
//
//	std::ofstream myfile;
//	myfile.open ("res.txt");
//	myfile << x + res;
//	myfile.close();
//
//	std::ofstream myfile2;
//	myfile2.open ("rhs.txt");
//	myfile2 << RHS;
//	myfile2.close();
//
//	std::ofstream myfile3;
//	myfile2.open ("matrix.txt");
//	myfile2 << MAT;
//	myfile2.close();



	//Erster Test mit Spacy
	std::function<double(::Eigen::VectorXd)> value_f = [&](const ::Eigen::VectorXd& x)
	{
		return odeopt.cs_f(x);
	};

	std::function<::Eigen::VectorXd(::Eigen::VectorXd)> derivative_f = [&](const ::Eigen::VectorXd& x)
	{

		return odeopt.cs_f_derivative(x);
	};

	std::function<::Eigen::MatrixXd(::Eigen::VectorXd)> secDerivative_f = [&](const ::Eigen::VectorXd& x)
	{
		return odeopt.cs_f_secDerivative(x);
	};

	std::function<::Eigen::VectorXd(::Eigen::VectorXd)> value_c = [&](const ::Eigen::VectorXd& x)
	{
		return odeopt.cs_c(x);
	 };

	std::function<::Eigen::MatrixXd(::Eigen::VectorXd)> derivative_c = [&](const ::Eigen::VectorXd& x)   // c'(x)
	{
	    return odeopt.cs_c_derivative(x);
	};

	std::function<::Eigen::MatrixXd(::Eigen::VectorXd, ::Eigen::VectorXd)> secDerivative_c = [&](const ::Eigen::VectorXd& x, const ::Eigen::VectorXd& p)
	{
		Eigen::MatrixXd c_xx = Eigen::MatrixXd::Zero(J.rows(),J.rows());
	    return c_xx;
	};

	std::function<::Eigen::MatrixXd(::Eigen::VectorXd)> gramian = [&](const ::Eigen::VectorXd& x)
	{
		return odeopt.cs_f_secDerivative(x);
	};


//	std::cout << "value_f:  " << value_f(x) << std::endl;
//	std::cout << "derivative_f:  " << derivative_f(x).size() << std::endl;
//	std::cout << "secderivative_f:  " << secDerivative_f(x).rows() << "x" << secDerivative_f(x).cols() << std::endl;
//	std::cout << "value_c:  " << value_c(x).size()  << std::endl;
//	std::cout << "derivative_c:  " << derivative_c(x).rows() << "x" << derivative_c(x).cols() << std::endl;
//	std::cout << "secderivative_c:  " << secDerivative_c(x,x).rows() << "x" << secDerivative_c(x,x).cols() << std::endl;
//	std::cout << "gramian:  " << gramian(x).rows() << "x" << gramian(x).cols() << std::endl;
//	std::cout << "x:  " << x.size() << std::endl;
//




//	using namespace Spacy;
//
//	std::vector<std::shared_ptr< VectorSpace > > spaces(2);
//
//	spaces[0] = std::make_shared<Spacy::VectorSpace>(Spacy::Rn::makeHilbertSpace(83));
//	spaces[1] = std::make_shared<Spacy::VectorSpace>(Spacy::Rn::makeHilbertSpace(58));
//
//	auto domain = Spacy::ProductSpace::makeHilbertSpace(spaces);
//
//	Spacy::Rn::TangentialStepFunctional L_T(value_f,derivative_f,secDerivative_f,value_c,derivative_c,secDerivative_c,domain);
//	Spacy::Rn::NormalStepFunctional L_N(L_T,gramian);
//
//	auto x0=domain.creator()(&domain);
//	Spacy::Rn::copy(x,x0);
//
//
//	auto cs = Spacy::CompositeStep::AffineCovariantSolver( L_N , L_T , domain );
//	cs.setRelativeAccuracy(1e-6);
//	cs.setVerbosityLevel(1);
//	cs.setMaxSteps(10);
//    auto result = cs(x0);
//	L_T(x0);

}
