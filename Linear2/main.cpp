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
#include "../ODEopt.hh"
#include "../util.hh"
#include "../ODEoptVector.hh"
#include <fstream>
#include <ctime>
#include <string>

#define EIGEN_INITIALIZE_MATRICES_BY_ZERO


using Vector = Eigen::VectorXd;
using Matrix = Eigen::MatrixXd;

std::function<double(Vector,Vector)> g = [] (const Vector& x, const Vector& u)
		{
			return u(0)*u(0) + x(0)*x(0);
		};

std::function<Vector(Vector,Vector)> dxg = [] (const Vector& x, const Vector& u)
		{
			Vector d(1);
			d(0) = 2*x(0);
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
			Matrix D(1,1);
			D(0,0) = 2;
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
			Matrix D(1,1);
			D(0,0) = 0;
			return D;
		};

std::function<Vector(Vector,Vector)> f = [] (const Vector& x, const Vector& u)
		{
			Vector d(1);
			d(0) = x(0) + u(0);
			return d;
		};

std::function<Matrix(Vector,Vector)> duf = [] (const Vector& x, const Vector& u)
		{
			Matrix D(1,1);
			D(0,0) = 1;
			return D;
		};

std::function<Matrix(Vector,Vector)> dxf = [] (const Vector& x, const Vector& u)
		{
			Matrix D(1,1);
			D(0,0) = 1;
			return D;
		};

std::function<Matrix(Vector,Vector,Vector)> pduuf = [] (const Vector& p, const Vector& x, const Vector& u)
		{
			Matrix D(1,1);
			D(0,0) = 0;
			return D;
		};

std::function<Matrix(Vector,Vector,Vector)> pduxf = [] (const Vector& p, const Vector& x, const Vector& u)
		{
			Matrix D(1,1);
			D(0,0) = 0;
			return D;
		};

std::function<Matrix(Vector,Vector,Vector)> pdxxf = [] (const Vector& p, const Vector& x, const Vector& u)
		{
			Matrix D(1,1);
			D(0,0) = 0;
			return D;
		};

std::function<Vector(Vector,Vector)> r = [] (const Vector& x, const Vector& y)
		{
			Vector d(1);
			d(0) = x(0)-1;
			return d;
		};

std::function<Matrix(Vector,Vector)> dxr = [] (const Vector& x, const Vector& y)
		{
			Matrix D = Matrix::Zero(1,1);
			D(0,0)=1;
			return D;
		};

std::function<Matrix(Vector,Vector)> dyr = [] (const Vector& x, const Vector& y)
		{
			Matrix D = Matrix::Zero(1,1);
			D(0,0)=0;
			return D;
		};


int main() {
	//Kollokationspunkte
	Vector c(3);
	c << 0,(1.0/2-sqrt(3)/6),(1.0/2+sqrt(3)/6);

	int b = 5;
	//Gitter
	// int N = 20;
	// int N_1 = 10;
	//
	// std::string st = std::to_string(N) +"_" + std::to_string(N_1);
	//
	// std::vector<double> grid(N,0);
	// for(int i = 0;i<N_1;i++){
	// 	grid[i] = (double) b*i*0.1/(N_1-1);
	// }
	//
	// for(int i = 0;i<N-N_1+1;i++){
	// 	grid[N_1 + i - 1] = (double) (b*0.1 + b*i*0.9/(N-N_1));
	// }
	//
	// for(int i = 0; i < grid.size();i++)
	// 	{
	// 		std::cout << grid[i] << std::endl;
	// 	}


	//Gitter äquvidistant
	int N = 20;


	std::string st = std::to_string(N) + "_dense";



	std::vector<double> grid(N,0);
	for(int i = 0;i<N;i++){
		grid[i] = (double) b*i*1./(N-1);
	}

	//Dimensionen
	int dim_y = 1;
	int dim_u = 1;
	int dim_r = 1;

	//Vector y0(N*dim_y*c.size());

	//Startwert
	Vector y0 = Vector::Zero((N-1)*dim_y*c.size() + dim_y);
	Vector u0 = Vector::Zero((N-1)*dim_u*c.size());
	Vector p0 = Vector::Zero((N-1)*dim_y*c.size() + dim_r);

	for(int i = 0; i<y0.size();i++)
	{
		y0(i) = 0.5;
	}
	for(int i = 0; i<u0.size();i++)
	{
		u0(i) = 1;
	}

	for(int i = 0; i<p0.size();i++)
	{
		p0(i) = 0;
	}

	Vector x_init(y0.size() + u0.size() + p0.size());
	x_init << y0 , u0, p0;

	auto odeopt = ODEopt(g,dxg,dug,dxxg,duug,dxug,f,duf,dxf,pduuf,pduxf,pdxxf,
						 r,dxr,dyr,c,grid,dim_y,dim_u,dim_r);

        Vector test(y0.size() + u0.size());
        test << y0,u0;


	//Spacy
	std::function<double(::Eigen::VectorXd)> value_f = [&](const ::Eigen::VectorXd& x)
	{
		return odeopt.cs_f(x);
	};

	std::function<::Eigen::VectorXd(::Eigen::VectorXd)> derivative_f = [&](const ::Eigen::VectorXd& x)
	{
		return odeopt.cs_f_derivative(x);
	};

	std::function<Eigen::MatrixXd(::Eigen::VectorXd)> secDerivative_f = [&](const ::Eigen::VectorXd& x)
	{
		return Eigen::MatrixXd(odeopt.cs_f_secDerivative(x));
	};

	std::function<::Eigen::VectorXd(::Eigen::VectorXd)> value_c = [&](const ::Eigen::VectorXd& x)
	{
		return odeopt.cs_c(x);
	 };

	std::function<Eigen::MatrixXd(::Eigen::VectorXd)> derivative_c = [&](const ::Eigen::VectorXd& x)   // c'(x)
	{
	    return Eigen::MatrixXd(odeopt.cs_c_derivative(x));
	};

	std::function<Eigen::MatrixXd(::Eigen::VectorXd, ::Eigen::VectorXd)> secDerivative_c = [&](const ::Eigen::VectorXd& x, const ::Eigen::VectorXd& p)
	{
	    return Eigen::MatrixXd(odeopt.cs_c_secDerivative(x,p));
	};

	std::function<Eigen::MatrixXd(::Eigen::VectorXd)> gramian = [&](const ::Eigen::VectorXd& x)
	{
	    return Eigen::MatrixXd(odeopt.cs_M(x));
    };



	using namespace Spacy;

	std::vector<std::shared_ptr< VectorSpace > > spaces(2);

	spaces[0] = std::make_shared<Spacy::VectorSpace>(Spacy::Rn::makeHilbertSpace(y0.size() + u0.size()));
	spaces[1] = std::make_shared<Spacy::VectorSpace>(Spacy::Rn::makeHilbertSpace(p0.size()));

	auto domain = Spacy::ProductSpace::makeHilbertSpace(spaces);

	Spacy::Rn::TangentialStepFunctional L_T(value_f,derivative_f,secDerivative_f,value_c,derivative_c,secDerivative_c,domain);
	Spacy::Rn::NormalStepFunctional L_N(L_T,gramian);

	//auto x0=domain.creator()(&domain);
	auto x0 = zero(domain);
	Spacy::Rn::copy(x_init,x0);

	auto cs = Spacy::CompositeStep::AffineCovariantSolver( L_N , L_T , domain );
	cs.setRelativeAccuracy(1e-12);
	cs.set_eps(1e-12);
	cs.setVerbosityLevel(2);
	cs.setMaxSteps(100);
    auto result = cs(x0);
    Spacy::Rn::copy(result,x_init);
//

    // std::ofstream myfile;
    // myfile.open ("x_spacy.txt");
    // myfile << x_init;
    // myfile.close();
	//
	// std::ofstream myfile_grid;
	// myfile_grid.open ("x_spacy_grid.txt");
	// for(int i = 0; i < grid.size(); i ++)
	// {
	// 	myfile_grid << grid[i] << std::endl;
	// }
	// myfile_grid.close();
//
	auto odevector = ODEoptVector(x_init,dim_y,dim_u,dim_r,grid,c);

	std::ofstream myfile1;
	std::ofstream myfile1_grid;
	std::ofstream myfile1_u;

	myfile1_grid.open("stetig_grid_" + st + ".txt");
    myfile1.open ("stetig_" + st + ".txt");
	myfile1_u.open ("stetig_u_" + st + ".txt");
	for(int i =0;i<2000-1;i++)
	{
		myfile1 << odevector.eval_x((double) b*i*1./(2000-1)) << std::endl;
		myfile1_u <<  odevector.eval_u((double) b*i*1./(2000-1)) << std::endl;
		myfile1_grid << (double) b*i*1./(2000-1) << std::endl;
	}
    myfile1.close();
	myfile1_u.close();
	myfile1_grid.close();
}
