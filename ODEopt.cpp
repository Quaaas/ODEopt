#include "ODEopt.hh"

//Constructor
ODEopt::ODEopt(
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
		) : g_(g), dyg_(dyg), dug_(dug), dyyg_(dyyg), duug_(duug), dyug_(dyug), f_(f), duf_(duf), dyf_(dyf),
			pduuf_(pduuf), pduyf_(pduyf), pdyyf_(pdyyf), r_(r), dar_(dar), dbr_(dbr), c_(c), grid_(grid), dim_y_(dim_y),
			dim_u_(dim_u), dim_r_(dim_r)
	{
		N_grid_ = grid.size();
		N_col_ = c.size();
	}

Eigen::VectorXd ODEopt::get_y_ij(int i,int j,const Eigen::VectorXd& x)
{
	Eigen::VectorXd y(dim_y_);
	y = x.segment(i*N_col_*dim_y_ + j*dim_y_,dim_y_);

	return y;
}

Eigen::VectorXd ODEopt::get_u_ij(int i,int j,const Eigen::VectorXd& x)
{
	Eigen::VectorXd u(dim_u_);
	u = x.segment((N_grid_-1)*dim_y_*N_col_ + dim_y_ + i*N_col_*dim_u_ + j*dim_u_,dim_u_);

	return u;
}

double ODEopt::cs_f(const Eigen::VectorXd& x)
{

	double result = 0;
	double temp_res = 0;
	double h = 0;

	// quadrature formula
	Eigen::VectorXd omega(3);
	Eigen::VectorXd b(3);
	omega << 5.0/18, 4.0/9, 5.0/18;
	b << 1.0/2.0 - sqrt(15)/10,  1.0/2.0,  1.0/2.0 + sqrt(15)/10;

	std::vector<Eigen::VectorXd> Eval;
	std::vector<Eigen::VectorXd> X;
	std::vector<Eigen::VectorXd> U;
	std::vector<Eigen::VectorXd> X_b;
	std::vector<Eigen::VectorXd> U_b;

	for(int i = 0; i < N_col_;i++)
	{
		X.push_back(Eigen::VectorXd::Zero(dim_y_));
		X_b.push_back(Eigen::VectorXd::Zero(dim_y_));
		U.push_back(Eigen::VectorXd::Zero(dim_u_));
		U_b.push_back(Eigen::VectorXd::Zero(dim_u_));
		Eval.push_back(Polynomial::evalOperator(b(i),c_));
	}

	//Iteration through grid
	for(int i = 0;i<N_grid_-1;i++)
	{
		h = grid_[i+1] - grid_[i];
		for(int j = 0; j < N_col_;j++)
		{
			X[j] = get_y_ij(i,j,x);
			U[j] = get_u_ij(i,j,x);
		}

		// calculate points at b
		for(int k = 0; k < N_col_;k++)
		{
			X_b[k] = Eigen::VectorXd::Zero(dim_y_);
			for(int s = 0; s < N_col_;s++)
			{
				X_b[k] += Eval[k](s)*X[s];
			}
		}

		for(int k = 0; k < N_col_;k++)
		{
			U_b[k] = Eigen::VectorXd::Zero(dim_u_);
			for(int s = 0; s < N_col_;s++)
			{
				U_b[k] += Eval[k](s)*U[s];
			}
		}

		// quadrature formula
		for(int k = 0; k < b.size();k++)
		{
			temp_res += omega(k)*g_(X_b[k],U_b[k]);
		}

		result += temp_res*h;
		temp_res = 0;
	}
	return result;
}

Eigen::VectorXd ODEopt::cs_f_y(const Eigen::VectorXd& x)
{
	Eigen::VectorXd result = Eigen::VectorXd::Zero(dim_y_*N_col_*(N_grid_-1) + dim_y_);
	std::vector<Eigen::VectorXd> g;

	for(int k = 0; k < N_col_; k++)
	{
		g.push_back(Eigen::VectorXd::Zero(dim_y_));
	}

	std::vector<Eigen::VectorXd> Eval;
	std::vector<Eigen::VectorXd> X;
	std::vector<Eigen::VectorXd> U;
	std::vector<Eigen::VectorXd> X_b;
	std::vector<Eigen::VectorXd> U_b;

	Eigen::VectorXd b(3);
	b << 1.0/2.0 - sqrt(15)/10,  1.0/2.0,  1.0/2.0 + sqrt(15)/10;


	for(int i = 0; i < N_col_;i++)
	{
		X.push_back(Eigen::VectorXd::Zero(dim_y_));
		X_b.push_back(Eigen::VectorXd::Zero(dim_y_));
		U.push_back(Eigen::VectorXd::Zero(dim_u_));
		U_b.push_back(Eigen::VectorXd::Zero(dim_u_));
		Eval.push_back(Polynomial::evalOperator(b(i),c_));
	}


	for(int i = 0;i<N_grid_-1;i++)
	{
		for(int j = 0; j < N_col_;j++)
		{
			X[j] = get_y_ij(i,j,x);
			U[j] = get_u_ij(i,j,x);
		}

		for(int k = 0; k < N_col_;k++)
		{
			X_b[k] = Eigen::VectorXd::Zero(dim_y_);
			for(int s = 0; s < N_col_;s++)
			{
				X_b[k] += Eval[k](s)*X[s];
			}
		}

		for(int k = 0; k < N_col_;k++)
		{
			U_b[k] = Eigen::VectorXd::Zero(dim_u_);
			for(int s = 0; s < N_col_;s++)
			{
				U_b[k] += Eval[k](s)*U[s];
			}
		}

		for(int k =0;k<N_col_;k++)
		{
			g[k] = dyg_(X_b[k],U_b[k]);
		}

		result.segment(i*N_col_*dim_y_,dim_y_*N_col_) = Polynomial::intLocalOperator_lin(grid_[i+1]-grid_[i],g,b);
	}

	return result;
}

Eigen::VectorXd ODEopt::cs_f_u(const Eigen::VectorXd& x)
{
	Eigen::VectorXd result = Eigen::VectorXd::Zero(dim_u_*N_col_*(N_grid_-1));
	std::vector<Eigen::VectorXd> g;

	for(int k = 0; k < N_col_; k++)
	{
		g.push_back(Eigen::VectorXd::Zero(dim_u_));
	}

	std::vector<Eigen::VectorXd> Eval;
	std::vector<Eigen::VectorXd> X;
	std::vector<Eigen::VectorXd> U;
	std::vector<Eigen::VectorXd> X_b;
	std::vector<Eigen::VectorXd> U_b;

	Eigen::VectorXd b(3);
	b << 1.0/2.0 - sqrt(15)/10,  1.0/2.0,  1.0/2.0 + sqrt(15)/10;


	for(int i = 0; i < N_col_;i++)
	{
		X.push_back(Eigen::VectorXd::Zero(dim_y_));
		X_b.push_back(Eigen::VectorXd::Zero(dim_y_));
		U.push_back(Eigen::VectorXd::Zero(dim_u_));
		U_b.push_back(Eigen::VectorXd::Zero(dim_u_));
		Eval.push_back(Polynomial::evalOperator(b(i),c_));
	}


	for(int i = 0;i<N_grid_-1;i++)
	{
		for(int j = 0; j < N_col_;j++)
		{
			X[j] = get_y_ij(i,j,x);
			U[j] = get_u_ij(i,j,x);
		}

		for(int k = 0; k < N_col_;k++)
		{
			X_b[k] = Eigen::VectorXd::Zero(dim_y_);
			for(int s = 0; s < N_col_;s++)
			{
				X_b[k] += Eval[k](s)*X[s];
			}
		}

		for(int k = 0; k < N_col_;k++)
		{
			U_b[k] = Eigen::VectorXd::Zero(dim_u_);
			for(int s = 0; s < N_col_;s++)
			{
				U_b[k] += Eval[k](s)*U[s];
			}
		}


		//G, tau ausrechnen
		for(int k =0;k<N_col_;k++)
		{
			g[k] = dug_(X_b[k],U_b[k]);
			//std::cout << U_b[k] << std::endl;
		}

		result.segment(i*N_col_*dim_u_,dim_u_*N_col_) = Polynomial::intLocalOperator_lin(grid_[i+1]-grid_[i],g,b);
	}



	return result;
}

Eigen::VectorXd ODEopt::cs_f_derivative(const Eigen::VectorXd& x)
{
	Eigen::VectorXd omega(3);
	Eigen::VectorXd b(3);
	omega << 5.0/18, 4.0/9, 5.0/18;
	b << 1.0/2.0 - sqrt(15)/10,  1.0/2.0,  1.0/2.0 + sqrt(15)/10;

	Eigen::VectorXd result(N_col_*dim_y_*(N_grid_-1) + dim_y_ + N_col_*dim_u_*(N_grid_-1));


	result << cs_f_y(x), cs_f_u(x);


	return result;
}

Eigen::MatrixXd ODEopt::cs_J_yy(const Eigen::VectorXd& x)
{
	Eigen::MatrixXd result = Eigen::MatrixXd::Zero(dim_y_*N_col_*(N_grid_-1) + dim_y_ ,dim_y_*N_col_*(N_grid_-1) + dim_y_);
	std::vector<Eigen::MatrixXd> G;

	for(int k =0;k<N_col_;k++)
	{
		G.push_back(Eigen::MatrixXd::Zero(dim_y_,dim_y_));
	}

	std::vector<Eigen::VectorXd> Eval;
	std::vector<Eigen::VectorXd> X;
	std::vector<Eigen::VectorXd> U;
	std::vector<Eigen::VectorXd> X_b;
	std::vector<Eigen::VectorXd> U_b;


	Eigen::VectorXd b(3);
	b << 1.0/2.0 - sqrt(15)/10,  1.0/2.0,  1.0/2.0 + sqrt(15)/10;

	for(int i = 0; i < N_col_;i++)
	{
		X.push_back(Eigen::VectorXd::Zero(dim_y_));
		X_b.push_back(Eigen::VectorXd::Zero(dim_y_));
		U.push_back(Eigen::VectorXd::Zero(dim_u_));
		U_b.push_back(Eigen::VectorXd::Zero(dim_u_));
		Eval.push_back(Polynomial::evalOperator(b(i),c_));
	}

	for(int i = 0;i<N_grid_-1;i++)
	{
		for(int j = 0; j < N_col_;j++)
		{
			X[j] = get_y_ij(i,j,x);
			U[j] = get_u_ij(i,j,x);
		}

		//Prototyp für richtiges
		for(int k = 0; k < N_col_;k++)
		{
			X_b[k] = Eigen::VectorXd::Zero(dim_y_);
			for(int s = 0; s < N_col_;s++)
			{
				X_b[k] += Eval[k](s)*X[s];
			}
		}

		//Prototyp für richtiges
		for(int k = 0; k < N_col_;k++)
		{
			U_b[k] = Eigen::VectorXd::Zero(dim_u_);
			for(int s = 0; s < N_col_;s++)
			{
				U_b[k] += Eval[k](s)*U[s];
			}
		}


		//G, tau ausrechnen
		for(int k =0;k<N_col_;k++)
		{
			G[k] = dyyg_(X_b[k],U_b[k]);
		}

		result.block(i*N_col_*dim_y_,i*N_col_*dim_y_,dim_y_*N_col_,dim_y_*N_col_) = Polynomial::intLocalOperator(grid_[i+1]-grid_[i],G,b);
	}
	return result;
}

Eigen::MatrixXd ODEopt::cs_J_uu(const Eigen::VectorXd& x)
{
	Eigen::MatrixXd result = Eigen::MatrixXd::Zero(dim_u_*N_col_*(N_grid_-1),dim_u_*N_col_*(N_grid_-1));
	std::vector<Eigen::MatrixXd> G;

	for(int k =0;k<N_col_;k++)
	{
		G.push_back(Eigen::MatrixXd::Zero(dim_u_,dim_u_));
	}

	std::vector<Eigen::VectorXd> Eval;
	std::vector<Eigen::VectorXd> X;
	std::vector<Eigen::VectorXd> U;
	std::vector<Eigen::VectorXd> X_b;
	std::vector<Eigen::VectorXd> U_b;


	Eigen::VectorXd b(3);
	b << 1.0/2.0 - sqrt(15)/10,  1.0/2.0,  1.0/2.0 + sqrt(15)/10;

	for(int i = 0; i < N_col_;i++)
	{
		X.push_back(Eigen::VectorXd::Zero(dim_y_));
		X_b.push_back(Eigen::VectorXd::Zero(dim_y_));
		U.push_back(Eigen::VectorXd::Zero(dim_u_));
		U_b.push_back(Eigen::VectorXd::Zero(dim_u_));
		Eval.push_back(Polynomial::evalOperator(b(i),c_));
	}

	for(int i = 0;i<N_grid_-1;i++)
	{
		for(int j = 0; j < N_col_;j++)
		{
			X[j] = get_y_ij(i,j,x);
			U[j] = get_u_ij(i,j,x);
		}

		//Prototyp für richtiges
		for(int k = 0; k < N_col_;k++)
		{
			X_b[k] = Eigen::VectorXd::Zero(dim_y_);
			for(int s = 0; s < N_col_;s++)
			{
				X_b[k] += Eval[k](s)*X[s];
			}
		}

		//Prototyp für richtiges
		for(int k = 0; k < N_col_;k++)
		{
			U_b[k] = Eigen::VectorXd::Zero(dim_u_);
			for(int s = 0; s < N_col_;s++)
			{
				U_b[k] += Eval[k](s)*U[s];
			}
		}


		//G, tau ausrechnen
		for(int k =0;k<N_col_;k++)
		{
			G[k] = duug_(X_b[k],U_b[k]);
		}

		result.block(i*N_col_*dim_u_,i*N_col_*dim_u_,dim_u_*N_col_,dim_u_*N_col_) = Polynomial::intLocalOperator(grid_[i+1]-grid_[i],G,b);

	}

	return result;
}

Eigen::MatrixXd ODEopt::cs_J_uy(const Eigen::VectorXd& x)
{
	Eigen::MatrixXd result = Eigen::MatrixXd::Zero(dim_y_*N_col_*(N_grid_-1) + dim_y_,dim_u_*N_col_*(N_grid_-1));
	std::vector<Eigen::MatrixXd> G;

	for(int k =0;k<N_col_;k++)
	{
		G.push_back(Eigen::MatrixXd::Zero(dim_y_,dim_u_));
	}

	std::vector<Eigen::VectorXd> Eval;
	std::vector<Eigen::VectorXd> X;
	std::vector<Eigen::VectorXd> U;
	std::vector<Eigen::VectorXd> X_b;
	std::vector<Eigen::VectorXd> U_b;


	Eigen::VectorXd b(3);
	b << 1.0/2.0 - sqrt(15)/10,  1.0/2.0,  1.0/2.0 + sqrt(15)/10;

	for(int i = 0; i < N_col_;i++)
	{
		X.push_back(Eigen::VectorXd::Zero(dim_y_));
		X_b.push_back(Eigen::VectorXd::Zero(dim_y_));
		U.push_back(Eigen::VectorXd::Zero(dim_u_));
		U_b.push_back(Eigen::VectorXd::Zero(dim_u_));
		Eval.push_back(Polynomial::evalOperator(b(i),c_));
	}

	for(int i = 0;i<N_grid_-1;i++)
	{
		for(int j = 0; j < N_col_;j++)
		{
			X[j] = get_y_ij(i,j,x);
			U[j] = get_u_ij(i,j,x);
		}

		//Prototyp für richtiges
		for(int k = 0; k < N_col_;k++)
		{
			X_b[k] = Eigen::VectorXd::Zero(dim_y_);
			for(int s = 0; s < N_col_;s++)
			{
				X_b[k] += Eval[k](s)*X[s];
			}
		}

		//Prototyp für richtiges
		for(int k = 0; k < N_col_;k++)
		{
			U_b[k] = Eigen::VectorXd::Zero(dim_u_);
			for(int s = 0; s < N_col_;s++)
			{
				U_b[k] += Eval[k](s)*U[s];
			}
		}


		//G, tau ausrechnen
		for(int k =0;k<N_col_;k++)
		{
			G[k] = dyug_(X_b[k],U_b[k]);
		}

		result.block(i*N_col_*dim_y_,i*N_col_*dim_u_,dim_y_*N_col_,dim_u_*N_col_) = Polynomial::intLocalOperator(grid_[i+1]-grid_[i],G,b);

	}

	return result;
}

Eigen::MatrixXd ODEopt::cs_f_secDerivative(const Eigen::VectorXd& x)
{
	Eigen::MatrixXd J_yy = ODEopt::cs_J_yy(x);
	Eigen::MatrixXd J_uy = ODEopt::cs_J_uy(x);
	Eigen::MatrixXd J_uu = ODEopt::cs_J_uu(x);

	Eigen::MatrixXd result(J_yy.rows()+J_uy.cols(), J_yy.rows() + J_uy.cols());
	result << J_yy, J_uy, J_uy.transpose(), J_uu;

	return result;

	//return Eigen::MatrixXd::Zero(2,2);
}

Eigen::VectorXd ODEopt::cs_c(const Eigen::VectorXd& x)
{
	Eigen::VectorXd y = x.head((N_grid_-1)*N_col_*dim_y_ + dim_y_);
	Eigen::VectorXd result((N_grid_-1)*N_col_*dim_y_  + dim_r_);

	result = Eigen::VectorXd::Zero((N_grid_-1)*N_col_*dim_y_ + dim_r_);
	Eigen::VectorXd y_der = Polynomial::diffOperator(grid_,c_,dim_y_)*y;

	Eigen::VectorXd f = Eigen::VectorXd::Zero((N_grid_-1)*N_col_*dim_y_);


	std::vector<Eigen::VectorXd> f_u_y;
	Eigen::VectorXd temp(dim_y_);
	temp = Eigen::VectorXd::Zero(dim_y_);
	for(int i=0; i < N_grid_ -1;i++)
	{
		f_u_y.push_back(temp);
		for(int j=1;j<N_col_;j++)
		{
			f_u_y.push_back(f_(get_y_ij(i,j,x),get_u_ij(i,j,x)));
		}
	}

	f_u_y.push_back(temp);

	for(int i =0;i<f_u_y.size()-1;i++)
	{
		f.segment(i*dim_y_,dim_y_) = f_u_y[i];
	}

	//std::cout <<  r_(get_y_ij(0,0,x),get_y_ij(N_grid_-1,0,x))<< std::endl;


	result << f-y_der.head(y_der.size()-dim_y_), r_(get_y_ij(0,0,x),get_y_ij(N_grid_-1,0,x));
	return result;
}

Eigen::MatrixXd ODEopt::cs_c_y(const Eigen::VectorXd& x)
{
	Eigen::MatrixXd result(dim_y_*(N_grid_-1)*N_col_ + dim_r_, dim_y_*(N_grid_-1)*N_col_ + dim_y_);
	result = Eigen::MatrixXd::Zero(dim_y_*(N_grid_-1)*N_col_ + dim_r_, dim_y_*(N_grid_-1)*N_col_ + dim_y_);


	std::vector<Eigen::MatrixXd> F;

	for(int k =0;k<N_col_-1;k++)
	{
		F.push_back(Eigen::MatrixXd::Zero(dim_y_,dim_y_));
	}



	for(int i = 0; i < N_grid_ -1;i++)
	{

		for(int k =0;k<N_col_-1;k++)
		{
			F[k] = dyf_(get_y_ij(i,k+1,x),get_u_ij(i,k+1,x));
		}

		result.block(i*dim_y_*N_col_,i*dim_y_*(N_col_),dim_y_*N_col_,dim_y_*(N_col_ +1)) =
				Polynomial::collocationLocal(F,c_,dim_y_);
	}

	result.bottomLeftCorner(dim_r_,dim_y_) = dar_(get_y_ij(0,0,x), get_y_ij(N_col_,0,x));
	result.bottomRightCorner(dim_r_,dim_y_) = dbr_(get_y_ij(0,0,x), get_y_ij(N_col_,0,x));
	Eigen::MatrixXd D = Polynomial::diffOperator(grid_,c_,dim_y_);
	Eigen::MatrixXd D2(dim_y_*(N_grid_-1)*N_col_ + dim_r_, dim_y_*(N_grid_-1)*N_col_ + dim_y_);
	Eigen::MatrixXd z = Eigen::MatrixXd::Zero(dim_r_ - dim_y_ ,dim_y_*(N_grid_-1)*N_col_ + dim_y_);
	D2 << D,z;
	result = result - D2;

	return result;
}

Eigen::MatrixXd ODEopt::cs_c_u(const Eigen::VectorXd& x)
{
	Eigen::MatrixXd result(dim_y_*(N_grid_-1)*N_col_ + dim_r_, dim_u_*(N_grid_-1)*N_col_);
	result = Eigen::MatrixXd::Zero(dim_y_*(N_grid_-1)*N_col_ + dim_r_, dim_u_*(N_grid_-1)*N_col_);

	for(int i = 0; i < N_grid_-1;i++)
	{
		for(int j = 1;j<N_col_;j++)
		{
			result.block(i*dim_y_*N_col_ + j*dim_y_,i*dim_u_*N_col_ + j*dim_u_,dim_y_,dim_u_)
					= duf_(get_y_ij(i,j,x),get_u_ij(i,j,x));
		}
	}

	return result;
}

Eigen::MatrixXd ODEopt::cs_c_derivative(const Eigen::VectorXd& x)
{
	Eigen::MatrixXd result(dim_y_*(N_grid_-1)*N_col_ + dim_r_, dim_y_*(N_grid_-1)*N_col_ + dim_y_ +  dim_u_*(N_grid_-1)*N_col_);
	result << cs_c_y(x), cs_c_u(x);
	return result;
}

Eigen::MatrixXd ODEopt::cs_M_y(const Eigen ::VectorXd& x)
{
	Eigen::MatrixXd result = Eigen::MatrixXd::Zero(dim_y_*N_col_*(N_grid_-1) + dim_y_ ,dim_y_*N_col_*(N_grid_-1) + dim_y_);
	std::vector<Eigen::MatrixXd> G;

	for(int k =0;k<N_col_;k++)
	{
		G.push_back(Eigen::MatrixXd::Zero(dim_y_,dim_y_));
	}

	Eigen::VectorXd b(3);
	b << 1.0/2.0 - sqrt(15)/10,  1.0/2.0,  1.0/2.0 + sqrt(15)/10;

	for(int i = 0;i<N_grid_-1;i++)
	{
		//G, tau ausrechnen
		for(int k =0;k<N_col_;k++)
		{
			G[k] = Eigen::MatrixXd::Identity(dim_y_,dim_y_);
		}

		result.block(i*N_col_*dim_y_,i*N_col_*dim_y_,dim_y_*N_col_,dim_y_*N_col_) = Polynomial::intLocalOperator(grid_[i+1]-grid_[i],G,b);
	}
	return result;
}

Eigen::MatrixXd ODEopt::cs_M_u(const Eigen::VectorXd& x)
{
	Eigen::MatrixXd result = Eigen::MatrixXd::Zero(dim_u_*N_col_*(N_grid_-1),dim_u_*N_col_*(N_grid_-1));
	std::vector<Eigen::MatrixXd> G;

	for(int k =0;k<N_col_;k++)
	{
		G.push_back(Eigen::MatrixXd::Zero(dim_u_,dim_u_));
	}

	Eigen::VectorXd b(3);
	b << 1.0/2.0 - sqrt(15)/10,  1.0/2.0,  1.0/2.0 + sqrt(15)/10;

	for(int i = 0;i<N_grid_-1;i++)
	{
		//G, tau ausrechnen
		for(int k =0;k<N_col_;k++)
		{
			G[k] = Eigen::MatrixXd::Identity(dim_u_,dim_u_);
		}

		result.block(i*N_col_*dim_u_,i*N_col_*dim_u_,dim_u_*N_col_,dim_u_*N_col_) = Polynomial::intLocalOperator(grid_[i+1]-grid_[i],G,b);

	}

	return result;
}

Eigen::MatrixXd ODEopt::cs_M(const Eigen:: VectorXd& x)
{
	Eigen::MatrixXd M_y= ODEopt::cs_M_y(x);
	Eigen::MatrixXd M_u = ODEopt::cs_M_u(x);

	Eigen::MatrixXd result = Eigen::MatrixXd::Zero(M_u.rows()+M_y.rows(), M_u.cols() + M_y.cols());

	Eigen::MatrixXd Z = Eigen::MatrixXd::Zero(M_y.rows(),M_u.cols());
	result << M_y, Z, Z.transpose(), M_u;

	return result;
}
