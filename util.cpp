/*
 * util.cpp
 *
 *  Created on: May 31, 2017
 *      Author: sebastian
 */

#include "util.hh"

namespace Polynomial{

double laPol(double t, int i, const Eigen::VectorXd &c)
{
	int n = c.size();
	double res = 1;

	for(int j=0;j<n;j++)
	{
		if(j != i) res*=(t-c(j))/(c(i)-c(j));
	}

	return res;
};

double laPolDer(double t, int i, const Eigen::VectorXd &c)
{
	int n = c.size();
	double res = 0;
	double mult = 1;


	for(int j=0;j<n;j++)
	{
		mult = 1;
		if(j!=i)
		{
			for(int k=0;k<n;k++)
			{
				if((k!=i)&&(k!=j)) mult*=(t-c(k))/(c(i)-c(k));
			}
			res+= mult/(c(i)-c(j));
		}
	}

	return res;
};

Eigen::VectorXd evalOperator(double t, const Eigen::VectorXd &c)
{
	int n = c.size();
	Eigen::VectorXd v(n);
	for(int i=0; i<n;i++)
	{
		v(i) = laPol(t,i,c);
	}

	return v;
}

Eigen::MatrixXd collocationLocal(const std::vector<Eigen::MatrixXd>& F, const Eigen::VectorXd &c, int dim_y)
{
	int N_col = c.size();
	Eigen::MatrixXd result(dim_y*N_col, dim_y*N_col + dim_y);
	result = Eigen::MatrixXd::Zero(dim_y*N_col, dim_y*N_col + dim_y);
	Eigen::VectorXd L = evalOperator(1, c);

	for(int i = 0; i< dim_y;i++)
	{
		for(int j = 0; j < N_col; j++)
		{
			result(i, i + j*dim_y) = L(j);
		}
	}

	Eigen::MatrixXd I = Eigen::MatrixXd::Identity(dim_y,dim_y);
	result.block(0,dim_y*N_col,dim_y,dim_y) = -I;

	for(int i = 1; i < N_col;i++)
	{
		result.block(i*dim_y,i*dim_y,dim_y,dim_y) = F[i-1];
	}

	return result;
}

Eigen::MatrixXd diffLocalOperator(const Eigen::VectorXd &c, int dim_y)
{
	int s = c.size();
	Eigen::VectorXd v(s);
	Eigen::MatrixXd result(s*dim_y,s*dim_y);
	result = Eigen::MatrixXd::Zero(s*dim_y,s*dim_y);
	int p = 0;


	for(int j =1; j < s;j++){
		for(int i=0; i<s;i++)
		{
			v(i) = laPolDer(c(j),i,c);
		}

		for(int k = 0; k < dim_y;k++)
		{
			for(int l = k; l < s*dim_y; l = l+dim_y)
			{
				result(j*dim_y+k, l) = v(p);
				p++;
			}
			p = 0;

		}

	}

	return result;
}

Eigen::MatrixXd diffOperator(const std::vector<double> &grid, const Eigen::VectorXd &c, int dim_y)
{

	int s = c.size();
	int N = grid.size();
	Eigen::MatrixXd result((N-1)*s*dim_y + dim_y,(N-1)*s*dim_y + dim_y);
	result = Eigen::MatrixXd::Zero((N-1)*s*dim_y + dim_y,(N-1)*s*dim_y + dim_y);
	double tau = 0;

	for(int i=0;i<N-1;i++)
	{
		tau = grid[i+1] -grid[i];
		result.block(i*s*dim_y,i*s*dim_y,s*dim_y,s*dim_y) = (1/tau)*diffLocalOperator(c,dim_y);
	}

	return result;

}

Eigen::MatrixXd intW_ij(int i, int j, double tau,const std::vector<Eigen::MatrixXd>& G, const Eigen::VectorXd &c)
{
	int n = G[0].rows();
	int m = G[0].cols();
	int s = c.size();


	Eigen::MatrixXd result(n,m);
	Eigen::VectorXd omega(3);

	Eigen::VectorXd falsches_c(3);
	falsches_c << 0,(1.0/2-sqrt(3.0)/6),(1.0/2+sqrt(3.0)/6);


	omega << 5.0/18, 4.0/9, 5.0/18;

	for(int row = 0; row < n; row++)
	{
		for(int col = 0; col < m; col++)
		{
			result(row,col) = 0;
			for(int k = 0; k < s; k++)
			{
				result(row,col) += omega(k)*laPol(c(k),i,falsches_c )*laPol(c(k),j,falsches_c )*G[k](row,col);
			}
			result(row,col) = result(row,col)*tau;
		}
	}
	return result;
}

Eigen::MatrixXd intLocalOperator(double tau, const std::vector<Eigen::MatrixXd>& G, const Eigen::VectorXd &c)
{
	int n = G[0].rows();
	int m = G[0].cols();
	int s = c.size();
	Eigen::MatrixXd result(n*s,m*s);

	for(int row = 0; row < s; row++)
	{
		for(int col = 0; col < s; col++)
		{
			//std::cout << intW_ij(row,col,tau,G,c) << std::endl;
			result.block(row*n,col*m,n,m) = intW_ij(row,col,tau,G,c);
		}
	}

	return result;
}

Eigen::VectorXd intW_j(int j, double tau, const std::vector<Eigen::VectorXd>& g, const Eigen::VectorXd& c)
{
	int n = g[0].size();
	int s = c.size();

	Eigen::VectorXd result = Eigen::VectorXd::Zero(n);

	Eigen::VectorXd omega(3);
	Eigen::VectorXd falsches_c(3);
	falsches_c << 0,(1.0/2-sqrt(3.0)/6),(1.0/2+sqrt(3.0)/6);
	omega << 5.0/18, 4.0/9, 5.0/18;

	for(int i = 0; i<n;i++)
	{
		result(i) = 0;
		for(int k = 0; k < s; k++)
		{
			//std::cout << g[k](i) << std::endl;
			result(i) += omega(k)*laPol(c(k),j,falsches_c)*g[k](i);
		}
		result(i) = result(i)*tau;
	}

	return result;

}

Eigen::VectorXd intLocalOperator_lin(double tau, const std::vector<Eigen::VectorXd>& g , const Eigen::VectorXd &c)
{
	int n = g[0].size();
	int s = c.size();
	Eigen::VectorXd result(n*s);

	for(int j = 0; j < s; j++)
	{
			//std::cout << intW_j(j,tau,g,c) << std::endl;
			result.segment(j*n,n) = intW_j(j,tau,g,c);

	}

	return result;
}

}
