#include "ODEoptVector.hh"

ODEoptVector::ODEoptVector(Eigen::VectorXd x,
                            int dim_y,
                            int dim_u,
                            int dim_r,
                            std::vector<double> grid,
                            Eigen::VectorXd c)
                            : x_(x),
                              dim_y_(dim_y),
                              dim_u_(dim_u),
                              dim_r_(dim_r),
                              grid_(grid),
                              c_(c)
  {
    N_grid_ = grid.size();
    N_col_ = c.size();
    for(int i = 0; i < N_col_*(N_grid_-1); i++)
    {
        Y.push_back(x.segment(i*dim_y_, dim_y_));
        U.push_back(x.segment((N_grid_-1)*dim_y_*N_col_+ dim_y_ + i*dim_u_, dim_u_));
    }
  }

Eigen::VectorXd ODEoptVector::eval_x(double t)
{
  if((t<grid_[0]) || (t>grid_[N_grid_ - 1]))
  {
    std::cout<< "Error: out of interval:   "  << t << std::endl;
  }

  int j = 0;
  for(int i = 0; i<N_grid_-1;i++)
  {
    if((t>=grid_[i])&&(t<grid_[i+1])) j = i;
  }

  double tau = grid_[j+1] - grid_[j];
  Eigen::VectorXd eval = Polynomial::evalOperator(t,Eigen::VectorXd::Ones(c_.size())*grid_[j] + tau*c_);
  Eigen::VectorXd result = Eigen::VectorXd::Zero(dim_y_);

  for(int i = 0; i<N_col_; i++)
  {
    result += eval(i)*Y[N_col_*j + i];
  }

  return result;
}

Eigen::VectorXd ODEoptVector::eval_u(double t)
{
  if((t<grid_[0]) || (t>grid_[N_grid_ - 1]))
  {
    std::cout<< "Error: out of interval:   "  << t << std::endl;
  }

  int j = 0;
  for(int i = 0; i<N_grid_-1;i++)
  {
    if((t>=grid_[i])&&(t<grid_[i+1])) j = i;
  }

  double tau = grid_[j+1] - grid_[j];
  Eigen::VectorXd eval = Polynomial::evalOperator(t,Eigen::VectorXd::Ones(c_.size())*grid_[j] + tau*c_);
  Eigen::VectorXd result = Eigen::VectorXd::Zero(dim_u_);

  for(int i = 0; i<N_col_; i++)
  {
    result += eval(i)*U[N_col_*j + i];
  }

  return result;
}
