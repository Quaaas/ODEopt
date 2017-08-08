#include <vector>
#include <eigen3/Eigen/Dense>
#include "util.hh"

class ODEoptVector{
  public:
    int dim_y_;
    int dim_u_;
    int dim_r_;
    int N_col_;
    int N_grid_;

    Eigen::VectorXd x_;
    Eigen::VectorXd c_;

    std::vector<double> grid_;
    std::vector<Eigen::VectorXd> Y;
    std::vector<Eigen::VectorXd> U;

    ODEoptVector(Eigen::VectorXd x, int dim_y, int dim_u, int dim_r,
                  std::vector<double> grid, Eigen::VectorXd c);

    Eigen::VectorXd eval_x(double t);
    Eigen::VectorXd eval_u(double t);
  };
