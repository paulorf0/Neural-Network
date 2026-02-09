#pragma once

#include <eigen3/Eigen/Dense>
#include <vector>
#include <algorithm>
#include <random>

// x, y = Entrada, Esperado
using item = std::pair<Eigen::VectorXd, Eigen::VectorXd>;
using data = std::vector<item>;


struct prop{
  std::vector<Eigen::MatrixXd> nabla_w;
  std::vector<Eigen::VectorXd> nabla_b;
};

class Network
{
public:
  Network(std::vector<int> sizes);
  void fit(data &train, double eta, int size_batch, int epoch);
  Eigen::VectorXd transform(Eigen::VectorXd a);

private:
  void update_network(data &xj, double eta);
  struct prop backprop(const item &x);

  Eigen::VectorXd sigmoid(Eigen::VectorXd x);
  Eigen::VectorXd dSigmoid(Eigen::VectorXd z);

  int num_layers;
  std::vector<int> sizes;
  std::vector<Eigen::VectorXd> biases;
  std::vector<Eigen::MatrixXd> weights;
};
