#include "include/network.hpp"
#include "network.hpp"

Network::Network(std::vector<int> sizes)
    : sizes(sizes), num_layers(sizes.size())
{
  biases.reserve(num_layers);
  weights.reserve(num_layers);

  for (size_t i = 0; i < num_layers - 1; i++)
  {
    biases[i] = Eigen::VectorXd::Random(sizes[i]);
    weights[i] = Eigen::MatrixXd::Random(sizes[i + 1], sizes[i]);
  }
}

void Network::SGD(const data &train, double eta, int size_batch, int epoch)
{
  std::random_device rd;
  std::mt19937 g(rd());

  const auto n = train.size();

  while (epoch-- > 0)
  {
    std::shuffle(train.begin(), train.end(), g);

    std::vector<data> batchs;

    for (size_t i = 0; i < n; i += size_batch)
    {
      data Xj;

      const auto m = std::min(i + size_batch, n);
      for (size_t k = i; k < m; k++)
      {
        Xj.push_back(train[k]);
      }
      batchs.push_back(Xj);
    }

    for (auto &xj : batchs)
    {
      update_network(xj, eta);
    }
  }
}

void Network::update_network(data &Xj, double eta)
{
  std::vector<Eigen::VectorXd> nabla_w(weights.size());
  std::vector<Eigen::VectorXd> nabla_b(biases.size());

  for (const auto &xj : Xj)
  {
    const auto &[dnabla_b, dnabla_w] = backprop(xj);

    std::transform(
        nabla_w.begin(), nabla_w.end(), dnabla_w.begin(), nabla_w.begin(),
        [](Eigen::VectorXd &x, const Eigen::VectorXd &y) -> Eigen::VectorXd
        {
          x += y;
          return x;
        });
    std::transform(nabla_b.begin(), nabla_b.end(), dnabla_b.begin(),
                   nabla_b.begin(), [](Eigen::VectorXd &x, const Eigen::VectorXd &y)
                   {
                     x += y;
                     return x; });
  }

  for (size_t i = 0; i < nabla_w.size(); i++)
  {
    weights[i] -= (eta / Xj.size()) * nabla_w[i];
  }

  for (size_t i = 0; i < nabla_b.size(); i++)
  {
    biases[i] -= (eta / Xj.size()) * nabla_b[i];
  }
}

struct prop Network::backprop(const item &x)
{

  std::vector<Eigen::VectorXd> nabla_w;
  std::vector<Eigen::VectorXd> nabla_b;

  auto [activation, y] = x;
  std::vector<Eigen::VectorXd> al;
  std::vector<Eigen::VectorXd> zs;
  al.push_back(activation);

  for (size_t l = 0; l < num_layers; l++)
  {
    Eigen::VectorXd z = weights[l] * activation + biases[l];

    activation = sigmoid(z);
    al.push_back(activation);
    zs.push_back(z);
  }

  Eigen::VectorXd delta = (al.back() - y).cwiseProduct(dSigmoid(zs.back()));
  nabla_b.back() = delta;
  nabla_w.back() = al[al.size() - 2] * delta;

  // Camada penultima (num_layers - 2)
  for (size_t l = num_layers - 2; l > 1; l--)
  {
    auto z = zs[l];
    auto dsigz = dSigmoid(z);

    delta = (weights[l + 1].transpose() * delta).cwiseProduct(dsigz);
    nabla_b[l] = delta;
    nabla_w[l] = al[al.size() - 2] * delta;
  }

  struct prop p;
  p.nabla_b = nabla_b;
  p.nabla_w = nabla_w;

  return p;
}

Eigen::VectorXd Network::sigmoid(Eigen::VectorXd x)
{
  auto v = x.unaryExpr([](double x)
                       { return 1 / (1 + std::exp(-x)); });
  return v;
}

Eigen::VectorXd Network::dSigmoid(Eigen::VectorXd z)
{
  const auto sigz = sigmoid(z);
  const Eigen::VectorXd one_minus_sigz =
      z.unaryExpr([](double x)
                  { return 1 - x; });

  return sigz.cwiseProduct(one_minus_sigz);
}
