#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "tools.h"

namespace {

bool Near(double actual, double expected, double tolerance = 1e-9) {
  return std::fabs(actual - expected) <= tolerance;
}

void Require(bool condition, const std::string& message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << std::endl;
    std::exit(1);
  }
}

Eigen::VectorXd Vector4(double a, double b, double c, double d) {
  Eigen::VectorXd value(4);
  value << a, b, c, d;
  return value;
}

void TestRmseCharacterization() {
  Tools tools;
  std::vector<Eigen::VectorXd> estimations;
  estimations.push_back(Vector4(1.0, 1.0, 0.2, 0.1));
  estimations.push_back(Vector4(2.0, 2.0, 0.3, 0.2));
  estimations.push_back(Vector4(3.0, 3.0, 0.4, 0.3));

  std::vector<Eigen::VectorXd> truth;
  truth.push_back(Vector4(1.1, 1.1, 0.3, 0.2));
  truth.push_back(Vector4(1.9, 2.1, 0.4, 0.3));
  truth.push_back(Vector4(3.2, 2.8, 0.5, 0.4));

  const Eigen::VectorXd rmse = tools.CalculateRMSE(estimations, truth);
  Require(rmse.size() == 4, "RMSE has four state components");
  Require(Near(rmse(0), 0.14142135623730964), "RMSE position x");
  Require(Near(rmse(1), 0.14142135623730964), "RMSE position y");
  Require(Near(rmse(2), 0.1), "RMSE velocity x");
  Require(Near(rmse(3), 0.1), "RMSE velocity y");
}

void TestRmseExactEstimates() {
  Tools tools;
  std::vector<Eigen::VectorXd> values;
  values.push_back(Vector4(-2.0, 0.0, 3.5, -4.0));
  values.push_back(Vector4(1.0, 2.0, -3.0, 4.0));

  const Eigen::VectorXd rmse = tools.CalculateRMSE(values, values);
  Require(rmse.isZero(1e-12), "exact estimates have zero RMSE");
}

}  // namespace

int main() {
  TestRmseCharacterization();
  TestRmseExactEstimates();
  std::cout << "All EKF tests passed" << std::endl;
  return 0;
}
