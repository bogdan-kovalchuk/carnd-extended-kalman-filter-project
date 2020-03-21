#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "tools.h"
#include "FusionEKF.h"

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

MeasurementPackage Laser(double px, double py, long long timestamp) {
  MeasurementPackage measurement;
  measurement.sensor_type_ = MeasurementPackage::LASER;
  measurement.timestamp_ = timestamp;
  measurement.raw_measurements_ = Eigen::VectorXd(2);
  measurement.raw_measurements_ << px, py;
  return measurement;
}

MeasurementPackage Radar(double rho, double phi, double rho_dot, long long timestamp) {
  MeasurementPackage measurement;
  measurement.sensor_type_ = MeasurementPackage::RADAR;
  measurement.timestamp_ = timestamp;
  measurement.raw_measurements_ = Eigen::VectorXd(3);
  measurement.raw_measurements_ << rho, phi, rho_dot;
  return measurement;
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

void TestIncrementalRmseMatchesBatch() {
  Tools tools;
  std::vector<Eigen::VectorXd> estimations;
  estimations.push_back(Vector4(10.0, -3.0, 0.5, 2.0));
  estimations.push_back(Vector4(-1.0, 4.0, -2.0, 1.0));
  std::vector<Eigen::VectorXd> truth;
  truth.push_back(Vector4(8.0, -4.0, 1.5, 4.0));
  truth.push_back(Vector4(2.0, 3.0, -1.0, -2.0));

  const Eigen::VectorXd batch = tools.CalculateRMSE(estimations, truth);
  const Eigen::VectorXd incremental = tools.CalculateRMSEIncremental(estimations, truth);
  Require((batch - incremental).norm() < 1e-12,
          "incremental RMSE matches batch RMSE");
}

void TestRmseRejectsInvalidInputs() {
  Tools tools;
  std::vector<Eigen::VectorXd> valid(1, Vector4(1.0, 2.0, 3.0, 4.0));
  std::vector<Eigen::VectorXd> short_value;
  short_value.push_back(Eigen::VectorXd::Zero(3));
  std::vector<Eigen::VectorXd> non_finite(1, Vector4(1.0, 2.0, 3.0, 4.0));
  non_finite[0](2) = std::numeric_limits<double>::infinity();

  Require(tools.CalculateRMSE(valid, short_value).isZero(1e-12),
          "RMSE rejects a wrong state dimension");
  Require(tools.CalculateRMSE(non_finite, valid).isZero(1e-12),
          "RMSE rejects non-finite values");
  Require(tools.CalculateRMSEIncremental(std::vector<Eigen::VectorXd>(), valid).isZero(1e-12),
          "incremental RMSE rejects an empty input");
}

void TestJacobianCharacterization() {
  Tools tools;
  const Eigen::VectorXd state = Vector4(3.0, 4.0, 5.0, 6.0);
  const Eigen::MatrixXd jacobian = tools.CalculateJacobian(state);

  Require(jacobian.rows() == 3 && jacobian.cols() == 4,
          "Jacobian has radar measurement dimensions");
  Require(Near(jacobian(0, 0), 0.6, 1e-6) && Near(jacobian(0, 1), 0.8, 1e-6),
          "Jacobian range derivatives");
  Require(Near(jacobian(1, 0), -0.16, 1e-6) && Near(jacobian(1, 1), 0.12, 1e-6),
          "Jacobian bearing derivatives");
  Require(Near(jacobian(2, 0), 0.064, 1e-6) && Near(jacobian(2, 1), -0.048, 1e-6),
          "Jacobian range-rate position derivatives");
  Require(Near(jacobian(2, 2), 0.6, 1e-6) && Near(jacobian(2, 3), 0.8, 1e-6),
          "Jacobian range-rate velocity derivatives");
}

void TestNumericalJacobianMatchesAnalytic() {
  Tools tools;
  const Eigen::VectorXd state = Vector4(-2.5, 4.0, 1.5, -3.0);
  const Eigen::MatrixXd analytic = tools.CalculateJacobian(state);
  const Eigen::MatrixXd numerical = tools.CalculateJacobianNumerical(state);
  Require((analytic - numerical).norm() < 1e-5,
          "central-difference Jacobian matches the analytic Jacobian");
}

void TestJacobianRejectsUnsafeStates() {
  Tools tools;
  Eigen::VectorXd wrong_dimension = Eigen::VectorXd::Zero(3);
  Eigen::VectorXd non_finite = Vector4(1.0, 2.0, 3.0, 4.0);
  non_finite(0) = std::numeric_limits<double>::quiet_NaN();
  const Eigen::VectorXd near_origin = Vector4(1e-5, 0.0, 1.0, 1.0);

  Require(tools.CalculateJacobian(wrong_dimension).isZero(1e-12),
          "Jacobian rejects a wrong state dimension");
  Require(tools.CalculateJacobian(non_finite).isZero(1e-12),
          "Jacobian rejects non-finite states");
  Require(tools.CalculateJacobianNumerical(near_origin).isZero(1e-12),
          "numerical Jacobian rejects near-zero positions");
}

void TestLaserInitialization() {
  FusionEKF fusion;
  fusion.ProcessMeasurement(Laser(2.5, -1.5, 1000000));

  Require((fusion.ekf_.x_ - Vector4(2.5, -1.5, 0.0, 0.0)).norm() < 1e-12,
          "laser initialization preserves position and clears velocity");
  Require(Near(fusion.ekf_.P_(0, 0), 1.0) && Near(fusion.ekf_.P_(1, 1), 1.0) &&
              Near(fusion.ekf_.P_(2, 2), 1000.0) && Near(fusion.ekf_.P_(3, 3), 1000.0),
          "laser initialization keeps configured uncertainty");
}

void TestRadarInitializationAndMeasurementPaths() {
  FusionEKF radar_fusion;
  radar_fusion.ProcessMeasurement(Radar(5.0, 0.0, 2.0, 1000000));
  Require((radar_fusion.ekf_.x_ - Vector4(5.0, 0.0, 2.0, 0.0)).norm() < 1e-12,
          "radar initialization converts polar velocity");

  FusionEKF fusion;
  fusion.ProcessMeasurement(Laser(1.0, 1.0, 1000000));
  fusion.ProcessMeasurement(Laser(1.1, 1.0, 1100000));
  Require(Near(fusion.ekf_.F_(0, 2), 0.1, 1e-6) &&
              Near(fusion.ekf_.F_(1, 3), 0.1, 1e-6),
          "laser update applies the elapsed timestamp interval");
  Require(Near(fusion.ekf_.Q_(0, 0), 0.000225, 1e-7) &&
              Near(fusion.ekf_.Q_(2, 2), 0.09, 1e-6),
          "process noise follows the elapsed interval");
  Require(Near(fusion.ekf_.Q_(0, 2), 0.0045, 1e-6) &&
              Near(fusion.ekf_.Q_(2, 0), 0.0045, 1e-6),
          "process noise keeps position and velocity covariance coupled");

  fusion.ProcessMeasurement(Radar(1.5, 0.7, 0.0, 1200000));
  Require(fusion.ekf_.H_.rows() == 3 && fusion.ekf_.H_.cols() == 4,
          "radar path installs a Jacobian measurement model");
  Require(fusion.ekf_.x_.allFinite() && fusion.ekf_.P_.allFinite(),
          "measurement paths keep state and covariance finite");
  Require((fusion.ekf_.P_ - fusion.ekf_.P_.transpose()).norm() < 1e-9,
          "measurement paths preserve covariance symmetry");
}

void TestFusionRejectsMalformedAndStaleMeasurements() {
  FusionEKF fusion;
  MeasurementPackage malformed = Laser(1.0, 2.0, 1000000);
  malformed.raw_measurements_ = Eigen::VectorXd::Zero(3);
  fusion.ProcessMeasurement(malformed);
  fusion.ProcessMeasurement(Laser(1.0, 2.0, 1000000));

  const Eigen::VectorXd before_state = fusion.ekf_.x_;
  const Eigen::MatrixXd before_covariance = fusion.ekf_.P_;
  MeasurementPackage non_finite = Radar(2.0, 0.0, 1.0, 1100000);
  non_finite.raw_measurements_(1) = std::numeric_limits<double>::infinity();
  fusion.ProcessMeasurement(non_finite);
  fusion.ProcessMeasurement(Laser(9.0, 9.0, 999999));

  Require((fusion.ekf_.x_ - before_state).norm() < 1e-12 &&
              (fusion.ekf_.P_ - before_covariance).norm() < 1e-12,
          "malformed and stale measurements leave the filter unchanged");
}

void TestFusionHandlesRadarAtOrigin() {
  FusionEKF fusion;
  fusion.ProcessMeasurement(Radar(0.0, 0.0, 0.0, 1000000));
  fusion.ProcessMeasurement(Radar(0.0, 0.0, 0.0, 1100000));

  Require(fusion.ekf_.x_.allFinite() && fusion.ekf_.P_.allFinite(),
          "near-origin radar updates do not create non-finite filter values");
  Require(fusion.ekf_.H_.isZero(1e-12),
          "near-origin radar updates use the guarded zero Jacobian");
}

}  // namespace

int main() {
  TestRmseCharacterization();
  TestRmseExactEstimates();
  TestIncrementalRmseMatchesBatch();
  TestRmseRejectsInvalidInputs();
  TestJacobianCharacterization();
  TestNumericalJacobianMatchesAnalytic();
  TestJacobianRejectsUnsafeStates();
  TestLaserInitialization();
  TestRadarInitializationAndMeasurementPaths();
  TestFusionRejectsMalformedAndStaleMeasurements();
  TestFusionHandlesRadarAtOrigin();
  std::cout << "All EKF tests passed" << std::endl;
  return 0;
}
