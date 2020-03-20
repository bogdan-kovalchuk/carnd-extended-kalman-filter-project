#include "FusionEKF.h"
#include <iostream>
#include "Eigen/Dense"
#include "tools.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::cout;
using std::endl;
using std::vector;

/**
 * Constructor.
 */
FusionEKF::FusionEKF() {
  is_initialized_ = false;

  previous_timestamp_ = 0;

  // initializing matrices
  R_laser_ = MatrixXd(2, 2);
  R_radar_ = MatrixXd(3, 3);
  H_laser_ = MatrixXd(2, 4);
  Hj_ = MatrixXd(3, 4);

  //measurement covariance matrix - laser
  R_laser_ << 0.0225, 0,
              0, 0.0225;

  //measurement covariance matrix - radar
  R_radar_ << 0.09, 0, 0,
              0, 0.0009, 0,
              0, 0, 0.09;

  //laser measurement matrix
  H_laser_ << 1, 0, 0, 0,
                0, 1, 0, 0;

  //radar measurement matrix
  Hj_ << 1, 1, 0, 0,
         1, 1, 0, 0,
         1, 1, 1, 1;

  //state covariance matrix P_
  ekf_.P_ = MatrixXd(4, 4);
  ekf_.P_ << 1, 0, 0, 0,
             0, 1, 0, 0,
             0, 0, 1000, 0,
             0, 0, 0, 1000;

  //transition matrix F_
  ekf_.F_ = MatrixXd(4, 4);
  ekf_.F_ << 1, 0, 1, 0,
             0, 1, 0, 1,
             0, 0, 1, 0,
             0, 0, 0, 1;
}

/**
 * Destructor.
 */
FusionEKF::~FusionEKF() {}

void FusionEKF::Initialize(const MeasurementPackage &measurement_pack) {
  ekf_.x_ = VectorXd(4);
  ekf_.x_ << 1, 1, 1, 1;

  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
    const double rho = measurement_pack.raw_measurements_[0];
    const double rho_dot = measurement_pack.raw_measurements_(2);
    const double phi = measurement_pack.raw_measurements_[1];
    ekf_.x_(0) = rho * cos(phi);
    ekf_.x_(1) = rho * sin(phi);
    ekf_.x_(2) = rho_dot * cos(phi);
    ekf_.x_(3) = rho_dot * sin(phi);
  } else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
    ekf_.x_(0) = measurement_pack.raw_measurements_[0];
    ekf_.x_(1) = measurement_pack.raw_measurements_[1];
    ekf_.x_(2) = 0.0;
    ekf_.x_(3) = 0.0;
  }

  previous_timestamp_ = measurement_pack.timestamp_;
  is_initialized_ = true;
}

void FusionEKF::ConfigurePrediction(double delta_t) {
  const double dt_2 = delta_t * delta_t;
  const double dt_3 = dt_2 * delta_t;
  const double dt_4 = dt_3 * delta_t;
  const double noise_ax = 9.0;
  const double noise_ay = 9.0;

  ekf_.F_(0, 2) = delta_t;
  ekf_.F_(1, 3) = delta_t;
  ekf_.Q_ = MatrixXd(4, 4);
  ekf_.Q_ << dt_4 / 4.0 * noise_ax, 0, dt_3 / 2.0 * noise_ax, 0,
      0, dt_4 / 4.0 * noise_ay, 0, dt_3 / 2.0 * noise_ay,
      dt_3 / 2.0 * noise_ax, 0, dt_2 * noise_ax, 0,
      0, dt_3 / 2.0 * noise_ay, 0, dt_2 * noise_ay;
}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {
  if (!is_initialized_) {
    Initialize(measurement_pack);
    return;
  }

  /**
   * Prediction
   */

  /**
   * TODO: Update the state transition matrix F according to the new elapsed time.
   * Time is measured in seconds.
   * TODO: Update the process noise covariance matrix.
   * Use noise_ax = 9 and noise_ay = 9 for your Q matrix.
   */

   //compute the time elapsed between the current and previous measurements
   //dt - expressed in seconds
   const double dt = (measurement_pack.timestamp_ - previous_timestamp_) / 1000000.0;
   previous_timestamp_ = measurement_pack.timestamp_;
   ConfigurePrediction(dt);

   ekf_.Predict();
    /**
     * Update
     */

  /**
   * TODO:
   * - Use the sensor type to perform the update step.
   * - Update the state and covariance matrices.
   */

  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
      Hj_ = tools.CalculateJacobian(ekf_.x_);
      ekf_.H_ = Hj_;
      ekf_.R_ = R_radar_;
      ekf_.UpdateEKF(measurement_pack.raw_measurements_);

  } else {
      ekf_.H_ = H_laser_;
      ekf_.R_ = R_laser_;
      ekf_.Update(measurement_pack.raw_measurements_);
  }

  // print the output
  cout << "x_ = " << ekf_.x_ << endl;
  cout << "P_ = " << ekf_.P_ << endl;
}
