#include "tools.h"
#include <iostream>

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

namespace {

const int kStateDimension = 4;

bool ValidRmseInputs(const vector<VectorXd>& estimations,
                     const vector<VectorXd>& ground_truth) {
    if (estimations.empty() || estimations.size() != ground_truth.size()) {
        return false;
    }
    for (size_t i = 0; i < estimations.size(); ++i) {
        if (estimations[i].size() != kStateDimension ||
            ground_truth[i].size() != kStateDimension ||
            !estimations[i].allFinite() || !ground_truth[i].allFinite()) {
            return false;
        }
    }
    return true;
}

}  // namespace

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations, const vector<VectorXd> &ground_truth)
{
    VectorXd rmse = VectorXd::Zero(kStateDimension);

    // check the validity of the following inputs:
    //  * the estimation vector size should not be zero
    //  * the estimation vector size should equal ground truth vector size
    if (!ValidRmseInputs(estimations, ground_truth))
    {
        std::cout << "Invalid estimation or ground_truth data" << std::endl;
        return rmse;
    }

    // accumulate squared residuals
    for (unsigned int i=0; i < estimations.size(); ++i)
    {
        VectorXd residual = estimations[i] - ground_truth[i];

        // coefficient-wise multiplication
        residual = residual.array()*residual.array();
        rmse += residual;
    }

    // calculate the mean
    rmse = rmse/estimations.size();

    // calculate the squared root
    rmse = rmse.array().sqrt();

    // return the result
    return rmse.allFinite() ? rmse : VectorXd::Zero(kStateDimension);
}

VectorXd Tools::CalculateRMSEIncremental(const vector<VectorXd> &estimations,
                                         const vector<VectorXd> &ground_truth) {
    VectorXd rmse = VectorXd::Zero(kStateDimension);
    if (!ValidRmseInputs(estimations, ground_truth)) {
        std::cout << "Invalid estimation or ground_truth data" << std::endl;
        return rmse;
    }

    for (size_t i = 0; i < estimations.size(); ++i) {
        VectorXd residual = estimations[i] - ground_truth[i];
        const VectorXd squared = residual.array().square().matrix();
        rmse += (squared - rmse) / static_cast<double>(i + 1);
    }
    rmse = rmse.array().sqrt().matrix();
    return rmse.allFinite() ? rmse : VectorXd::Zero(kStateDimension);
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state)
{

    MatrixXd Hj(3,4);
    // recover state parameters
    float px = x_state(0);
    float py = x_state(1);
    float vx = x_state(2);
    float vy = x_state(3);

    // pre-compute a set of terms to avoid repeated calculation
    float c1 = px*px+py*py;
    float c2 = sqrt(c1);
    float c3 = (c1*c2);

    // check division by zero
    if (fabs(c1) < 0.0001) {
        std::cout << "CalculateJacobian () - Error - Division by Zero" << std::endl;
        return Hj;
    }

    // compute the Jacobian matrix
    Hj << (px/c2), (py/c2), 0, 0,
            -(py/c1), (px/c1), 0, 0,
            py*(vx*py - vy*px)/c3, px*(px*vy - py*vx)/c3, px/c2, py/c2;

    return Hj;
}
