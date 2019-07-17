#include "tools.h"
#include <iostream>

#define EPS 0.0001 // A very small number

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::cout;
using std::endl;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  /**
   * TODO: Calculate the RMSE here.
   */

  VectorXd rmse(4);
   rmse << 0,0,0,0;

   if(estimations.size() == 0){
      cout << "ERROR - CalculateRMSE () - The estimations vector is empty" << endl;
      return rmse;
   }

   if(ground_truth.size() == 0){
      cout << "ERROR - CalculateRMSE () - The ground-truth vector is empty" << endl;
      return rmse;
   }

   unsigned int n = estimations.size();
   if(n != ground_truth.size()){
      cout << "ERROR - CalculateRMSE () - The ground-truth and estimations vectors must have the same size." << endl;
      return rmse;
   }

   for(unsigned int i=0; i < estimations.size(); ++i){
      VectorXd diff = estimations[i] - ground_truth[i];
      diff = diff.array()*diff.array();
      rmse += diff;
   }

   rmse = rmse / n;
   rmse = rmse.array().sqrt();
   cout << "INFO - CalculateRMSE () - RMSE = " << rmse << endl;
   return rmse;
     
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {
  /**
   * TODO:
   * Calculate a Jacobian here.
   */

  MatrixXd Hj(3,4);

  if ( x_state.size() != 4 ) {
    cout << "ERROR - CalculateJacobian () - Invalid Vector Size ! The state vector must have size 4." << endl;
    return Hj;
  }
	//recovering state parameters
	double px = x_state(0);
	double py = x_state(1);
	double vx = x_state(2);
	double vy = x_state(3);

	//pre-computing a set of terms to avoid repeated calculation for Jacobian
	double c1 = px*px+py*py;
	double c2 = sqrt(c1);
	double c3 = (c1*c2);

	//checking division by zero
	if(fabs(c1) < EPS){
		cout << "ERROR - CalculateJacobian () - Division by Zero" << endl;
		return Hj;
	}

	//computing  the Jacobian matrix
	Hj << (px/c2), (py/c2), 0, 0,
		  -(py/c1), (px/c1), 0, 0,
		  py*(vx*py - vy*px)/c3, px*(px*vy - py*vx)/c3, px/c2, py/c2;

	return Hj;
}
