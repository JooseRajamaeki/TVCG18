/*

Part of Aalto University Game Tools. See LICENSE.txt for licensing info. 

*/


#ifndef EIGENMATHUTILS_H
#define EIGENMATHUTILS_H


#include <Eigen/Eigen> 

//Various math utilities depending on the Eigen library.
//Note that for maximum compatibility, there's also plain float array versions of may of the functions.

namespace AaltoGames
{
	void addEigenMatrixRow( Eigen::MatrixXf &m );

	void calcMeanWeighed(const float *input, const double *inputWeights, int vectorLength, int nVectors, Eigen::VectorXf &out_mean);
	//input vectors as columns (Eigen defaults to column major storage)
	void calcMeanWeighed(const Eigen::MatrixXf &input, const Eigen::VectorXd &inputWeights, Eigen::VectorXf &out_mean);
	void calcCovarWeighed(const float *input, const double *inputWeights, int vectorLength, int nVectors, Eigen::MatrixXf &out_covMat, const Eigen::VectorXf &mean);
	void calcCovarWeighed(const Eigen::MatrixXf &input, const Eigen::VectorXd &inputWeights, Eigen::MatrixXf &out_covMat, const Eigen::VectorXf &out_mean);
	void calcMeanAndCovarWeighed(const float *input, const double *inputWeights, int vectorLength, int nVectors, Eigen::MatrixXf &out_covMat, Eigen::VectorXf &out_mean);
	void calcMeanAndCovarWeighed(const Eigen::MatrixXf &input, const Eigen::VectorXd &inputWeights, Eigen::MatrixXf &out_covMat, Eigen::VectorXf &out_mean);
	//faster but not as accurate, as only single precision accumulator is used
	void calcMeanAndCovarWeighedVectorized(const Eigen::MatrixXf &input, const Eigen::VectorXd &inputWeights, Eigen::MatrixXf &out_covMat, Eigen::VectorXf &out_mean,Eigen::MatrixXf &temp);
	//Tries to find x for which the residuals are zero, regularized towards the xb (with residuals rb)
	//X contains samples of x as rows
	void gaussNewtonFromSamplesWeighed(const Eigen::VectorXf &xb, const Eigen::VectorXf &rb, const Eigen::MatrixXf &X, const Eigen::VectorXf &weights, const Eigen::VectorXf &residuals, float regularization, Eigen::VectorXf &out_result);

}//AaltoGames


#endif
