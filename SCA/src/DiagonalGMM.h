/*

Part of Aalto University Game Tools. See LICENSE.txt for licensing info. 

*/

#ifndef DIAGONALGMM_H
#define DIAGONALGMM_H
#include <Eigen/Eigen> 
#include <vector>


namespace AaltoGames
{
	class DynamicPdfSampler;  //forward declare

	class DiagonalGMM
	{
	public:
		DiagonalGMM();
		//Initialize a diagonal Gaussian mixture model with the given weights, means and covariance matrices.
		//The each row of the 'means' matrix is a mean vector of a mixture component
		//Only the diagonals in the 'cov' matrices are used.
		DiagonalGMM(const Eigen::VectorXf weights,const Eigen::MatrixXf& means, const std::vector<Eigen::MatrixXf>& cov);
		//Initialize a diagonal Gaussian mixture model with the given weights, means and covariance matrices.
		//The each row of the 'means' matrix is a mean vector of a mixture component
		DiagonalGMM(const Eigen::VectorXf weights,const Eigen::MatrixXf& means, const std::vector<Eigen::VectorXf> cov);
		void resize(int nComponents, int nDimensions);
		void resample(DiagonalGMM &dst, int nDstComponents);
		void copyFrom(DiagonalGMM &src);
		int sampleComponent();
		int maxWeightComponentIdx();
		void sample(Eigen::VectorXf &dst);
		//Sample within limits. Note that this is only an approximation, as the component normalizing constants are currently computed without limits
		void sampleWithLimits( Eigen::VectorXf &dst, const Eigen::VectorXf &minValues, const Eigen::VectorXf &maxValues  );
		void sampleWithLimits( Eigen::Map<Eigen::VectorXf> &dst, const Eigen::VectorXf &minValues, const Eigen::VectorXf &maxValues  );

		//Call this after manipulating the weights vector. Normalizes the weights and updates the internal data for sampling from the GMM
		void weightsUpdated();
		//inits the GMM with a single component of infinite variance
		//		void setUniform(int nDimensions);
		void setStds(Eigen::VectorXf &src);
		double p(Eigen::VectorXf &v);
		std::vector<Eigen::VectorXf> mean;
		std::vector<Eigen::VectorXf> std;
		//note: after you manipulate the weights, call weigthsUpdated() to normalize them and update the internal data needed for the sampling functions
		Eigen::VectorXd weights;
		//Note: src1 and src2 may also have 0 or FLT_MAX as std, corresponding to fixed or unconstrained variables.
		static void multiply(DiagonalGMM &src1, DiagonalGMM &src2, DiagonalGMM &dst);
		//FixedVars contains all the known variables (the lowest indices). Returns the sum of weights before normalizing to 1.
		//Pass in valid temp and temp2 vectors of same size as fixedVars to avoid heap allocs.
		double makeConditional(const Eigen::VectorXf &fixedVars, DiagonalGMM &dst, Eigen::VectorXf *temp=NULL, Eigen::VectorXf *temp2=NULL);
		//REturns true if all components have constant sd:s
		bool constantSds();
	protected:
		DynamicPdfSampler *sampler;
		int nDimensions;
		int nComponents;
	};

} //AaltoGames


#endif //DIAGONALGMM_H