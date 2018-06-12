/*

Part of Aalto University Game Tools. See LICENSE.txt for licensing info. 

*/

#include "EigenMathUtils.h"
using namespace Eigen;
namespace AaltoGames
{
	

	void addEigenMatrixRow( Eigen::MatrixXf &m )
	{
		Eigen::MatrixXf temp=m;
		m.resize(m.rows()+1,m.cols());
		m.setZero();
		m.block(0,0,temp.rows(),temp.cols())=temp;
	}

	void calcMeanWeighed(const float *input, const double *inputWeights, int vectorLength, int nVectors, Eigen::VectorXf &out_mean)
	{
		//compute mean
		for (int j=0; j<vectorLength; j++){
			double avg=0;
			double wSum=0;
			for (int i=0; i<nVectors; i++){
				double w=inputWeights[i];
				avg+=w*input[i*vectorLength+j];
				wSum+=w;
			}
			avg/=wSum;
			out_mean[j]=(float)avg;
		}
	}
	//input vectors as columns (Eigen defaults to column major storage)
	void calcMeanWeighed(const Eigen::MatrixXf &input, const Eigen::VectorXd &inputWeights, Eigen::VectorXf &out_mean)
	{
		assert(!input.IsRowMajor);
		calcMeanWeighed(input.data(),inputWeights.data(),input.rows(),input.cols(),out_mean);
	}
	void calcCovarWeighed(const float *input, const double *inputWeights, int vectorLength, int nVectors, Eigen::MatrixXf &out_covMat, const Eigen::VectorXf &mean)
	{
		//compute covariance matrix
		out_covMat.setZero();

		//Eigen::MatrixXf inMat = Eigen::MatrixXf::Zero(vectorLength,nVectors);
		//for (int i=0;i<vectorLength;i++){
		//	for(int k=0;k<nVectors;k++){
		//		inMat(i,k) = ((float)inputWeights[k])*(input[k*vectorLength+i]-mean[i]);
		//	}
		//}

		//out_covMat = (inMat*inMat.transpose())*(1.0/((float)nVectors-1.0));

		for (int i=0; i<vectorLength; i++){
			for (int j=0; j<vectorLength; j++){
				double avg=0;
				double iMean=mean[i];
				double jMean=mean[j];
				double wSum=0;
				for (int k=0; k<nVectors; k++){
					double w=inputWeights[k];
					avg+=w*(input[k*vectorLength+i]-iMean)*(input[k*vectorLength+j]-jMean);
					wSum+=w;
				}
				avg/=wSum;
				out_covMat(i,j)=(float)avg;
			}
		}
	}
	void calcCovarWeighed(const Eigen::MatrixXf &input, const Eigen::VectorXd &inputWeights, Eigen::MatrixXf &out_covMat, const Eigen::VectorXf &out_mean)
	{
		calcCovarWeighed(input.data(),inputWeights.data(),input.rows(),input.cols(),out_covMat,out_mean);
	}
	void calcMeanAndCovarWeighed(const float *input, const double *inputWeights, int vectorLength, int nVectors, Eigen::MatrixXf &out_covMat, Eigen::VectorXf &out_mean)
	{
		calcMeanWeighed(input,inputWeights,vectorLength,nVectors,out_mean);
		calcCovarWeighed(input,inputWeights,vectorLength,nVectors,out_covMat,out_mean);
	}
	//input vectors as columns (Eigen defaults to column major storage)
	void calcMeanAndCovarWeighed(const Eigen::MatrixXf &input, const Eigen::VectorXd &inputWeights, Eigen::MatrixXf &out_covMat, Eigen::VectorXf &out_mean)
	{
		calcMeanAndCovarWeighed(input.data(),inputWeights.data(),input.rows(),input.cols(),out_covMat,out_mean);

	}

	void calcMeanAndCovarWeighedVectorized(const Eigen::MatrixXf &input, const Eigen::VectorXd &inputWeights, Eigen::MatrixXf &out_covMat, Eigen::VectorXf &out_mean,Eigen::MatrixXf &temp)
	{
		out_mean=input.col(0); //to resize
		out_mean.setZero();
		double wSumInv=1.0/inputWeights.sum();
		for (int k=0;k<inputWeights.size();k++){
			double w=inputWeights[k];
			out_mean+=input.col(k)*(float)(w*wSumInv);
		}
		out_mean = input.rowwise().mean();
		temp = (input.colwise() - out_mean);
		for (int k=0;k<inputWeights.size();k++){
			temp.col(k) *= (float)(sqrt(inputWeights(k)*wSumInv));	//using square roots, as we only want the normalized weights to be included once for each result element in the multiplication below
		}
		out_covMat = temp*temp.transpose();
	}


	void gaussNewtonFromSamplesWeighed(const Eigen::VectorXf &xb, const Eigen::VectorXf &rb, const Eigen::MatrixXf &X, const Eigen::VectorXf &weights, const Eigen::VectorXf &residuals, float regularization, Eigen::VectorXf &out_result)
	{
		//Summary:
		//out_result=xb - G rb
		//xb is the best sample, rb is the best sample residual vector
		//G=AB'inv(BB'+kI)
		//A.col(i)=weights[i]*(X.row(i)-best sample)'
		//B.col(i)=weights[i]*(residuals - rb)'
		//k=regularization

		//Get xb, r(xb)
		//cv::Mat xb=X.row(bestIndex);
		//cv::Mat rb=residuals.row(bestIndex);

		//Compute A and B
		MatrixXf A=X.transpose();
		MatrixXf B=residuals.transpose();
		for (int i=0; i<A.cols(); i++)
		{
			A.col(i)=weights[i]*(X.row(i).transpose()-xb);
			B.col(i)=weights[i]*(residuals.row(i).transpose()-rb);
		}
		MatrixXf I=MatrixXf::Identity(B.rows(),B.rows());
		I=I*regularization;
		MatrixXf G=(A*B.transpose())*(B*B.transpose()+I).inverse();
		out_result=xb - G * rb;
	}

}//AaltoGames