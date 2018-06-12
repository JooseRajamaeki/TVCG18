/*

Part of Aalto University Game Tools. See LICENSE.txt for licensing info. 

*/
//#define EIGEN_RUNTIME_NO_MALLOC // Define this symbol to enable runtime tests for allocations
#include "DiagonalGMM.h"
#include "DynamicPdfSampler.h"
#include "ClippedGaussianSampling.h"


namespace AaltoGames
{

	static inline double diagGaussianNormalizingConstant(const Eigen::VectorXf &std)
	{
		//the norm. constant for a k-dimensional gaussian is given by 1/sqrt[(2pi)^k det(cov)]
		//For diag. covariance, det(cov)=trace(cov)=squared(std.prod())
		double det=squared(std.prod());
		assert(validFloat(det));
		double result=1/sqrt(pow(2*PI,std.rows())*det);
		assert(result!=0);
		return result;
	}

	static inline double evalGaussianPdf(float x, float mean, float std)
	{
		return 1.0/(std*sqrt(2*PI))*exp(-0.5*squared(x-mean)/squared(std));
	}
	static inline double evalGaussianPdfUnnorm(float x, float mean, float std)
	{
		return exp(-0.5*squared(x-mean)/squared(std));
	}

	static inline double evalGaussianPdf(const Eigen::VectorXf &x, const Eigen::VectorXf &mean, const Eigen::VectorXf &std,bool normalize)
	{
		double normConst=1.0;
		int k=x.rows();
		float quadraticForm=((x-mean).cwiseQuotient(std)).squaredNorm();
		double result=expf(-0.5f*quadraticForm);
		if (normalize)
			return diagGaussianNormalizingConstant(std)*result;
		else
			return result;
	}

	void DiagonalGMM::multiply( DiagonalGMM &src1, DiagonalGMM &src2, DiagonalGMM &dst )
	{
		bool needNormalizing=!(src1.constantSds() && src2.constantSds());
		assert(src1.nDimensions == src2.nDimensions);
		dst.resize(src1.nComponents*src2.nComponents,src1.nDimensions);
		for (int i=0; i<src1.nComponents; i++)
		{
			for (int j=0; j<src2.nComponents; j++)
			{
				double w=1;
				int productIdx=i*src2.nComponents+j;
				if (needNormalizing)
				{
					for (int d=0; d<dst.nDimensions; d++)
					{
						float mean1=src1.mean[i][d], std1=src1.std[i][d];
						double w1=src1.weights[i];
						float mean2=src2.mean[j][d], std2=src2.std[j][d];
						double w2=src2.weights[j];
						float &mean12=dst.mean[productIdx][d], &std12=dst.std[productIdx][d];
						productNormalDist(mean1,std1,mean2,std2,mean12,std12);
						double pSrc1=w1*evalGaussianPdf(mean12,mean1,std1);
						double pSrc2=w2*evalGaussianPdf(mean12,mean2,std2);
						//w12*evalGaussianPdf(mean12,mean12,std12)=pSrc1*pSrc2 => 
						double w12=pSrc1*pSrc2/evalGaussianPdf(mean12,mean12,std12);
						//the multivariate weight is a product of all w12
						w*=w12;
					}
				}
				else
				{
					for (int d=0; d<dst.nDimensions; d++)
					{
						float mean1=src1.mean[i][d], std1=src1.std[i][d];
						double w1=src1.weights[i];
						float mean2=src2.mean[j][d], std2=src2.std[j][d];
						double w2=src2.weights[j];
						float &mean12=dst.mean[productIdx][d], &std12=dst.std[productIdx][d];
						productNormalDist(mean1,std1,mean2,std2,mean12,std12);
						double pSrc1=w1*evalGaussianPdfUnnorm(mean12,mean1,std1);
						double pSrc2=w2*evalGaussianPdfUnnorm(mean12,mean2,std2);
						double w12=pSrc1*pSrc2/evalGaussianPdfUnnorm(mean12,mean12,std12);
						w*=w12;
					}
					
				}
				dst.weights[productIdx]=w;
			}
		}
		dst.weightsUpdated();
	}


	void DiagonalGMM::resize( int nComponents, int nDimensions )
	{
		if (nComponents!=this->nComponents || nDimensions!=this->nDimensions)
		{
			if (sampler!=NULL)
				delete sampler;
			sampler=new DynamicPdfSampler(nComponents);
			this->nComponents=nComponents;
			this->nDimensions=nDimensions;
			mean.resize(nComponents);
			std.resize(nComponents);
			weights.resize(nComponents);
			weights.setConstant(0);
			for (int i=0; i<nComponents; i++)
			{
				mean[i].resize(nDimensions);
				std[i].resize(nDimensions);
			}
		}
	}

	DiagonalGMM::DiagonalGMM()
	{
		nDimensions=-1;
		nComponents=-1;
		sampler=NULL;
	}


	DiagonalGMM::DiagonalGMM(const Eigen::VectorXf weightsIn,const Eigen::MatrixXf& means, const std::vector<Eigen::MatrixXf>& cov){
		nDimensions = means.cols();
		nComponents = means.rows();
		//TODO Find out what sampler does and initialize it here properly
		sampler=NULL;
		this->weights = (weightsIn*(1.0f/weightsIn.sum())).cast<double>();
		this->mean.clear();
		for (int i=0;i < means.rows();i++){
			this->mean.push_back(means.row(i).transpose());
			//TODO Find out if this is really the standard deviation or the variance. If this really is the standard deviation, we can make all the things faster by using directly variances in the computations
			this->std.push_back(cov[i].diagonal());
		}
	}

	DiagonalGMM::DiagonalGMM(const Eigen::VectorXf weightsIn,const Eigen::MatrixXf& means, const std::vector<Eigen::VectorXf> cov){
		nDimensions = means.cols();
		nComponents = means.rows();
		//TODO Find out what sampler does and initialize it here properly
		sampler=NULL;
		this->weights = (weightsIn*(1.0f/weightsIn.sum())).cast<double>();
		this->mean.clear();
		for (int i=0;i < means.rows();i++){
			this->mean.push_back(means.row(i).transpose());
		}
		this->std = cov;
	}

	void DiagonalGMM::copyFrom( DiagonalGMM &src )
	{
		resize(src.nComponents,src.nDimensions);
		mean=src.mean;
		std=src.std;
		weights=src.weights;
		weightsUpdated();
	}

	void DiagonalGMM::weightsUpdated()
	{
		//normalize weights
		double maxWeight=weights.maxCoeff();
		double sum=weights.sum();
		if (validFloat(maxWeight/sum))
			weights/=sum;

		//update the component sampling
		for (int i=0; i<nComponents; i++)
		{
			sampler->setDensity(i,weights[i]);
		}
	}

	int DiagonalGMM::sampleComponent()
	{
		return sampler->sample();
	}
	void DiagonalGMM::setStds( Eigen::VectorXf &src )
	{
		for (int i=0; i<nComponents; i++)
		{
			std[i]=src;
		}
	}
	void DiagonalGMM::resample( DiagonalGMM &dst, int nDstComponents )
	{
		dst.resize(nDstComponents,nDimensions);
		for (int i=0; i<nDstComponents; i++)
		{
			int c=sampleComponent();
			dst.weights[i]=1;
			dst.mean[i]=mean[c];
			dst.std[i]=std[c];
		}
		dst.weightsUpdated();
	}

	double DiagonalGMM::p(Eigen::VectorXf &v )
	{
		double densitySum=0;
		for (int i=0; i<nComponents; i++)
		{
			densitySum+=weights[i]*evalGaussianPdf(v,mean[i],std[i],true);
		}
		return densitySum;
	}

	double DiagonalGMM::makeConditional(const Eigen::VectorXf &fixedVars, DiagonalGMM &dst,Eigen::VectorXf *temp, Eigen::VectorXf *temp2 )
	{
		bool needNormalizing=!constantSds();
		int nFixed=fixedVars.rows();
		int nNonFixed=nDimensions-nFixed;
		dst.resize(nComponents,nNonFixed);

		//	printf("State kernel std: %e\n",std[nComponents-1][0]);
		for (int i=0; i<nComponents; i++)
		{
			//Diagonal covariance => the conditional gmm component means and stdevs stay the same, only weights differ
			dst.mean[i]=mean[i].tail(nNonFixed);
			dst.std[i]=std[i].tail(nNonFixed);
			if (temp==NULL && temp2==NULL)
				dst.weights[i]=weights[i]*evalGaussianPdf(fixedVars,mean[i].head(nFixed),std[i].head(nFixed),needNormalizing);
			else
			{
				*temp=mean[i].head(nFixed);
				*temp2=std[i].head(nFixed);
				dst.weights[i]=weights[i]*evalGaussianPdf(fixedVars,*temp,*temp2,needNormalizing);
			}
		}
		//		std::cout << "fixed vars" << fixedVars << "\n";
		double result=dst.weights.sum();
		//	printf("Weight sum before makeConditional() %e, after %e\n",weights.sum(),dst.weights.sum());
		dst.weightsUpdated();

		return result;
	}

	void DiagonalGMM::sample( Eigen::VectorXf &dst )
	{
		int idx=sampleComponent();
		if (dst.rows()!=nDimensions)
			dst.resize(nDimensions);
		for (int d=0; d<nDimensions; d++)
		{
			dst[d]=randGaussianClipped(mean[idx][d],std[idx][d],mean[idx][d]-std[idx][d]*10.0f,mean[idx][d]+std[idx][d]*10.0f);
		}
	}


	void DiagonalGMM::sampleWithLimits( Eigen::VectorXf &dst, const Eigen::VectorXf &minValues, const Eigen::VectorXf &maxValues  )
	{
		int idx=sampleComponent();
		//printf("sampleWithLimits selected component %d\n",idx);
		if (dst.rows()!=nDimensions)
			dst.resize(nDimensions);
		for (int d=0; d<nDimensions; d++)
		{
			dst[d]=randGaussianClipped(mean[idx][d],std[idx][d],minValues[d],maxValues[d]);
		}
	}

	void DiagonalGMM::sampleWithLimits( Eigen::Map<Eigen::VectorXf> &dst, const Eigen::VectorXf &minValues, const Eigen::VectorXf &maxValues )
	{
		int idx=sampleComponent();
		//printf("sampleWithLimits selected component %d\n",idx);
		if (dst.rows() != nDimensions)
			std::exception("Invalid dst dimensions for sampleWithLimits()!");
		for (int d=0; d<nDimensions; d++)
		{
			dst[d]=randGaussianClipped(mean[idx][d],std[idx][d],minValues[d],maxValues[d]);
		}
	}

	int DiagonalGMM::maxWeightComponentIdx()
	{
		int result=0;
		double maxw=-1;
		for (int i=0; i<weights.rows(); i++)
		{
			if (weights[i]>maxw)
			{
				maxw=weights[i];
				result=i;
			}
		}
		return result;
	}

	bool DiagonalGMM::constantSds()
	{
		for (int i=1; i<nComponents; i++)
		{
			if (std[i]!=std[0])
				return false;
		}
		return true;
	}

} //AaltoGames