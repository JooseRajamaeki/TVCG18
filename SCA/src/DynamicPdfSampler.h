/*

Part of Aalto University Game Tools. See LICENSE.txt for licensing info. 

*/


#ifndef DYNAMICPDFSAMPLER_H
#define DYNAMICPDFSAMPLER_H 
#include <vector>
#include "MathUtils.h"

namespace AaltoGames
{

	///Sampler for sampling from a discrete and dynamically changing pdf.
	///This is implemented using a tree structure where the discrete probabilities are propagated towards the root
	class DynamicPdfSampler
	{
	public:
		DynamicPdfSampler(int N, DynamicPdfSampler *parent=NULL);
		~DynamicPdfSampler();
		void setDensity(int idx, double density);
		double getDensity(int idx);
		void addUniformBias(float uniformSamplingProbability);
		int __fastcall sample();
		void normalize(double sum=1.0);
		double getSum();
	protected:
		DynamicPdfSampler *children[2];
		DynamicPdfSampler *parent;
		DynamicPdfSampler *root;
		double probability;
		bool hasChildren;
		int elemIdx;
		std::vector<DynamicPdfSampler *> leaves;
	};

}

#endif