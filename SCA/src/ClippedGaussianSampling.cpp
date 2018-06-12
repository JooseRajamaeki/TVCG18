/*

Part of Aalto University Game Tools. See LICENSE.txt for licensing info. 

*/
#include "ClippedGaussianSampling.h"
#include <math.h>
#include <vector>
#include "Mathutils.h"
#include <algorithm>
#include <assert.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace AaltoGames
{

	class ClippedGaussianSampler{

	public:

		//Contructor, sets the default parameters
		ClippedGaussianSampler();

		//Sample a Gaussian random number between minValue and maxValue with the given mean and stdev.
		float sample(float mean, float stdev, float minValue, float maxValue );



	private:
		//Lookup table for the standard normal distribution cdf values.
		std::vector<float>	standardCdfTable;
		//Lookup table for the uniform distribution to inverse standardNormal distribution 
		std::vector<float> inverseStandardCdfTable;
		//The interval for the computation of the inverse cdf lookup table values
		float deltaInverse;
		//Lower limit for standardCdfTable
		float lowLim;
		//The interval for the computation of the cdf lookup table values
		float delta;
		//The upper limit for standardCdfTable
		float upLim;
		//standardCdfTable has the values from N(X < lowLim) to N(X < upLim). The values are computed with the interval dX = delta.


		//This function computes the standard normal distribution cdf values to the table standardCdfTable
		void computeCdfTable(void);

		//This function computes the inverse standard normal distribution cdf values to the table standardCdfTable
		void computeinverseCdfTable(void);

		//If x ~ N(mean,stdDev) the function returns y ~ N(0,1)
		float nonstandardToStandard(float x,float mean, float stdDev);

		//If x ~ N(0,1) the function returns y ~ N(mean,stdDev)
		float standardToNonstandard(float x,float mean, float stdDev);

	};



	//This function computes the standard normal distribution cdf values to the table standardCdfTable
	//standardCdfTable has the values from N(X < lowLim) to N(X < upLim). The values are computed with the interval dX = delta.
	void ClippedGaussianSampler::computeCdfTable(void){
		standardCdfTable.clear();
		inverseStandardCdfTable.clear();
		float temp = 0.0;
		float uniTemp =0.0;
		float scalingConst = 1.0f/sqrtf(2.0f*(float)M_PI);
		for (float position = lowLim; position < upLim; position = position + delta){
			temp += delta*scalingConst*expf(-0.5f*position*position);
			while(uniTemp < temp){
				inverseStandardCdfTable.push_back(position-delta);
				uniTemp += deltaInverse;
			}
			standardCdfTable.push_back(temp);
		}
	}

	//Contructor, sets the default parameters
	ClippedGaussianSampler::ClippedGaussianSampler(){
		//Lower limit for standardCdfTable
		lowLim = -6.0;
		//The upper limit for standardCdfTable
		upLim = 6.0;
		//The interval for the computation of the values
		delta = 1.0/256.0;

		deltaInverse = 1.0/2048.0;
		standardCdfTable.clear();
		inverseStandardCdfTable.clear();
	}

	//If x ~ N(mean,stdDev) the function returns y ~ N(0,1)
	float ClippedGaussianSampler::nonstandardToStandard(float x,float mean, float stdDev){
		return (x-mean)/stdDev;
	}

	//If x ~ N(0,1) the function returns y ~ N(mean,stdDev)
	float ClippedGaussianSampler::standardToNonstandard(float x,float mean, float stdDev){
		return (mean + x*stdDev);
	}

	//Clamp x between lower limit lowLim and upper limit upLim.
	float clamp(float x, float lowLim, float upLim){
		assert(lowLim <= upLim);
		return std::max(lowLim,std::min(upLim,x));
	}

	//Sample a Gaussian random number between minValue and maxValue with the given mean and stdev. This is done using cdf inverting.
	float ClippedGaussianSampler::sample(float mean, float stdev, float minValue, float maxValue ){
		if (stdev==0)
			return mean;
		//If the lookup table is empty, populate it.
		if (standardCdfTable.empty()){
			computeCdfTable();
		}

		//Map the values to be used with standard normal distribution
		float minValStd = nonstandardToStandard(minValue,mean,stdev);
		float maxValStd = nonstandardToStandard(maxValue,mean,stdev);

		//Find the indices of the places corresponding to the minimum and maximum allowed value
		int minPlace = (int)ceil( (minValStd - lowLim)/delta );
		int maxPlace = (int)floor( (maxValStd - lowLim)/delta );

		//Find the standard normal distribution cdf values corresponding to the  minimum and maximum allowed value
		minValStd = standardCdfTable[clipMinMaxi(minPlace,0,(int)(standardCdfTable.size()-1))]; 
		maxValStd = standardCdfTable[clipMinMaxi(maxPlace,0,(int)(standardCdfTable.size()-1))];

		float transRand, position;

		//Sample a uniformly distributed random number from interval [0,1]
		transRand = ((float) rand() / (RAND_MAX));

		//Scale the random number appropriately
		transRand = (maxValStd - minValStd)*transRand + minValStd;

		int invCdfIndex = (int)(transRand/deltaInverse);

		invCdfIndex=clipMinMaxi(invCdfIndex,0,inverseStandardCdfTable.size()-1); //to avoid index out of bounds errors on the next line
		position = inverseStandardCdfTable[invCdfIndex];

		//Scale position properly to obtain a truncated Gaussian random number from the originally specified normal distribution
		transRand = standardToNonstandard(position,mean,stdev);


		////Position will correspond to the sampled value in standard normal distribution
		//float position = lowLim;
		////Index for the cdf lookup table
		//int temp = 0;
		////The table size
		//int tabSize = standardCdfTable.size();
		////Find the standard normal distribution cdf value that is greater than the sampled and scaled uniformly distributed random number transRand.
		//while (standardCdfTable[temp] < transRand && temp < tabSize){
		//	temp++;
		//};
		////Transform position to a truncated gaussian random number
		//position = position + temp*delta;

		//Test that the number fullfils the requirements
		//AALTO_ASSERT1(transRand >= minValue);
		//AALTO_ASSERT1(transRand <= maxValue);

		//Clamp just in case of some floating point imprecision or the bounds being outside the tabled range
		transRand = clipMinMaxf(transRand,minValue,maxValue);

		//Return the value
		return transRand;

	}

	static ClippedGaussianSampler s_sampler;
	float randGaussianClipped( float mean, float stdev, float minValue, float maxValue )
	{
		return s_sampler.sample(mean,stdev,minValue,maxValue);
	}

	float randGaussian( float mean, float stdev)
	{
		return s_sampler.sample(mean,stdev,mean-stdev*10.0f,mean+stdev*10.0f);
	}

}