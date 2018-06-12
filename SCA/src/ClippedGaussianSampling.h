/*

Part of Aalto University Game Tools. See LICENSE.txt for licensing info. 

*/
#ifndef CLIPPEDGAUSSIANSAMPLING_H
#define CLIPPEDGAUSSIANSAMPLING_H

namespace AaltoGames
{
	//Draws a random number from a Gaussian within the bounds.
	float randGaussianClipped(float mean, float stdev, float minValue, float maxValue);
	//Draws a random number from a Gaussian without bounds.
	float randGaussian(float mean, float stdev);
}

#endif