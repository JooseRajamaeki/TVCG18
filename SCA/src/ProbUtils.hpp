/*

MIT License

Copyright (c) 2017 Joose Rajamäki

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


*/


#ifndef PROB_UTILS_H
#define PROB_UTILS_H

#ifndef M_PI
#define M_PI 3.141592653589793238462643383279502884197169399375105820974944592307816406286208998628
#endif

#include <Eigen/Eigen>
#include "MiscUtils.hpp"

#include <time.h>
#include <iostream>
#include <vector>

#define LOG_2_PI 1.83787706641


double factorial(unsigned long long int);
double doubleFactorial(unsigned long long int);
double error_function(double);
//Return the 'row'th row of Pascal triangle
Eigen::Matrix<int,Eigen::Dynamic,1> pascalTriangle(unsigned int);
//Sample index (limit excluded)
unsigned sampleIndex(const unsigned int&);


template<typename Scalar>
Scalar sampleUniform(void){

	return ((Scalar)rand()/(Scalar)RAND_MAX);

}


template<typename Scalar>
Scalar gaussian_mutual_information(const Scalar& correlation){

	return -(Scalar)0.5 * std::log((Scalar)1 - correlation*correlation);

}

template<typename Scalar>
Scalar gaussian_entropy(const Scalar& variance){

	const Scalar two_pi_e = 17.07946844534713413092710173909314899006977707153022992375;

	return (Scalar)0.5*std::log(two_pi_e*variance);

}

template<typename Scalar>
Scalar epanechnikov_kernel(const Scalar& point_of_evaluation){

	if (std::abs(point_of_evaluation) > 1){
		return 0;
	}

	return (Scalar)0.75*((Scalar)1 - point_of_evaluation*point_of_evaluation);
}

template<typename Scalar>
Scalar gaussian_kullback_leibner_divergence(const Eigen::Matrix<Scalar,Eigen::Dynamic,1>& mu0, const Eigen::Matrix<Scalar,Eigen::Dynamic,1>& mu1, const Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic>& cov0, const Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic>& cov1){

	assert(mu1.size() == mu0.size());

	Scalar result = -mu1.size();

	Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic> precision1 = cov1.inverse();

	result += (precision1*cov0).trace();

	Eigen::Matrix<Scalar,Eigen::Dynamic,1> diff = mu0 - mu1;

	result += diff.dot(precision1*diff);

	result += std::log(cov1.determinant() / cov0.determinant());

	result /= (Scalar)2;

	return result;

}

template<typename Scalar>
Scalar gaussian_kullback_leibner_divergence(const Scalar& mu0, const Scalar& mu1, const Scalar& var0, const Scalar& var1){

	Scalar result = -1;

	result += var0/var1;

	Scalar diff = mu0 - mu1;

	result += diff*diff / (var1);

	result -= std::log(var0 / var1);

	result /= (Scalar)2;

	return result;

}


template<typename Scalar>
Scalar gaussian_variation_of_information(const Scalar& mu0, const Scalar& mu1, const Scalar& var0, const Scalar& var1){

	return gaussian_kullback_leibner_divergence(mu0,mu1,var0,var1) + gaussian_kullback_leibner_divergence(mu1,mu0,var1,var0);

}


template<typename Scalar>
Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic> covariance_matrix_to_correlation_matrix(const Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic>& cov){

	Eigen::Matrix<Scalar,Eigen::Dynamic,1> stds = cov.diagonal();
	stds = stds.cwiseSqrt();

	Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic> correlation_matrix = cov;

	//Diagonal is ones
	for (int row = 0; row < correlation_matrix.rows(); row++){
		correlation_matrix(row,row) = 1;
	}

	//Compute the correlation coefficients to the upper triangle
	for (int row = 0; row < cov.rows(); row++){
		for (int column = row + 1; column < cov.cols(); column++){
			correlation_matrix(row,column) = cov(row,column) / (stds(row) * stds(column));

			if (!finiteNumber(correlation_matrix(row,column))){
				correlation_matrix(row,column) = 0;
			}

		}
	}

	//Copy the values to the lower triangle
	for (int row = 0; row < cov.rows(); row++){
		for (int column = 0; column < row; column++){
			correlation_matrix(row,column) = correlation_matrix(column,row);
		}
	}

	return correlation_matrix;

}


template<typename Scalar>
unsigned sampleLabel(Eigen::Matrix<Scalar,Eigen::Dynamic,1> labelProbs){

	for (int i = 0; i < labelProbs.size();i++){
		//Ensure that all of the entries are positive
		labelProbs(i) = std::abs(labelProbs(i));

		//If some of the entries is inf, make the selection based only on infs
		if (labelProbs(i) > std::numeric_limits<Scalar>::max()){
			for (int j = 0; j < labelProbs.size();j++){
				if (labelProbs(j) > std::numeric_limits<Scalar>::max()){
					labelProbs(j) = 1;
				}
				else{
					labelProbs(j) = 0;
				}
			}
			break;
		}
	}

	Scalar sum = labelProbs.sum();

	//If the sum equals zero return random label
	if (sum == 0){
		return rand() % labelProbs.size();
	}

	while (sum > std::numeric_limits<Scalar>::max()){

		labelProbs = labelProbs/(Scalar)2.0;
		sum = labelProbs.sum();

	}

	Scalar ayn = sum*sampleUniform<Scalar>();
	Scalar cumulant = 0;

	int label = 0;
	while (label < labelProbs.size() - 1){

		cumulant += labelProbs(label);

		if (cumulant >= ayn){
			break;
		}

		label++;
	}

	return label;

}

template<typename Scalar>
Scalar sampleTriangularDistribution(Scalar lowBound, Scalar peak, Scalar upBound){

	assert(lowBound <= peak && peak <= upBound);

	Scalar ayn = sampleUniform<Scalar>();

	if (ayn <= (peak - lowBound)/(upBound - lowBound)){

		return lowBound + sqrt(ayn*(upBound - lowBound)*(peak - lowBound));

	}
	else{

		return upBound - sqrt(((Scalar)1 - ayn)*(upBound - lowBound)*(upBound - peak));

	}

}

template<typename Scalar>
//Return a sample from the exponential distribution by inverting the cdf
Scalar sampleExponentialDistribution(Scalar lambda){
	if (lambda <= std::numeric_limits<Scalar>::min()){
		return std::numeric_limits<Scalar>::infinity();
	}
	Scalar arg = std::max(std::numeric_limits<Scalar>::epsilon(),sampleUniform<Scalar>());
	return -log(arg)/lambda;
}

//Get a sample from the Poisson distribution with the parameter lambda
unsigned int samplePoissonDistribution(const double& lambda);

template<typename Scalar>
//Draw a random numbers from the standard normal distribution
Eigen::Matrix<Scalar,Eigen::Dynamic,1> BoxMuller(unsigned int amount){
	Eigen::Matrix<Scalar,Eigen::Dynamic,1> result = Eigen::Matrix<Scalar,Eigen::Dynamic,1>::Zero(amount);
	for(unsigned int i = 0;i<amount;i = i + 2){
		double rand1 = ((double) rand() / (RAND_MAX));
		while(rand1 < std::numeric_limits<Scalar>::min() ){
			rand1 = ((double) rand() / (RAND_MAX));
		}
		double rand2 = ((double) rand() / (RAND_MAX));
		double module = sqrt(-2.0*log(rand1));
		result(i) = (Scalar)(module*cos(2*M_PI*rand2));
		if (i + 1 < amount){
			result(i+1) = (Scalar)(module*sin(2*M_PI*rand2));
		}
	}
	return result;
}

template<typename Scalar>
//Draw a random numbers from the standard normal distribution
void BoxMuller(Eigen::Matrix<Scalar,Eigen::Dynamic,1>& vector_to_be_filled){

	int idx = 0;

	while (idx < vector_to_be_filled.size()){

		double rand1 = ((double) rand() / (RAND_MAX));
		while(rand1 < std::numeric_limits<Scalar>::min() ){
			rand1 = ((double) rand() / (RAND_MAX));
		}
		double rand2 = ((double) rand() / (RAND_MAX));
		double module = sqrt(-2.0*log(rand1));

		if (idx < vector_to_be_filled.size()){
			vector_to_be_filled[idx++] = (Scalar)(module*cos(2*M_PI*rand2));
		}

		if (idx < vector_to_be_filled.size()){
			vector_to_be_filled[idx++] = (Scalar)(module*sin(2*M_PI*rand2));
		}

	}
}



template<typename Scalar>
//Draw a random numbers from the standard normal distribution
void BoxMuller(Scalar* container_to_be_filled, int amount) {

	int idx = 0;

	while (idx < amount) {

		double rand1 = ((double)rand() / (RAND_MAX));
		while (rand1 <= std::numeric_limits<Scalar>::min()) {
			rand1 = ((double)rand() / (RAND_MAX));
		}
		double rand2 = ((double)rand() / (RAND_MAX));
		double module = sqrt(-2.0*log(rand1));

		if (idx < amount) {
			container_to_be_filled[idx++] = (Scalar)(module*cos(2 * M_PI*rand2));
		}

		if (idx < amount) {
			container_to_be_filled[idx++] = (Scalar)(module*sin(2 * M_PI*rand2));
		}

	}
}


template<typename Scalar>
//Draw a random numbers from the normal distribution N(mu,sigma)
Eigen::Matrix<Scalar,Eigen::Dynamic,1> randn(Eigen::Matrix<Scalar,Eigen::Dynamic,1>& mu, Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic>& sigma, Scalar regularization_noise = 0.0){

	assert(mu.size() == sigma.rows());
	assert(sigma.cols() == sigma.rows());

	//A sample from the standard normal distribution
	Eigen::Matrix<Scalar,Eigen::Dynamic,1> result = BoxMuller<Scalar>((unsigned)mu.size());

	//Scale and shift appropriately
	result = (sigma*result).eval();
	result += mu;

	if (regularization_noise > 0.0f) {
		result += BoxMuller<Scalar>((unsigned)mu.size())*regularization_noise;
	}

	return result;
}


template<typename Scalar>
//Draw a random numbers from the normal distribution N(mu,sigma)
Scalar mahalanobis_distance(const Eigen::Matrix<Scalar,Eigen::Dynamic,1>& pt1, const Eigen::Matrix<Scalar,Eigen::Dynamic,1>& pt2, const Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic>& precision_matrix){

	assert(pt2.size() == pt1.size());
	assert(precision_matrix.cols() == precision_matrix.rows());
	assert(pt2.size() == precision_matrix.rows());

	//A sample from the standard normal distribution
	Eigen::Matrix<Scalar,Eigen::Dynamic,1> diff = pt1 - pt2;

	return diff.dot(precision_matrix*diff);
}


template<typename Scalar>
//Draw a random numbers from the normal distribution N(mu,sigma)
Scalar population_kurtosis(const std::vector<Scalar>& data){

	Scalar kurtosis = (Scalar)0;

	Scalar first_moment = (Scalar)0;
	Scalar second_moment = (Scalar)0;
	Scalar fourth_moment = (Scalar)0;

	for (const Scalar& datum : data){
		first_moment += datum;
	}
	first_moment /= (Scalar)data.size();

	for (const Scalar& datum : data){
		second_moment += std::pow(datum - first_moment,(Scalar)2);
	}
	second_moment /= (Scalar)data.size();

	for (const Scalar& datum : data){
		fourth_moment += std::pow(datum - first_moment,(Scalar)4);
	}
	fourth_moment /= (Scalar)data.size();

	if (second_moment < std::numeric_limits<Scalar>::epsilon()){
		second_moment += std::numeric_limits<Scalar>::epsilon();
	}

	kurtosis = fourth_moment / (second_moment * second_moment);

	assert(finiteNumber(kurtosis));

	return kurtosis;
}

template<typename Scalar>
Scalar normpdf(const Scalar& x,const Scalar& mu,const Scalar& sigma){
	const Scalar INV_SQRT_TWO_TIMES_PI = (Scalar)0.398942280401432677939946059934381868475858631164934657665925;

	Scalar diff = x-mu;
	Scalar arg = -diff*diff/((Scalar)2*sigma*sigma);
	return INV_SQRT_TWO_TIMES_PI/sigma*std::exp(arg);

}

template<typename Scalar>
Scalar normpdf_without_scaling(const Eigen::Matrix<Scalar,Eigen::Dynamic,1>& x,const Eigen::Matrix<Scalar,Eigen::Dynamic,1>& mu,const Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic>& sigma, const Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic>* sigma_inverse = nullptr){
	assert(x.size() == mu.size());
	assert(x.size() == sigma.cols());
	assert(x.size() == sigma.rows());

	if (sigma_inverse){

		return std::exp((Scalar)(-0.5) * (x - mu).transpose() * (*sigma_inverse) * (x - mu));

	}
	else{
		return std::exp((Scalar)(-0.5) * (x - mu).transpose() * pseudoInverseWithTikhonov(sigma) * (x - mu));
	}
}

template<typename Scalar>
//Returns the parameters of the gaussian distribution when the last dimensions are conditioned at point_of_conditioning. <regularization> is the regularization associated with ridge regression.
std::pair<Eigen::Matrix<Scalar,Eigen::Dynamic,1>, Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic> > condition_normal_multivariate_gaussian(const Eigen::Matrix<Scalar,Eigen::Dynamic,1>& mu,const Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic>& sigma, const Eigen::Matrix<Scalar,Eigen::Dynamic,1>& point_of_conditioning, Scalar regularization){


	unsigned conditioning_dim = point_of_conditioning.size();

	Eigen::Matrix<Scalar,Eigen::Dynamic,1> result_mean = mu.head(mu.size() - point_of_conditioning.size());
	//Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic> conditioning_precision_matrix = pseudoInverseWithTikhonov<Scalar>( sigma.bottomRightCorner(conditioning_dim,conditioning_dim) );
	Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic> cov_yy_inv = (sigma.bottomRightCorner(conditioning_dim,conditioning_dim) + Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic>::Identity(conditioning_dim,conditioning_dim)*regularization).inverse();

	////Compute the mean
	//auto& cov_xy = sigma.topRightCorner(result_mean.size(),conditioning_dim);
	//auto& cov_yx = sigma.bottomLeftCorner(conditioning_dim,result_mean.size());

	//Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic> debug_check = cov_yx - cov_xy.transpose();

	result_mean += sigma.topRightCorner(result_mean.size(),conditioning_dim) * cov_yy_inv * (point_of_conditioning - mu.tail(conditioning_dim));

	//Compute the covariance matrix
	Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic> result_covariance = sigma.topLeftCorner(result_mean.size(),result_mean.size()) - sigma.topRightCorner(result_mean.size(),conditioning_dim) * cov_yy_inv * sigma.bottomLeftCorner(conditioning_dim,result_mean.size());

	return std::move(std::make_pair(std::move(result_mean),std::move(result_covariance)));
}



template<typename Scalar>
//Returns the parameters of the gaussian distribution when the last dimensions are conditioned at point_of_conditioning
std::pair<Eigen::Matrix<Scalar,Eigen::Dynamic,1>, Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic> > condition_normal_multivariate_gaussian(const Eigen::Matrix<Scalar,Eigen::Dynamic,1>& mu,const Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic>& sigma, const Eigen::Matrix<Scalar,Eigen::Dynamic,1>& point_of_conditioning){


	unsigned conditioning_dim = point_of_conditioning.size();

	Eigen::Matrix<Scalar,Eigen::Dynamic,1> result_mean = mu.head(mu.size() - point_of_conditioning.size());
	//Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic> conditioning_precision_matrix = pseudoInverseWithTikhonov<Scalar>( sigma.bottomRightCorner(conditioning_dim,conditioning_dim) );
	Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic> cov_yy_inv = invert<Scalar>(sigma.bottomRightCorner(conditioning_dim,conditioning_dim));

	//Compute the mean
	auto& cov_xy = sigma.topRightCorner(result_mean.size(),conditioning_dim);
	auto& cov_yx = sigma.bottomLeftCorner(conditioning_dim,result_mean.size());

	//Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic> debug_check = cov_yx - cov_xy.transpose();

	result_mean += cov_xy * cov_yy_inv * (point_of_conditioning - mu.tail(conditioning_dim));

	//Compute the covariance matrix
	Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic> result_covariance = sigma.topLeftCorner(result_mean.size(),result_mean.size()) - cov_xy * cov_yy_inv * cov_yx;

	return std::make_pair(result_mean,result_covariance);
}



template<typename Scalar>
//Returns the parameters of the gaussian distribution when the last dimensions are conditioned at point_of_conditioning. The unprocessed parts are left as they were and they have no meaning.
void condition_normal_multivariate_gaussian_in_place(Eigen::Matrix<Scalar,Eigen::Dynamic,1>& mu,Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic>& sigma, const Eigen::Matrix<Scalar,Eigen::Dynamic,1>& point_of_conditioning){

	unsigned conditioned_size = mu.size() - point_of_conditioning.size();

	//Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic> conditioning_precision_matrix = pseudoInverseWithTikhonov<Scalar>( sigma.bottomRightCorner(conditioning_dim,conditioning_dim) );
	Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic> conditioning_precision_matrix = sigma.bottomRightCorner(point_of_conditioning.size(),point_of_conditioning.size()).inverse();

	//Compute the mean
	mu.head(conditioned_size) += sigma.topRightCorner(conditioned_size,point_of_conditioning.size()) * conditioning_precision_matrix * ( point_of_conditioning - mu.tail(point_of_conditioning.size()));

	//Compute the covariance matrix
	sigma.topLeftCorner(conditioned_size,conditioned_size) -= sigma.topRightCorner(conditioned_size,point_of_conditioning.size()) * conditioning_precision_matrix * sigma.bottomLeftCorner(point_of_conditioning.size(),conditioned_size);

}


template<typename Scalar>
//Returns the parameters of the gaussian distribution when the last dimensions are conditioned at point_of_conditioning
std::pair<Eigen::Matrix<Scalar,Eigen::Dynamic,1>, Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic> > covariance_matrix_of_conditioned_multivariate_gaussian(const Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic>& sigma, const unsigned conditioning_dim){

	int conditioned_dim = sigma.rows() - conditioning_dim;

	//Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic> conditioning_precision_matrix = pseudoInverseWithTikhonov<Scalar>( sigma.bottomRightCorner(conditioning_dim,conditioning_dim) );
	Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic> conditioning_precision_matrix = sigma.bottomRightCorner(conditioning_dim,conditioning_dim).inverse();

	//Compute the covariance matrix
	Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic> result_covariance = sigma.topLeftCorner(conditioned_dim,conditioned_dim) - sigma.topRightCorner(conditioned_dim,conditioning_dim) * conditioning_precision_matrix * sigma.bottomLeftCorner(conditioning_dim,conditioned_dim);

	return result_covariance;
}


template<typename Scalar>
Scalar normcdf(const Scalar& x,const Scalar& mu,const Scalar& sigma){

	Scalar arg = (x-mu)/(std::sqrt((Scalar)2)*sigma);
	return 0.5*((Scalar)1+error_function(arg));

}


template<typename Scalar>
void covariance_matrix_min(Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic>& cov, const Scalar& min_variance){

	const Scalar min_std = std::sqrt(min_variance);

	Eigen::Matrix<Scalar,Eigen::Dynamic,1> stds = cov.diagonal();
	stds = stds.cwiseSqrt();

	//Compute the correlation coefficients
	for (int row = 0; row < cov.rows(); row++){
		for (int column = 0; column < cov.cols(); column++){
			cov(row,column) = cov(row,column) / (stds(row) * stds(column));

			if (!finiteNumber(cov(row,column))){
				if (row != column){
					cov(row,column) = 0;
				}
				else{
					cov(row,column) = 1;
				}
			}

		}
	}

	//Choose the standard deviations
	for (int i = 0; i < stds.size(); i++){
		stds(i) = std::max(min_std,stds(i));
	}

	//Reassemble the covariance matrix
	for (int row = 0; row < cov.rows(); row++){
		for (int column = 0; column < cov.cols(); column++){
			cov(row,column) = cov(row,column) * (stds(row) * stds(column));
		}
	}

}

template<typename Scalar>
std::vector<Scalar> softmax(std::vector<Scalar> input){

	Scalar sum = 0;
	for (Scalar& element : input){
		element = std::exp(element);
		sum += element;
	}

	if (sum <= (Scalar)0){
		for (Scalar& element : input){
			element = (Scalar)1 / (Scalar)input.size();
		}
		return input;
	}

	for (Scalar& element : input){
		element /= sum;
	}

	return input;
}


template<typename Scalar>
//This function first maps the inputs such that the largest value maps to 'scale' (May be negative) and the smallest value maps to zero. After that it is the regular softmax function. 
std::vector<Scalar> softmax_scaled(std::vector<Scalar> input, const Scalar& scale){

	Scalar minimum = std::numeric_limits<Scalar>::infinity();
	Scalar maximum = -std::numeric_limits<Scalar>::infinity();

	for (Scalar& element : input){
		if (element == element){
			minimum = std::min(element,minimum);
			maximum = std::max(element,maximum);
		}
	}

	Scalar min_max_diff = maximum - minimum;

	//If there is practically no difference just return even distribution.
	if (min_max_diff < std::numeric_limits<Scalar>::epsilon()){
		for (Scalar& element : input){
			element = (Scalar)1 / (Scalar)input.size();
		}
		return input;
	}

	for (Scalar& element : input){
		element = (element-minimum)*scale/min_max_diff;
	}

	maximum = (Scalar)0;
	for (Scalar& element : input){
		element = std::exp(element);
		if (element != element){
			element = (Scalar)0;
		}
		maximum = std::max(element,maximum);
	}

	Scalar sum = 0;
	for (Scalar& element : input){
		element /= maximum;
		sum += element;
	}

	assert(check_valid_vector(input));

	if (sum <= (Scalar)1){
		for (Scalar& element : input){
			element = (Scalar)1 / (Scalar)input.size();
		}
		assert(check_valid_vector(input));
		return input;
	}

	for (Scalar& element : input){
		element /= sum;
	}

	assert(check_valid_vector(input));

	return input;
}

template<typename Scalar>
Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic> invert_covariance_matrix(Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic> cov, Scalar minimum_variance){

	assert (cov.cols() == cov.rows());

	if (cov.diagonal().maxCoeff() < minimum_variance){
		return Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic>::Identity(cov.cols(),cov.rows()) *( (Scalar)1/minimum_variance ) ;
	}
	else{
		Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic> cov_inv = cov.inverse();
		return cov_inv;
	}

}

double inv_normcdf_approx(double p);
double normcdf_approx(const double& x,const double& mu,const double& stdev);

double sample_clipped_gaussian(const double& mean, const double& std, const double& lower_clipping_point, const double& upper_clipping_point);

template<typename Scalar>
//Normalize the elements in 'vector' such that they sum up to one. If the elements sum up to zero, use uniform distribution. If the weights sum up to infinity, set the maximum element(s) to have equal probability.
//Usually (when the figures sum up to a sensible number) divide by the sum of the 'vector' elements.
void normalizeProbabilities(Eigen::Matrix<Scalar,Eigen::Dynamic,1>& vector){
	typedef Eigen::Matrix<Scalar,Eigen::Dynamic,1> Vector;

	if (vector.minCoeff() < 0){
		assert(false);
		vector = (Vector::Ones(vector.size()).array()/(Scalar)vector.size()).matrix();
	}

	for (int i = 0; i < vector.size(); i++){
		if (vector[i] > std::numeric_limits<Scalar>::max()){
			vector[i] = std::numeric_limits<Scalar>::max();
		}
	}
	vector = (vector.array()/vector.maxCoeff()).matrix();

	vector = (vector.array()/vector.sum()).matrix();


	if (!isFinite(vector)){
		vector = (Vector::Ones(vector.size()).array()/(Scalar)vector.size()).matrix();
	}

	assert(isFinite(vector));

}

template<typename Scalar>
Scalar normal_distribution_normalizing_factor(Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic>& covariance_matrix){
	typedef Eigen::Matrix<Scalar,Eigen::Dynamic,1> Vector;
	typedef Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic> Matrix;

	Scalar normalizing_factor = -(Scalar)(0.5*LOG_2_PI)*(Scalar)covariance_matrix.rows();

	Vector eigs = covariance_matrix.eigenvalues().real();
	for (int i = 0; i < eigs.size(); i++){
		eigs[i] = std::max(eigs[i],std::numeric_limits<Scalar>::min());
	}
	normalizing_factor -= (Scalar)0.5*eigs.array().log().matrix().sum();

	normalizing_factor = (Scalar)1/std::exp(normalizing_factor);
	return normalizing_factor;

}



#endif