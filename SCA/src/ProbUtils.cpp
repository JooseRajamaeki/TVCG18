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

#include "ProbUtils.hpp"


double factorial(unsigned long long int arg){
	if (arg <= 1){
		return (double)1;
	}
	else{
		assert(arg-1 != arg);
		return (double)arg*(double)factorial(arg-1);
	}
}


double doubleFactorial(unsigned long long int arg){
	if (arg <= 1){
		return (double)1;
	}
	else{
		return (double)arg*(double)doubleFactorial(arg-2);
	}
}

//Sample index (argument excluded)
unsigned sampleIndex(const unsigned int& max){
	return rand() % max;
}

//Return the 'row'th row of Pascal triangle
Eigen::Matrix<int,Eigen::Dynamic,1> pascalTriangle(unsigned int row){
	if (row == 0){
		return Eigen::Matrix<int,Eigen::Dynamic,1>::Ones(1);
	}
	else if(row == 1){
		return Eigen::Matrix<int,Eigen::Dynamic,1>::Ones(2);
	}
	else{
		Eigen::Matrix<int,Eigen::Dynamic,1> output = Eigen::Matrix<int,Eigen::Dynamic,1>::Ones(row+1);
		Eigen::Matrix<int,Eigen::Dynamic,1> oneLayerDown = pascalTriangle(row - 1);


		for(int i = 0; i < oneLayerDown.size() - 1; i++){
			output(i+1) = oneLayerDown(i+1) + oneLayerDown(i);
		}
		return output;
	}
}


double error_function(double arg){
	double tolerance = std::numeric_limits<double>::epsilon();

	if (std::abs(arg) < 5){ //For small arguments use the McLaurin series

		double result = 0;
		double term = std::numeric_limits<double>::max();
		long long int termNum = 0;
		while(std::abs(term) > tolerance){
			double tmp = (double)(termNum*2+1);
			double sign = -1;
			if (termNum%2 == 0){
				sign = 1;
			}
			double denominator = (double)tmp*(double)factorial(termNum);
			term = sign*std::pow(arg,tmp)/denominator;
			termNum++;
			result += term;
		}
		return result*(double)2/std::sqrt(M_PI);
	}
	else{ 

		return (double)1;

		//With large absolute value arguments compute with A&S approximation

		if (arg < 0){
			return -error_function(-arg);
		}

		double a1 = 0.0705230784;
		double a2 = 0.0422820123;
		double a3 = 0.0092705272;
		double a4 = 0.0001520143;
		double a5 = 0.0002765672;
		double a6 = 0.0000430638;

		double denominator = 1;
		denominator += a1*std::pow(arg,1);
		denominator += a2*std::pow(arg,2);
		denominator += a3*std::pow(arg,3);
		denominator += a4*std::pow(arg,4);
		denominator += a5*std::pow(arg,5);
		denominator += a6*std::pow(arg,6);

		return (double)1 - (double)1/std::pow(denominator,16);

	}

}


//Get a sample from the Poisson distribution with the parameter lambda
unsigned int samplePoissonDistribution(const double& lambda){
	unsigned int element = 0;
	double prob = 0;
	double tolerance = 10e-5;

	double ayn = sampleUniform<double>();

	double expFactor = std::exp(-lambda);
	while(prob < 1){
		double tmp = std::pow(lambda,element)*expFactor/factorial(element);

		prob += tmp;

		if (prob > ayn){
			return element;
		}

		if (tmp < tolerance){
			break;
		}

		element++;
	}

	return element;
}

double inverse_gaussian_cdf_rational_approximation(double t)
{
    // Abramowitz and Stegun formula 26.2.23.
    // The absolute value of the error should be less than 4.5 e-4.
    double c[] = {2.515517, 0.802853, 0.010328};
    double d[] = {1.432788, 0.189269, 0.001308};
    return t - ((c[2]*t + c[1])*t + c[0]) /
                (((d[2]*t + d[1])*t + d[0])*t + 1.0);
}
 
double inv_normcdf_approx(double p)
{
    //if (p <= 0.0 || p >= 1.0)
    //{
    //    std::stringstream os;
    //    os << "Invalid input argument (" << p << "); must be larger than 0 but less than 1.";
    //    throw std::invalid_argument( os.str() );
    //}

	if (p <= std::numeric_limits<double>::epsilon()){
		p = std::numeric_limits<double>::epsilon();
	}

	if (p >= 1.0){
		p = 1.0 - std::numeric_limits<double>::epsilon();
	}
 
    if (p < 0.5)
    {
        // F^-1(p) = - G^-1(p)
        return -inverse_gaussian_cdf_rational_approximation( sqrt(-2.0*log(p)) );
    }
    else
    {
        // F^-1(p) = G^-1(1-p)
        return inverse_gaussian_cdf_rational_approximation( sqrt(-2.0*log(1-p)) );
    }
}

double normcdf_approx(const double& x,const double& mu,const double& sigma)
{
	//Approximation by Abramowitz and Stegun

    // constants
    static const double a1 =  0.254829592;
    static const double a2 = -0.284496736;
    static const double a3 =  1.421413741;
    static const double a4 = -1.453152027;
    static const double a5 =  1.061405429;
    static const double p  =  0.3275911;

	double x_standardized = (x-mu)/sigma;

    // Save the sign of x
    int sign = 1;
    if (x_standardized < 0)
        sign = -1;
    x_standardized = std::abs(x_standardized)/sqrt(2.0);

    // A&S formula 7.1.26
    double t = 1.0/(1.0 + p*x_standardized);
    double y = 1.0 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t*exp(-x_standardized*x_standardized);

	 return 0.5*(1.0 + sign*y);


}

double sample_clipped_gaussian(const double& mean, const double& std, const double& lower_clipping_point, const double& upper_clipping_point){

	if (std <= 0.0){

		if (mean < lower_clipping_point){
			return lower_clipping_point;
		}

		if (mean > upper_clipping_point){
			return upper_clipping_point;
		}

		return mean;
	}

	double u_rand = sampleUniform<double>();
	while (u_rand == 0.0 || u_rand == 1.0){
		u_rand = sampleUniform<double>();
	}
	double standard_lower_clip = (lower_clipping_point - mean) / std;
	double standard_upper_clip = (upper_clipping_point - mean) / std;

	double phi_alpha = normcdf_approx(standard_lower_clip,0,1);
	double phi_beta =  normcdf_approx(standard_upper_clip,0,1);

	double x_standard = inv_normcdf_approx(u_rand * (phi_beta - phi_alpha) + phi_alpha );
	double result = mean+std*x_standard;

	return result;

}


