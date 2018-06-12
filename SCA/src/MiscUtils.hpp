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

#ifndef MISC_UTILS
#define MISC_UTILS

#include <Eigen/Eigen>

#include <exception>
#include <time.h>
#include <vector>
#include <list>
#include <memory>

std::string get_time_string();


template<typename EntryType>
EntryType squareSum(std::vector<EntryType>& vectorToSum) {
	EntryType result = 0;
	for (auto& entry : vectorToSum) {
		result += entry*entry;
	}
	return result;
}

template<typename Matrix1, typename Matrix2>
void copyMatrix(Matrix1& src, Matrix2& dst) {

	dst.resize(src.rows(), src.cols());

	for (int row = 0; row < src.rows(); row++) {
		for (int col = 0; col < src.cols(); col++) {
			dst(row, col) = src(row, col);
		}
	}
}

template<typename Scalar>
//<point> is projected to the line defined by <start> and <direction>. Return value is such that the projection is <start> + <return_value> * <direction>.
Scalar projection_of_point_to_line(const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& point, const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& start, const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& direction) {

	return direction.dot(point - start) / std::abs(direction.norm());

}


template<typename Scalar>
Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> get_principal_axes(const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& matrix) {
	Eigen::JacobiSVD<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> > svd(matrix, Eigen::ComputeFullU | Eigen::ComputeThinV);
	return svd.matrixU();
}


template <typename Scalar>
bool system_is_controllable(const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& A, const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& B, Scalar tolerance = 0.01) {

	typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
	typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;

	assert(A.cols() == A.rows());
	assert(B.rows() == A.cols());

	Matrix controllability_matrix = Matrix::Zero(A.rows(), A.rows()*B.cols());

	Matrix A_power = Matrix::Identity(A.rows(), A.cols());
	Matrix tmp = Matrix::Identity(A.rows(), B.cols());
	for (int i = 0; i < A.rows(); i++) {
		tmp = A_power*B;
		controllability_matrix.middleCols(i*B.cols(), B.cols()) = tmp;
		A_power *= A;
	}

	Eigen::JacobiSVD<Matrix> svd(controllability_matrix, Eigen::ComputeFullU | Eigen::ComputeFullV | Eigen::FullPivHouseholderQRPreconditioner);
	Vector singular_values = svd.singularValues();

	for (int i = 0; i < singular_values.size(); i++) {
		if (singular_values(i) < tolerance) {
			return false;
		}
	}

	return true;

}

template<typename Scalar>
Scalar extremes_to_median_ratio(std::vector<Scalar> input) {
	auto it_begin = input.begin();
	auto it_end = input.end();
	std::sort(it_begin, it_end);

	Scalar median = (Scalar)0;

	//Mode
	if (input.size() % 2 == 0) {
		int half_way = input.size() / 2;
		median = (input[half_way] + input[half_way - 1]) / (Scalar)2;
	}
	else {
		int half_way = input.size() / 2;
		median = input[half_way];
	}

	float ratio = (median - input[input.size() - 1]) / (input[0] - median);

	if (!finiteNumber(ratio)) {
		ratio = 1;
	}

	return ratio;
}

template<typename Scalar>
void zero_denormalized(Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& matrix) {
	for (int row = 0; row < matrix.rows(); row++) {
		for (int column = 0; column < matrix.cols(); column++) {
			if (matrix(row, column) < std::numeric_limits<Scalar>::min()) {
				matrix(row, column) = (Scalar)0;
			}
		}
	}
}

template<typename Scalar>
Scalar frobenius_norm(Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& matrix) {
	Scalar norm = (Scalar)0.0;
	for (int row = 0; row < matrix.rows(); row++) {
		for (int column = 0; column < matrix.cols(); column++) {
			norm += matrix(row, column)*matrix(row, column);
		}
	}
	return std::sqrt(norm);
}

template<typename Scalar>
Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> positive_semidefinite_approximation(Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& matrix) {
	typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
	typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;

	//Ensure that the entries in input are not infinite
	capInfToMax(matrix);
	Eigen::EigenSolver<Matrix> eigen_solver;
	eigen_solver.compute(matrix, /* computeEigenvectors = */ true);

	//The eigen values
	auto eigs = eigen_solver.eigenvalues();

	Vector imaginary_parts = eigs.imag();
	Vector real_parts = eigs.real();

	//Zero negative and complex eigenvalues
	for (int i = 0; i < eigs.size(); i++) {
		if (imaginary_parts(i) != (Scalar)0 || real_parts(i) < (Scalar)0) {
			real_parts(i) = (Scalar)0;
		}
	}

	assert(isFinite(eigs));

	//Eigen vectors
	const Matrix eig_vects = eigen_solver.eigenvectors().real();
	Matrix reconstuction = eig_vects * real_parts.asDiagonal() * eig_vects.inverse();
	assert(isFinite(reconstuction));

	//Return the reconstruction.
	return reconstuction;


}

template<typename DataType>
void remove_duplicates(std::list<std::unique_ptr<DataType> >& list, double(*distance_measure)(DataType&, DataType&)) {

	std::list<std::unique_ptr<DataType> >::iterator iterator1 = list.begin();

	while (iterator1 != list.end()) {

		std::list<std::unique_ptr<DataType> >::iterator iterator2 = iterator1;
		iterator2++;

		bool was_erased = false;

		while (iterator2 != list.end()) {

			//Have to use double dereference because the iterators point to unique pointers.
			double dist = distance_measure(**iterator1, **iterator2);

			if (dist <= 0) {
				iterator1 = list.erase(iterator1);
				was_erased = true;
				break;
			}
			else {
				iterator2++;
			}

		}

		if (was_erased) {
			continue;
		}

		iterator1++;

	}

}


template<typename DataType>
void remove_doubles(std::list<DataType>& list) {

	std::list<DataType >::iterator iterator1 = list.begin();

	while (iterator1 != list.end()) {

		std::list<DataType >::iterator iterator2 = iterator1;
		iterator2++;

		bool was_erased = false;

		while (iterator2 != list.end()) {

			if (*iterator1 == *iterator2) {
				iterator1 = list.erase(iterator1);
				was_erased = true;
				break;
			}
			else {
				iterator2++;
			}

		}

		if (was_erased) {
			continue;
		}

		iterator1++;

	}

}


template<typename Matrix>
//Return true if all the elements in 'x' are finite numbers.
bool isFinite(const Matrix& x)
{
	return  x.array().allFinite();
}

template<typename Scalar>
bool finiteNumber(Scalar num) {
	return (num <= std::numeric_limits<Scalar>::max() && num >= std::numeric_limits<Scalar>::lowest());
}

template<typename Scalar>
bool check_valid_vector(std::vector<Scalar>& check) {


	for (unsigned int j = 0; j < check.size(); j++) {
		if (!finiteNumber(check[j])) {
			return false;
		}
	}

	return true;
}

template<typename Scalar>
bool check_valid_matrix(Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& check) {

	for (int i = 0; i < check.rows(); i++) {
		for (int j = 0; j < check.cols(); j++) {
			if (!finiteNumber(check(i, j))) {
				return false;
			}

		}
	}

	return true;
}

template<typename Scalar>
bool check_valid_matrix(Scalar* check, int numElements) {

	for (int i = 0; i < numElements; i++) {
		if (!finiteNumber(check[i])) {
			return false;

		}
	}

	return true;
}

template<typename Scalar>
bool check_valid_matrix(Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& check) {

	for (int j = 0; j < check.size(); j++) {
		if (check(j) != check(j)) {
			return false;
		}
	}

	return true;
}

class Matrix_has_nonfinite_elements : public std::exception
{
	virtual const char* what() const throw()
	{
		return "Matrix has indeterminate or infinite elements";
	}
};

template <typename T> int sgn(T val) {
	if (val < (T)0) {
		return -(T)1;
	}

	return (T)1;
}

template<typename EntryType>
EntryType sum(std::vector<EntryType>& vectorToSum) {
	EntryType result = 0;
	for (auto& entry : vectorToSum) {
		result += entry;
	}
	return result;
}


//template<typename EntryType>
//void shuffle(std::vector<EntryType>& vectorToShuffle){
//
//
//	//Create a vector full of indices
//	std::vector<unsigned int> indexVectorTmp;
//	indexVectorTmp.reserve(vectorToShuffle.size());
//	for (unsigned int i = 0; i < vectorToShuffle.size(); i++){
//		indexVectorTmp.push_back(i);
//	}
//
//	std::vector<unsigned int> indexVector;
//	indexVector.reserve(vectorToShuffle.size());
//	//Shuffle the indices
//	while(indexVectorTmp.size() > 0){
//		int randIndex = rand() % indexVectorTmp.size();
//		indexVector.push_back(indexVectorTmp[randIndex]);
//		indexVectorTmp.erase(indexVectorTmp.begin()+randIndex);
//	}
//
//	std::vector<EntryType> tmp = vectorToShuffle;
//
//	for (unsigned int i = 0; i < vectorToShuffle.size(); i++){
//		vectorToShuffle[i] = tmp[indexVector[i]];
//	}
//}

template<typename EntryType>
void shuffle(std::vector<EntryType>& vectorToShuffle) {

	unsigned size = vectorToShuffle.size();
	for (unsigned int i = 0; i < size; i++) {
		unsigned rand_idx = rand() % size;

		EntryType tmp = vectorToShuffle[rand_idx];

		vectorToShuffle[rand_idx] = vectorToShuffle[i];
		vectorToShuffle[i] = std::move(tmp);
	}
}

//template<typename DataType>
//void removeDuplicates(std::vector<DataType>& vector){
//
//
//	for(auto mainIterator = vector.begin(); mainIterator != vector.end(); mainIterator++){
//
//		for (auto iterator = mainIterator + 1; iterator != vector.end(); iterator++){
//
//			if (iterator == vector.end()){
//				break;
//			}
//
//			if (*iterator == *mainIterator){
//				iterator = vector.erase(iterator);
//				iterator--;
//			}
//
//		}
//
//	}
//
//}

template<typename DataType>
void removeDuplicates(std::vector<DataType>& vector) {

	std::vector<DataType> vector_with_unique_elements;
	vector_with_unique_elements.reserve(vector.size());

	for (auto mainIterator = vector.begin(); mainIterator != vector.end(); mainIterator++) {

		bool new_data_item = true;

		for (auto iterator = vector_with_unique_elements.begin(); iterator != vector_with_unique_elements.end(); iterator++) {
			if (*iterator == *mainIterator) {
				new_data_item = false;
				break;
			}
		}

		if (new_data_item) {
			vector_with_unique_elements.push_back(*mainIterator);
		}

	}

	vector = vector_with_unique_elements;

}

template<typename integer_type>
//Turns the coordinate of a grid to a linear index verctor.
integer_type subscript_to_index(const std::vector<integer_type>& sub, const std::vector<integer_type>& array_dimensions) {

	assert(array_dimensions.size() == sub.size());

	integer_type return_value = 0;
	integer_type rest_dimensions_size = 1;

	for (integer_type i = 0; i < (integer_type)sub.size(); i++) {

		assert(sub[i] >= 0);
		assert(sub[i] < array_dimensions[i]);

		return_value += rest_dimensions_size*sub[i];

		rest_dimensions_size *= array_dimensions[i];

	}

	return return_value;

}

template<typename integer_type>
//Turns a linear index vector to a coordinate in the grid whose dimensions are given in <array_dimensions>.
std::vector<integer_type> index_to_subscript(integer_type index, const std::vector<integer_type>& array_dimensions) {

	std::vector<integer_type> sub;

	assert(index >= 0);

	integer_type dim = 1;
	for (integer_type i = 0; i < (integer_type)array_dimensions.size(); i++) {
		dim *= array_dimensions[i];
	}

	for (integer_type i = (integer_type)(array_dimensions.size() - 1); i >= 0; i--) {

		dim /= array_dimensions[i];
		sub.push_back(index / dim);
		index %= dim;

	}

	return flip(sub);

}

template<typename data_type>
//Returns the content of <data> in reverse order.
std::vector<data_type> flip(const std::vector<data_type>& data) {

	std::vector<data_type> flipped_vector;
	flipped_vector.reserve(data.size());

	for (int i = data.size() - 1; i >= 0; i--) {

		flipped_vector.push_back(data[i]);

	}

	return flipped_vector;
}

template<typename Scalar>
Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> make_invertable(const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& matrix_to_invert, const Scalar& max_condition_number = (Scalar)1000) {

	Eigen::JacobiSVD<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> > svd(matrix_to_invert, Eigen::ComputeFullU | Eigen::ComputeFullV);

	Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> U = svd.matrixU();
	Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> V = svd.matrixV();
	Eigen::Matrix<Scalar, Eigen::Dynamic, 1> sigma = svd.singularValues();

	Scalar max_sigma = -std::numeric_limits<Scalar>::infinity();
	for (int i = 0; i < sigma.size(); i++) {
		max_sigma = std::max(max_sigma, std::abs(sigma(i)));
	}

	if (max_sigma < std::numeric_limits<Scalar>::epsilon()) {
		max_sigma = std::numeric_limits<Scalar>::epsilon();
		for (int i = 0; i < sigma.size(); i++) {
			sigma(i) = std::numeric_limits<Scalar>::epsilon();
		}
	}

	for (int i = 0; i < sigma.size(); i++) {
		Scalar sign = (Scalar)sgn(sigma(i));
		if (sign == 0) {
			sign = 1;
		}
		sigma(i) = sign*std::max(std::abs(sigma(i)), std::abs(max_sigma / max_condition_number));
	}

	Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> reconstruction = U*(sigma.asDiagonal())*V.transpose();

	return reconstruction;

}


template<typename Scalar>
void ill_conditioned_inverse(const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& matrix_to_invert, Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& inverted_matrix, const Scalar maximum_allowed_condition_number, const Scalar absolute_minimum_eigen_value) {

	Eigen::JacobiSVD<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> > svd(matrix_to_invert, Eigen::ComputeFullU | Eigen::ComputeFullV);

	const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& U = svd.matrixU();
	const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& V = svd.matrixV();
	Eigen::Matrix<Scalar, Eigen::Dynamic, 1> sigma = svd.singularValues().cwiseAbs();

	Scalar max_sigma = sigma.maxCoeff();
	Scalar minimum_allowed_singular_value = sigma.minCoeff();


	if (max_sigma < absolute_minimum_eigen_value) {
		max_sigma = absolute_minimum_eigen_value;
	}

	if (minimum_allowed_singular_value < absolute_minimum_eigen_value) {
		minimum_allowed_singular_value = absolute_minimum_eigen_value;
	}

	if (max_sigma / minimum_allowed_singular_value > maximum_allowed_condition_number) {
		minimum_allowed_singular_value = max_sigma / maximum_allowed_condition_number;
	}

	if (minimum_allowed_singular_value < absolute_minimum_eigen_value) {
		minimum_allowed_singular_value = absolute_minimum_eigen_value;
	}


	for (int i = 0; i < sigma.size(); i++) {

		if (sigma[i] < minimum_allowed_singular_value) {
			sigma[i] = 0;
		}

		sigma[i] = (Scalar)1 / sigma[i];

		if ((sigma[i] - sigma[i]) != (sigma[i] - sigma[i])) {
			sigma[i] = 0;
		}

	}

	inverted_matrix = V*(sigma.asDiagonal())*U.adjoint();
}


template<typename Scalar>
void adaptive_conditioning_psd(Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& matrix, const Scalar maximum_allowed_condition_number, const Scalar absolute_minimum_singular_value) {

	Eigen::JacobiSVD<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> > svd(matrix, Eigen::ComputeFullU | Eigen::ComputeFullV);

	const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& U = svd.matrixU();
	const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& V = svd.matrixV();
	Eigen::Matrix<Scalar, Eigen::Dynamic, 1> sigma = svd.singularValues();

	Scalar max_sigma = sigma.maxCoeff();
	Scalar minimum_allowed_singular_value = sigma.minCoeff();


	if (max_sigma < absolute_minimum_singular_value) {
		max_sigma = absolute_minimum_singular_value;
	}

	if (minimum_allowed_singular_value < absolute_minimum_singular_value) {
		minimum_allowed_singular_value = absolute_minimum_singular_value;
	}

	if (max_sigma / minimum_allowed_singular_value > maximum_allowed_condition_number) {
		minimum_allowed_singular_value = max_sigma / maximum_allowed_condition_number;
	}

	if (minimum_allowed_singular_value < absolute_minimum_singular_value) {
		minimum_allowed_singular_value = absolute_minimum_singular_value;
	}


	for (int i = 0; i < sigma.size(); i++) {

		if (sigma[i] < minimum_allowed_singular_value) {
			sigma[i] = minimum_allowed_singular_value;
		}

	}

	matrix = U*(sigma.asDiagonal())*V.adjoint();
}

template<typename Scalar>
void sqrt_svd(const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& matrix, Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& square_root, const Scalar maximum_allowed_condition_number, const Scalar absolute_minimum_eigen_value) {

	Eigen::JacobiSVD<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> > svd(matrix, Eigen::ComputeFullU | Eigen::ComputeFullV);

	const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& U = svd.matrixU();
	const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& V = svd.matrixV();
	Eigen::Matrix<Scalar, Eigen::Dynamic, 1> sigma = svd.singularValues();

	Scalar max_sigma = sigma.maxCoeff();
	Scalar minimum_allowed_singular_value = sigma.minCoeff();


	if (max_sigma < absolute_minimum_eigen_value) {
		max_sigma = absolute_minimum_eigen_value;
	}

	if (minimum_allowed_singular_value < absolute_minimum_eigen_value) {
		minimum_allowed_singular_value = absolute_minimum_eigen_value;
	}

	if (max_sigma / minimum_allowed_singular_value > maximum_allowed_condition_number) {
		minimum_allowed_singular_value = max_sigma / maximum_allowed_condition_number;
	}

	if (minimum_allowed_singular_value < absolute_minimum_eigen_value) {
		minimum_allowed_singular_value = absolute_minimum_eigen_value;
	}


	for (int i = 0; i < sigma.size(); i++) {

		sigma[0] = std::max((Scalar)0, sigma[0]);

		if (sigma[i] < minimum_allowed_singular_value) {
			sigma[i] = minimum_allowed_singular_value;
		}

		sigma[i] = std::sqrt(sigma[i]);

		if ((sigma[i] - sigma[i]) != (sigma[i] - sigma[i])) {
			sigma[i] = 0;
		}

	}

	square_root = U*(sigma.asDiagonal())*V.adjoint();
}

template<typename Scalar>
Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> make_invertable_positive_definite(const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& matrix_to_invert, const Scalar& max_condition_number = (Scalar)1000) {

	Eigen::EigenSolver<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>> es(matrix_to_invert);

	Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Q = es.eigenvectors().real();
	Eigen::Matrix<Scalar, Eigen::Dynamic, 1> lambda = es.eigenvalues().real();

	Scalar max_lambda = -std::numeric_limits<Scalar>::infinity();
	for (int i = 0; i < lambda.size(); i++) {
		max_lambda = std::max(max_lambda, std::abs(lambda(i)));
	}

	if (max_lambda < std::numeric_limits<Scalar>::epsilon()) {
		max_lambda = std::numeric_limits<Scalar>::epsilon();
		for (int i = 0; i < lambda.size(); i++) {
			lambda(i) = std::numeric_limits<Scalar>::epsilon();
		}
	}

	for (int i = 0; i < lambda.size(); i++) {
		lambda(i) = std::max(std::abs(lambda(i)), std::abs(max_lambda / max_condition_number));
	}

	Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> reconstruction = Q*(lambda.asDiagonal())*Q.transpose();

	return reconstruction;

}

template<typename Scalar>
Scalar log_determinant_positive_definite(const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& matrix_to_invert, const Scalar& max_condition_number = (Scalar)1000) {

	Eigen::EigenSolver<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>> es(matrix_to_invert);

	Eigen::Matrix<Scalar, Eigen::Dynamic, 1> lambda = es.eigenvalues().real();

	Scalar max_lambda = -std::numeric_limits<Scalar>::infinity();
	for (int i = 0; i < lambda.size(); i++) {
		max_lambda = std::max(max_lambda, std::abs(lambda(i)));
	}

	if (max_lambda < std::numeric_limits<Scalar>::epsilon()) {
		max_lambda = std::numeric_limits<Scalar>::epsilon();
		for (int i = 0; i < lambda.size(); i++) {
			lambda(i) = std::numeric_limits<Scalar>::epsilon();
		}
	}

	for (int i = 0; i < lambda.size(); i++) {
		lambda(i) = std::max(std::abs(lambda(i)), std::abs(max_lambda / max_condition_number));
	}

	lambda.array() = lambda.array().log();

	return lambda.sum();

}

template<typename Scalar>
Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> invert(const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& matrix_to_invert, const Scalar& max_condition_number = (Scalar)1000) {

	return make_invertable(matrix_to_invert, max_condition_number).inverse();

}

template<typename Scalar>
Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> invert_positive_definite(const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& matrix_to_invert, const Scalar& max_condition_number = (Scalar)1000) {

	return make_invertable_positive_definite(matrix_to_invert, max_condition_number).inverse();

}

template<typename EntryType>
EntryType mean(std::vector<EntryType>& vector) {
	EntryType mean = 0;
	for (EntryType& element : vector) {
		mean += element;
	}
	mean /= (double)vector.size();
	return mean;
}

//Compute the weighted mean of observations in 'input'
//'input' has the observations in its rows
template <typename Scalar>
Eigen::Matrix<Scalar, Eigen::Dynamic, 1> weightedMean(Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> input, Eigen::Matrix<Scalar, Eigen::Dynamic, 1> weights = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>::Zero(0)) {
	Scalar one = 1;

	if (weights.size() == 0) {
		weights = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>::Ones(input.rows());
	}
	normalizeProbabilities(weights);

	//Weight the ROWS of the input matrix by the normalized weights
	input = (weights.asDiagonal())*input;
	//As the weights sum up to one, the weighted mean is obtained by column sums
	return input.colwise().sum();
}

template<typename Scalar>
bool is_valid_num(Scalar& num) {
	return (num <= std::numeric_limits<Scalar>::max() && num >= std::numeric_limits<Scalar>::lowest());
}

template<typename Scalar>
Scalar nanmean(const std::vector<Scalar>& data) {
	int items = 0;
	Scalar mean = (Scalar)0;
	for (const Scalar& datum : data) {
		if (is_valid_num(datum)) {
			items++;
			mean += datum;
		}
	}
	mean /= (Scalar)items;
	return mean;
}


void repulse(float** sequence1, int dim, int data_points, int repulse_points, float shift_amount);

void swap_closest(float** sequence1, int dim, int data_points, int repulse_points);

std::vector<std::pair<int,float>> indices_of_k_nearest(float** sequence1, int dim, int data_points, int point_idx, int k);
std::vector<std::pair<int, float>> indices_of_k_nearest(float ** sequence1, const Eigen::VectorXf& current, int dim, int data_points, int k);

void project_to_unit_sphere(float** sequence, int dim, int data_points);

//Aligns sequence2 to sequence1
template<typename Scalar>
Scalar dtw_alignment(Scalar** sequence1, Scalar** sequence2, Scalar** matched_sequence, const int dim, const int length1, const int length2) {

	Scalar dtw_distance = (Scalar)0;

	Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> dtw;
	dtw.resize(length1, length2);

	auto dist = [](Scalar* pt1, Scalar* pt2, int dim) {

		Scalar distance = 0;

		for (int i = 0; i < dim; i++) {
			Scalar local_dist = pt1[i] - pt2[i];
			distance += local_dist*local_dist;
		}

		distance = std::sqrt(distance);

		return distance;
	};

	//Computing the alignments
	for (int i = 0; i < length1; i++) {
		for (int j = 0; j < length2; j++) {

			Scalar cost = dist(sequence1[i], sequence2[j], dim);

			if (i == 0 && j == 0) {
				dtw(i, j) = cost;
				continue;
			}

			Scalar prev = std::numeric_limits<Scalar>::infinity();

			if (j > 0) {
				prev = dtw(i, j - 1);
			}

			if (i > 0) {
				prev = std::min(prev, dtw(i - 1, j));
			}

			if (i > 0 && j > 0) {
				prev = std::min(prev, dtw(i - 1, j - 1));
			}

			dtw(i, j) = cost + prev;

		}
	}


	dtw_distance = dtw(length1 - 1, length2 - 1);


	int i = length1 - 1;
	int j = length2 - 1;



	while (i > 0 && j > 0) {

		for (int idx = 0; idx < dim; idx++) {
			matched_sequence[i][idx] = sequence2[j][idx];
		}

		Scalar& match = dtw(i - 1, j - 1);
		Scalar& insertion = dtw(i - 1, j);
		Scalar& deletion = dtw(i, j - 1);

		Scalar min_val = std::min(match, insertion);
		min_val = std::min(min_val, deletion);

		if (match == min_val) {
			i--;
			j--;
		}
		else {

			if (insertion < deletion) {
				i--;
			}

			if (deletion < insertion) {
				j--;
			}

			if (deletion == insertion) {
				if (rand() % 2 == 0) {
					j--;
				}
				else {
					i--;
				}
			}


		}

	}


	return dtw_distance;
}

//If nums is one hot vector, result will be hard max.
void softmax(const float* nums, float* result , int dim);



#endif