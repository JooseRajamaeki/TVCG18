/*

Part of Aalto University Game Tools. See LICENSE.txt for licensing info.

*/

#ifndef GENERIC_DENSITY_FOREST_LINEAR_H
#define GENERIC_DENSITY_FOREST_LINEAR_H




#include <Eigen/Eigen>
#include <memory>
#include <list>
#include <deque>
#include <omp.h>
#include <algorithm>
#include "ProbUtils.hpp"
#include "MathUtils.h"

#include <exception>

class ZeroKeyVectorException : std::exception
{
public:
	const char* what() const throw() // my call to the std exception class function (doesn't nessasarily have to be virtual).
	{
		return "The key vector does not match normal."; // my error message
	}
};



typedef float generic_density_forest_scalar;
typedef Eigen::Matrix<generic_density_forest_scalar, Eigen::Dynamic, Eigen::Dynamic> generic_density_forest_matrix;
typedef Eigen::Matrix<generic_density_forest_scalar, Eigen::Dynamic, 1> generic_density_forest_vector;

template<typename first_type>
static bool second_is_smaller(const std::pair<first_type, generic_density_forest_scalar>& first, const std::pair<first_type, generic_density_forest_scalar>& second) {
	return first.second < second.second;
}

template<typename data_type>
static bool is_in_array(const data_type elements[], int end_idx, const data_type element) {
	for (int i = 0; i < end_idx; i++) {
		if (elements[i] == element) {
			return true;
		}
	}

	return false;
}

enum Covariance_computation_mode
{
	AXIS_ALIGNED, FULL
};

enum Split_Optimization_Mode
{
	soINFO_GAIN, soKMEANS
};


template<typename DataType>
class GenericDensityNode
{

	typedef GenericDensityNode<DataType>* NodePointer;
	typedef std::unique_ptr<GenericDensityNode<DataType> > UniqueNodePointer;
private:

	generic_density_forest_vector left_mean_;
	generic_density_forest_vector right_mean_;

	generic_density_forest_matrix left_cov_;
	generic_density_forest_matrix right_cov_;

public:
	static const int MAX_SAMPLES_FOR_INFO_GAIN_MEASURING = 256;

	generic_density_forest_vector separating_normal_;
	generic_density_forest_scalar separating_bias_;
	std::pair<generic_density_forest_vector, generic_density_forest_matrix> gaussian_model_;

	//Parameters for regressor use.
	generic_density_forest_matrix regressor_mean_;
	generic_density_forest_vector regressand_mean_;
	generic_density_forest_matrix regressand_cov_cholesky_;
	generic_density_forest_matrix regression_;

	Covariance_computation_mode covariance_computation_mode_;

	NodePointer parent_;
	UniqueNodePointer left_child_;
	UniqueNodePointer right_child_;

	Split_Optimization_Mode split_mode;
	std::list<std::unique_ptr<DataType> > data_items_;

	const generic_density_forest_vector* (*get_key_vector_)(const DataType& datum);

	void copy_data(GenericDensityNode<DataType>& other) {
		separating_bias_ = other.separating_bias_;
		separating_normal_ = other.separating_normal_;
		gaussian_model_ = other.gaussian_model_;
		regressor_mean_ = other.regressor_mean_;
		regressand_mean_ = other.regressand_mean_;
		regressand_cov_cholesky_ = other.regressand_cov_cholesky_;
		regression_ = other.regression_;
		get_key_vector_ = other.get_key_vector_;
		covariance_computation_mode_ = other.covariance_computation_mode_;
		split_mode = other.split_mode;

		std::list<std::unique_ptr<DataType> >::iterator iter = other.data_items_.begin();

		while (iter != other.data_items_.end()) {
			data_items_.push_back(std::unique_ptr<DataType>(new DataType));
			*(data_items_.back()) = **iter;
			iter++;
		}

	}

	GenericDensityNode() {
		parent_ = nullptr;
		left_child_ = UniqueNodePointer(nullptr);
		right_child_ = UniqueNodePointer(nullptr);
		gaussian_model_ = std::make_pair<generic_density_forest_vector, generic_density_forest_matrix>(generic_density_forest_vector::Zero(0), generic_density_forest_matrix::Zero(0, 0));
		separating_bias_ = std::numeric_limits<generic_density_forest_scalar>::quiet_NaN();
		separating_normal_ = generic_density_forest_vector::Zero(0);
		regressor_mean_ = generic_density_forest_vector::Zero(0);
		regressand_mean_ = generic_density_forest_vector::Zero(0);
		regressand_cov_cholesky_ = generic_density_forest_matrix::Zero(0, 0);
		regression_ = generic_density_forest_matrix::Zero(0, 0);
		covariance_computation_mode_ = AXIS_ALIGNED;
		split_mode = soINFO_GAIN;
		get_key_vector_ = nullptr;
	}

	GenericDensityNode(GenericDensityNode<DataType>& other) {
		parent_ = nullptr;
		left_child_ = UniqueNodePointer(nullptr);
		right_child_ = UniqueNodePointer(nullptr);
		copy_data(other);
	}

	GenericDensityNode<DataType> operator=(GenericDensityNode<DataType>& other) {
		parent_ = nullptr;
		left_child_ = UniqueNodePointer(nullptr);
		right_child_ = UniqueNodePointer(nullptr);
		copy_data(other);

		return *this;
	}

	void build_empty_tree(int depth, int data_dimension, const generic_density_forest_vector* (*key_vector_function)(const DataType&)) {

		get_key_vector_ = key_vector_function;
		separating_bias_ *= 0;
		separating_normal_ = generic_density_forest_vector::Zero(data_dimension);
		left_mean_ = generic_density_forest_vector::Zero(data_dimension);
		right_mean_ = generic_density_forest_vector::Zero(data_dimension);

		if (covariance_computation_mode_ == Covariance_computation_mode::FULL) {
			left_cov_ = generic_density_forest_matrix::Zero(data_dimension, data_dimension);
			right_cov_ = generic_density_forest_matrix::Zero(data_dimension, data_dimension);
		}

		if (covariance_computation_mode_ == Covariance_computation_mode::AXIS_ALIGNED) {
			left_cov_ = generic_density_forest_matrix::Zero(data_dimension, 1);
			right_cov_ = generic_density_forest_matrix::Zero(data_dimension, 1);
		}

		gaussian_model_.first = generic_density_forest_vector::Zero(data_dimension);
		gaussian_model_.second = generic_density_forest_matrix::Zero(data_dimension, data_dimension);

		if (depth < 0) {
			return;
		}

		UniqueNodePointer tmp = UniqueNodePointer(new GenericDensityNode<DataType>());
		tmp->get_key_vector_ = get_key_vector_;
		setLeftChild(tmp);
		left_child_->build_empty_tree(depth - 1, data_dimension, key_vector_function);
		tmp = UniqueNodePointer(new GenericDensityNode<DataType>());
		tmp->get_key_vector_ = get_key_vector_;
		setRightChild(tmp);
		right_child_->build_empty_tree(depth - 1, data_dimension, key_vector_function);

	}

	void setLeftChild(UniqueNodePointer& newLeftChild) {

		if (newLeftChild.get()) {
			if (left_child_.get()) {
				left_child_->parent_ = nullptr;
			}
			left_child_ = std::move(newLeftChild);
			left_child_->parent_ = this;
		}
		else {
			left_child_ = UniqueNodePointer(nullptr);
		}

	}

	void setRightChild(UniqueNodePointer& newRightChild) {

		if (newRightChild.get()) {
			if (right_child_.get()) {
				right_child_->parent_ = nullptr;
			}
			right_child_ = std::move(newRightChild);
			right_child_->parent_ = this;
		}
		else {
			right_child_ = UniqueNodePointer(nullptr);
		}
	}

	bool is_empty_leaf(void) {
		if (!hasChildren()) {

			if (data_items_.size() == 0) {
				return true;
			}

		}
		return false;
	}

	NodePointer get_leaf(const generic_density_forest_vector &key_vector) {
		NodePointer node = this;
		NodePointer prev_node = node;
		while (node) {
			prev_node = node;

			if (node->left_or_right(key_vector)) {
				node = node->right_child_.get();
			}
			else {
				node = node->left_child_.get();
			}

		}
		return prev_node;
	}

	NodePointer get_leaf(const DataType& datum) {
		const generic_density_forest_vector &key_vector = *(get_key_vector_(datum));
		return get_leaf(key_vector);
	}

	void form_regression_parameters(int regressor_dim) {

		int regressand_dim = gaussian_model_.first.size() - regressor_dim;

		assert(regressand_dim > 0);

		regressor_mean_ = gaussian_model_.first.tail(regressor_dim);
		regressand_mean_ = gaussian_model_.first.head(regressand_dim);

		generic_density_forest_matrix cov_xy = gaussian_model_.second.topRightCorner(regressand_dim, regressor_dim);
		generic_density_forest_matrix cov_yy = gaussian_model_.second.bottomRightCorner(regressor_dim, regressor_dim);
		generic_density_forest_matrix cov_yy_inv = invert_positive_definite(cov_yy);

		regression_ = cov_xy*cov_yy_inv;

		regressand_cov_cholesky_ = gaussian_model_.second.topLeftCorner(regressand_dim, regressand_dim) - cov_xy*cov_yy_inv*cov_xy.transpose();
		regressand_cov_cholesky_ = make_invertable_positive_definite(regressand_cov_cholesky_).llt().matrixL();

	}

	bool hasChildren() {
		if (right_child_.get() && left_child_.get()) {
			return true;
		}
		else {
			return false;
		}
	}

	generic_density_forest_scalar information_gain(std::list<DataType* >& left_data, std::list<DataType* >& right_data) {

		if (left_data.size() == 0 || right_data.size() == 0) {
			return -std::numeric_limits<generic_density_forest_scalar>::infinity();
		}

		const generic_density_forest_vector& tmp_key_vector = *get_key_vector_(**left_data.begin());
		int data_dim = tmp_key_vector.size();

		if (left_cov_.rows() != data_dim || right_cov_.rows() != data_dim) {

			if (covariance_computation_mode_ == Covariance_computation_mode::FULL) {
				left_cov_ = generic_density_forest_matrix::Zero(data_dim, data_dim);
				right_cov_ = generic_density_forest_matrix::Zero(data_dim, data_dim);
			}

			if (covariance_computation_mode_ == Covariance_computation_mode::AXIS_ALIGNED) {
				left_cov_ = generic_density_forest_matrix::Zero(data_dim, 1);
				right_cov_ = generic_density_forest_matrix::Zero(data_dim, 1);
			}
		}
		else {
			left_cov_ *= 0;
			right_cov_ *= 0;
		}

		if (left_mean_.size() != data_dim || right_mean_.size() != data_dim) {
			left_mean_ = generic_density_forest_vector::Zero(data_dim);
			right_mean_ = generic_density_forest_vector::Zero(data_dim);
		}
		else {
			left_mean_ *= 0;
			right_mean_ *= 0;
		}

		int left_size = 0;
		int right_size = 0;

		for (DataType* datum : left_data) {
			const generic_density_forest_vector& sample_key_vector = *get_key_vector_(*datum);
			left_mean_ += sample_key_vector;
			left_size++;
		}
		left_mean_ /= (generic_density_forest_scalar)left_size;

		for (DataType* datum : right_data) {
			const generic_density_forest_vector& sample_key_vector = *get_key_vector_(*datum);
			right_mean_ += sample_key_vector;
			right_size++;
		}
		right_mean_ /= (generic_density_forest_scalar)right_size;

		generic_density_forest_vector diff;

		for (DataType* datum : left_data) {
			const generic_density_forest_vector& sample_key_vector = *get_key_vector_(*datum);
			diff = sample_key_vector - left_mean_;
			if (covariance_computation_mode_ == Covariance_computation_mode::FULL) {
				left_cov_ += diff*diff.transpose();
			}

			if (covariance_computation_mode_ == Covariance_computation_mode::AXIS_ALIGNED) {
				left_cov_ += diff.cwiseAbs2();
			}
		}
		left_cov_ /= (generic_density_forest_scalar)left_size;

		for (DataType* datum : right_data) {
			const generic_density_forest_vector& sample_key_vector = *get_key_vector_(*datum);
			diff = sample_key_vector - right_mean_;

			if (covariance_computation_mode_ == Covariance_computation_mode::FULL) {
				right_cov_ += diff*diff.transpose();
			}

			if (covariance_computation_mode_ == Covariance_computation_mode::AXIS_ALIGNED) {
				right_cov_ += diff.cwiseAbs2();
			}
		}
		right_cov_ /= (generic_density_forest_scalar)right_size;

		int joint_size = left_size + right_size;

		generic_density_forest_scalar info_gain = (generic_density_forest_scalar)0;
		if (covariance_computation_mode_ == Covariance_computation_mode::FULL) {
			info_gain -= (generic_density_forest_scalar)left_size / (generic_density_forest_scalar)joint_size*log_determinant_positive_definite(left_cov_);
			assert(finiteNumber(info_gain));
			info_gain -= (generic_density_forest_scalar)right_size / (generic_density_forest_scalar)joint_size*log_determinant_positive_definite(right_cov_);
			assert(finiteNumber(info_gain));
		}

		if (covariance_computation_mode_ == Covariance_computation_mode::AXIS_ALIGNED) {
			for (int i = 0; i < left_cov_.size(); i++) {
				left_cov_(i) = std::max(std::numeric_limits<generic_density_forest_scalar>::min(), left_cov_(i));
			}
			info_gain -= (generic_density_forest_scalar)left_size / (generic_density_forest_scalar)joint_size*(left_cov_.array().log().sum());
			//assert(finiteNumber(info_gain));
			for (int i = 0; i < right_cov_.size(); i++) {
				right_cov_(i) = std::max(std::numeric_limits<generic_density_forest_scalar>::min(), right_cov_(i));
			}
			info_gain -= (generic_density_forest_scalar)right_size / (generic_density_forest_scalar)joint_size*(right_cov_.array().log().sum());
			//assert(finiteNumber(info_gain));
		}

		return info_gain;

	}




	//generic_density_forest_scalar information_gain_maximum_separation(std::vector<DataType*> left_data, std::vector<DataType*> right_data){

	//	generic_density_forest_scalar margin = std::numeric_limits<generic_density_forest_scalar>::infinity();
	//	generic_density_forest_scalar current_margin = std::numeric_limits<generic_density_forest_scalar>::infinity();

	//	for (DataType* datum : left_data){
	//		current_margin = std::abs(separating_bias_ + separating_normal_.dot(*get_key_vector_(*datum)));
	//		margin = std::min(margin,current_margin);
	//	}

	//	for (DataType* datum : right_data){
	//		current_margin = std::abs(separating_bias_ + separating_normal_.dot(*get_key_vector_(*datum)));
	//		margin = std::min(margin,current_margin);
	//	}

	//	return margin*left_data.size()*right_data.size();

	//}


	////Obsolete
	//std::vector<std::pair<DataType*,generic_density_forest_scalar> > project_data(std::vector<DataType*>& data, generic_density_forest_vector& start, generic_density_forest_vector& direction){
	//	std::vector<std::pair<DataType*,generic_density_forest_scalar> > data_with_projection;
	//	data_with_projection.reserve(data.size());
	//	for (int i = 0; i < (int)data.size(); i++){
	//		const generic_density_forest_vector* location = get_key_vector_(*(data[i]));
	//		generic_density_forest_scalar alpha = projection_of_point_to_line(*(location),start,direction);
	//		data_with_projection.push_back(std::make_pair(data[i],alpha));
	//	}
	//	return data_with_projection;
	//}

	////Obsolete
	////It is assumed that the <gaussian_model_> is computed.
	//void get_separating_hyper_plane_perpendicular_to_first_principal_axis(std::vector<DataType*>& data){

	//	int data_dim = (*get_key_vector_(*data[0])).size();

	//	if (data.size() < 2){
	//		separating_normal_ = generic_density_forest_vector::Zero(data_dim);
	//		separating_bias_ = 0;
	//		return;
	//	}

	//	separating_normal_ = get_principal_axes(gaussian_model_.second).col(0);

	//	left_cov_ = generic_density_forest_matrix::Zero(data_dim,1);
	//	right_cov_ = generic_density_forest_matrix::Zero(data_dim,1);
	//	left_mean_ = generic_density_forest_vector::Zero(data_dim);
	//	right_mean_ = generic_density_forest_vector::Zero(data_dim);

	//	//Project the data to the first principal axis of the data.
	//	std::vector<std::pair<DataType*,generic_density_forest_scalar> > data_with_projection = project_data(data,gaussian_model_.first,separating_normal_);

	//	//Sort
	//	std::sort(data_with_projection.begin(),data_with_projection.end(),second_is_smaller<DataType*>);


	//	int left_size = 0;
	//	int right_size = data.size();
	//	generic_density_forest_scalar joint_size = (generic_density_forest_scalar)(left_size + right_size);
	//	right_mean_ = gaussian_model_.first;
	//	right_cov_ = gaussian_model_.second.diagonal();

	//	generic_density_forest_scalar max_info_gain = -std::numeric_limits<generic_density_forest_scalar>::infinity();
	//	generic_density_forest_scalar current_info_gain = -std::numeric_limits<generic_density_forest_scalar>::infinity();
	//	int max_gain_index = 0; // The index of the first element that should be left on the 

	//	generic_density_forest_vector left_square_sum = left_mean_*0;
	//	right_mean_ *= (generic_density_forest_scalar)right_size;
	//	generic_density_forest_vector right_square_sum = right_mean_*0;
	//	for (int i = 0; i < (int)data.size();i++){
	//		right_square_sum += get_key_vector_(*data[i])->cwiseAbs2();
	//	}

	//	//NB! left_mean_ and right_mean_ are actually used to store the sum instead of mean in this loop!!!
	//	for (int i = 0; i < (int)data_with_projection.size()-1; i++){
	//		current_info_gain = 0;

	//		const generic_density_forest_vector* location = get_key_vector_(*(data_with_projection[i].first));
	//		left_square_sum += location->cwiseAbs2();
	//		left_mean_ += *location;
	//		right_square_sum -= location->cwiseAbs2();
	//		right_mean_ -= *location;

	//		left_size++;
	//		right_size--;

	//		left_cov_ = left_square_sum - (left_mean_.cwiseAbs2())/(generic_density_forest_scalar)left_size;
	//		left_cov_ /= (generic_density_forest_scalar)left_size;
	//		right_cov_ = right_square_sum - (right_mean_.cwiseAbs2())/(generic_density_forest_scalar)right_size;
	//		right_cov_ /= (generic_density_forest_scalar)right_size;

	//		for (int i = 0; i < left_cov_.size(); i++){
	//			left_cov_(i) = std::max(std::numeric_limits<generic_density_forest_scalar>::min(),left_cov_(i));
	//		}
	//		current_info_gain -= (generic_density_forest_scalar)left_size/(generic_density_forest_scalar)joint_size*(left_cov_.array().log().sum());
	//		for (int i = 0; i < right_cov_.size(); i++){
	//			right_cov_(i) = std::max(std::numeric_limits<generic_density_forest_scalar>::min(),right_cov_(i));
	//		}
	//		current_info_gain -= (generic_density_forest_scalar)right_size/(generic_density_forest_scalar)joint_size*(right_cov_.array().log().sum());

	//		if (current_info_gain > max_info_gain){
	//			max_info_gain = current_info_gain;
	//			max_gain_index = i;
	//		}

	//	}

	//	const generic_density_forest_vector* location1 = get_key_vector_(*(data_with_projection[max_gain_index].first));
	//	const generic_density_forest_vector* location2 = get_key_vector_(*(data_with_projection[max_gain_index+1].first));
	//	generic_density_forest_scalar bias1 = -location1->dot(separating_normal_);
	//	generic_density_forest_scalar bias2 = -location2->dot(separating_normal_);

	//	separating_bias_ = bias1 + (bias2-bias1)*sampleUniform<generic_density_forest_scalar>();


	//}

	std::vector<DataType*> compute_gaussian_models(void) {

		std::vector<DataType*> data;

		if (!hasChildren()) {
			for (const std::unique_ptr<DataType>& tmp : data_items_) {
				data.push_back(tmp.get());
			}
			assert(data.size() > 0);
		}
		else {
			std::vector<DataType*> tmp_data = left_child_->compute_gaussian_models();
			for (DataType* datum : tmp_data) {
				data.push_back(datum);
			}
			assert(data.size() > 0);

			tmp_data = right_child_->compute_gaussian_models();
			for (DataType* datum : tmp_data) {
				data.push_back(datum);
			}
			assert(data.size() > 0);
		}

		assert(data.size() > 0);

		//Build a gaussian model of all the incoming data
		const generic_density_forest_vector& tmp_key_vector = *get_key_vector_(*(data[0]));
		gaussian_model_.first = tmp_key_vector * 0;
		gaussian_model_.second = generic_density_forest_matrix::Zero(gaussian_model_.first.size(), gaussian_model_.first.size());

		for (size_t i = 0; i < data.size(); i++) {
			const generic_density_forest_vector& sample_key_vector = *get_key_vector_(*(data[i]));
			gaussian_model_.first += sample_key_vector;
		}
		gaussian_model_.first /= (generic_density_forest_scalar)data.size();

		generic_density_forest_vector diff;
		for (size_t i = 0; i < data.size(); i++) {
			const generic_density_forest_vector& sample_key_vector = *get_key_vector_(*(data[i]));
			diff = sample_key_vector - gaussian_model_.first;
			gaussian_model_.second += diff*diff.transpose();
		}
		gaussian_model_.second /= (generic_density_forest_scalar)data.size();

		return data;

	}


	void build_tree(int tries, int minimum_data_in_leaf, std::list<std::unique_ptr<DataType> > data) {

		data_items_.clear();

		UniqueNodePointer tmp = UniqueNodePointer(nullptr);
		setLeftChild(tmp);
		tmp = UniqueNodePointer(nullptr);
		setRightChild(tmp);


		//If the split breaks too few data points away, this branch is ready.
		if (data.size() <= (unsigned)minimum_data_in_leaf) {

			data_items_ = std::move(data);

			return;

		}

		if (split_mode == soKMEANS)
		{
			//split plane through kmeans with 2 clusters and downsampling of data
			const generic_density_forest_vector& vect_tmp = *get_key_vector_(**data.begin());
			int keyDim = vect_tmp.size();
			generic_density_forest_vector mean[2], sum[2];
			float w[2];
			for (int i = 0; i < 2; i++)
			{
				mean[i] = generic_density_forest_vector(keyDim);
				sum[i] = generic_density_forest_vector(keyDim);
				mean[i].setZero();
				sum[i].setZero();
				w[i] = 0;
			}
			int nSamples = std::min(tries, (int)data.size());

			int counter = 0;
			//first init clusters randomly
			for (std::unique_ptr<DataType>& datum : data)
			{
				const generic_density_forest_vector &key = *get_key_vector_(*datum);
				int clusterIdx;
				if (w[0] == 0)
					clusterIdx = 0;
				else if (w[1] == 0)
					clusterIdx = 1;
				else
					clusterIdx = AaltoGames::rand01();
				sum[clusterIdx] += key;
				w[clusterIdx]++;
				counter++;
				if (counter == nSamples) {
					break;
				}
			}
			for (int i = 0; i < 2; i++)
			{
				if (w[i] != 0)
					mean[i] = sum[i] / w[i];
				sum[i].setZero();
				w[i] = 0;
			}

			//now do the k-means iteration
			const int kMeansIter = 3;
			for (int iter = 0; iter < kMeansIter; iter++)
			{
				counter = 0;
				for (std::unique_ptr<DataType>& datum : data)
				{
					const generic_density_forest_vector &key = *get_key_vector_(*datum);
					int clusterIdx;
					float dist0 = (key - mean[0]).norm();
					float dist1 = (key - mean[1]).norm();
					if (dist0 < dist1)
						clusterIdx = 0;
					else if (dist1 < dist0)
						clusterIdx = 1;
					else
					{
						if (w[0] == 0)
							clusterIdx = 0;
						else if (w[1] == 0)
							clusterIdx = 1;
						else
							clusterIdx = AaltoGames::rand01();
					}
					sum[clusterIdx] += key;
					w[clusterIdx]++;
					counter++;
					if (counter == nSamples) {
						break;
					}
				}
				for (int i = 0; i < 2; i++)
				{
					if (w[i] != 0)
						mean[i] = sum[i] / w[i];
					sum[i].setZero();
					w[i] = 0;
				}
			}
			std::pair<generic_density_forest_scalar, generic_density_forest_vector> hyper_plane;
			hyper_plane.second = (mean[1] - mean[0]).normalized();
			separating_normal_ = hyper_plane.second;
			hyper_plane.first = -separating_normal_.dot(((generic_density_forest_scalar)0.5)*(mean[0] + mean[1]));
			separating_bias_ = hyper_plane.first;
		}
		else
		{
			std::pair<generic_density_forest_scalar, generic_density_forest_vector> hyper_plane = get_separating_hyperplane(data);
			separating_bias_ = hyper_plane.first;
			separating_normal_ = hyper_plane.second;

			std::list<DataType* > left_samples;
			std::list<DataType* > right_samples;

			if (data.size() > MAX_SAMPLES_FOR_INFO_GAIN_MEASURING)
				divide_data_sampled(data, left_samples, right_samples, MAX_SAMPLES_FOR_INFO_GAIN_MEASURING);
			else
				divide_data_info(data, left_samples, right_samples);

			generic_density_forest_scalar max_information_gain = information_gain(left_samples, right_samples);
			//generic_density_forest_scalar max_information_gain = information_gain_maximum_separation(separated_data.first,separated_data.second);
			for (int i = 0; i < tries; i++) {

				std::pair<generic_density_forest_scalar, generic_density_forest_vector> tmp_plane = get_separating_hyperplane(data);
				separating_bias_ = tmp_plane.first;
				separating_normal_ = tmp_plane.second;
				if (data.size() > MAX_SAMPLES_FOR_INFO_GAIN_MEASURING)
					divide_data_sampled(data, left_samples, right_samples, MAX_SAMPLES_FOR_INFO_GAIN_MEASURING);
				else
					divide_data_info(data, left_samples, right_samples);

				generic_density_forest_scalar tmp_information_gain = information_gain(left_samples, right_samples);
				//generic_density_forest_scalar tmp_information_gain = information_gain_maximum_separation(separated_data.first,separated_data.second);

				if (tmp_information_gain > max_information_gain) {
					max_information_gain = tmp_information_gain;
					hyper_plane = tmp_plane;
				}

			}

			separating_bias_ = hyper_plane.first;
			separating_normal_ = hyper_plane.second;

			//std::pair<std::vector<DataType*>,std::vector<DataType*> > separated_data;
			//get_separating_hyper_plane_perpendicular_to_first_principal_axis(data);

		}

		std::list<std::unique_ptr<DataType> > left_data;
		std::list<std::unique_ptr<DataType> > right_data;
		divide_data(data, left_data, right_data);


		//If the split breaks too few data points away, this branch is ready.
		if (left_data.size() == 0 || right_data.size() == 0) {

			data_items_.clear();
			for (std::unique_ptr<DataType>& datum : left_data) {
				std::unique_ptr<DataType> tmp_ptr = std::unique_ptr<DataType>(nullptr);
				datum.swap(tmp_ptr);
				data_items_.push_back(std::move(tmp_ptr));
			}

			for (std::unique_ptr<DataType>& datum : right_data) {
				std::unique_ptr<DataType> tmp_ptr = std::unique_ptr<DataType>(nullptr);
				datum.swap(tmp_ptr);
				data_items_.push_back(std::move(tmp_ptr));
			}

			return;

		}


		tmp = UniqueNodePointer(new GenericDensityNode<DataType>());
		tmp->get_key_vector_ = get_key_vector_;
		setLeftChild(tmp);
		left_child_->build_tree(tries, minimum_data_in_leaf, std::move(left_data));

		tmp = UniqueNodePointer(new GenericDensityNode<DataType>());
		tmp->get_key_vector_ = get_key_vector_;
		setRightChild(tmp);
		right_child_->build_tree(tries, minimum_data_in_leaf, std::move(right_data));


	}

	void add_sample(int tries, int minimum_data_in_leaf, int maximum_data_in_leaf, std::unique_ptr<DataType> datum) {



		data_items_.push_back(std::move(datum));


		if ((int)data_items_.size() > maximum_data_in_leaf) {
			std::list<std::unique_ptr<DataType> > new_data_items = std::move(data_items_);
			data_items_.clear();
			build_tree(tries, minimum_data_in_leaf, std::move(new_data_items));
		}




	}

	////Obsolete
	//void fill_tree(int tries, int minimum_data_in_leaf, std::vector<DataType*>& data){

	//	data_items_.clear();

	//	//Build a gaussian model of all the incoming data
	//	gaussian_model_.first = *get_key_vector_(*(data[0])) * 0;
	//	gaussian_model_.second = generic_density_forest_matrix::Zero(gaussian_model_.first.size(),gaussian_model_.first.size());

	//	for (size_t i = 0; i < data.size(); i++){
	//		gaussian_model_.first += *get_key_vector_(*(data[i]));
	//	}
	//	gaussian_model_.first /= (generic_density_forest_scalar)data.size();

	//	generic_density_forest_vector diff;
	//	for (size_t i = 0; i < data.size(); i++){
	//		diff = *get_key_vector_(*(data[i])) - gaussian_model_.first;
	//		gaussian_model_.second += diff*diff.transpose();
	//	}
	//	gaussian_model_.second /= (generic_density_forest_scalar)data.size();




	//	//If the split breaks too few data points away, this branch is ready.
	//	if (data.size() < (unsigned)minimum_data_in_leaf){

	//		left_child_.reset();
	//		right_child_.reset();

	//		data_items_.clear();
	//		data_item_pointers_.clear();
	//		if (data_storage_mode_ == Storage_mode::COPY){
	//			data_items_.reserve(data.size());
	//			for (size_t i = 0; i < data.size(); i++){
	//				data_items_.push_back(std::move(*data[i]));
	//			}
	//		}

	//		if (data_storage_mode_ == Storage_mode::POINTER){
	//			data_item_pointers_.reserve(data.size());
	//			for (size_t i = 0; i < data.size(); i++){
	//				data_item_pointers_.push_back(data[i]);
	//			}
	//		}

	//		return;

	//	}


	//	std::pair<generic_density_forest_scalar,generic_density_forest_vector> hyper_plane = get_separating_hyperplane(data);
	//	separating_bias_ = hyper_plane.first;
	//	separating_normal_ = hyper_plane.second;

	//	std::pair<std::vector<DataType*>,std::vector<DataType*> > separated_data;
	//	divide_data(data,separated_data.first,separated_data.second);

	//	generic_density_forest_scalar max_information_gain = information_gain(separated_data.first,separated_data.second);
	//	//generic_density_forest_scalar max_information_gain = information_gain_maximum_separation(separated_data.first,separated_data.second);
	//	for (int i = 0; i < tries; i++){

	//		std::pair<generic_density_forest_scalar,generic_density_forest_vector> tmp_plane = get_separating_hyperplane(data);
	//		separating_bias_ = tmp_plane.first;
	//		separating_normal_ = tmp_plane.second;

	//		divide_data(data,separated_data.first,separated_data.second);

	//		generic_density_forest_scalar tmp_information_gain = information_gain(separated_data.first,separated_data.second);
	//		//generic_density_forest_scalar tmp_information_gain = information_gain_maximum_separation(separated_data.first,separated_data.second);

	//		if (tmp_information_gain > max_information_gain){
	//			max_information_gain = tmp_information_gain;
	//			hyper_plane = tmp_plane;
	//		}

	//	}

	//	separating_bias_ = hyper_plane.first;
	//	separating_normal_ = hyper_plane.second;

	//	divide_data(data,separated_data.first,separated_data.second);

	//	left_child_->fill_tree(tries,minimum_data_in_leaf,separated_data.first);
	//	right_child_->fill_tree(tries,minimum_data_in_leaf,separated_data.second);


	//}

	std::pair<generic_density_forest_scalar, generic_density_forest_vector> get_separating_hyperplane(const std::list<std::unique_ptr<DataType> >& data) {

		if (data.size() < 1) {
			throw ZeroKeyVectorException();
			return std::make_pair(0, generic_density_forest_vector::Zero(0));
		}

		const generic_density_forest_vector& tmp_key_vect = *get_key_vector_(**data.begin());
		generic_density_forest_vector normal = generic_density_forest_vector::Random(tmp_key_vect.size());
		normal.normalize();

		int tmp_idx_1 = rand() % data.size();
		int tmp_idx_2 = rand() % data.size();
		while (tmp_idx_2 == tmp_idx_1) {
			tmp_idx_2 = rand() % data.size();
		}

		DataType* element1 = nullptr;
		DataType* element2 = nullptr;

		//if (data.size() <= 1){
		//	std::cout << "Too few elements.";
		//}

		int counter = 0;
		for (const std::unique_ptr<DataType>& datum : data) {

			if (counter == tmp_idx_1) {
				element1 = datum.get();
			}

			if (counter == tmp_idx_2) {
				element2 = datum.get();
			}

			if (element1 && element2) {
				break;
			}
			counter++;
		}

		//generic_density_forest_vector first_random_element = -*get_key_vector_(*(data[tmp_idx_1]));
		//generic_density_forest_vector second_random_element = -*get_key_vector_(*(data[tmp_idx_2]));

		//std::cout << element1 << " " << element2 << std::endl;

		const generic_density_forest_vector key_vect1 = *get_key_vector_(*element1);
		const generic_density_forest_vector key_vect2 = *get_key_vector_(*element2);
		generic_density_forest_scalar bias_1 = normal.dot(-key_vect1);
		generic_density_forest_scalar bias_2 = normal.dot(-key_vect2);

		generic_density_forest_scalar bias = bias_1 + (bias_2 - bias_1)*sampleUniform<generic_density_forest_scalar>();

		return std::make_pair(bias, normal);

	}

	//True signals right, false signals left.
	bool left_or_right(const generic_density_forest_vector& datum) {

		generic_density_forest_scalar decision = 0;

		if (separating_normal_.size() == 0) {
			if (AaltoGames::rand01() == 0) {
				return false;
			}
			else {
				return true;
			}
		}

		if (datum.size() != separating_normal_.size()) {
			throw ZeroKeyVectorException();
		}

		decision = separating_normal_.dot(datum) + separating_bias_;

		if (decision > 0) {
			return true;
		}
		else if (decision < 0) {
			return false;
		}
		else {
			if (AaltoGames::rand01() == 0) {
				return false;
			}
			else {
				return true;
			}
		}



	}

	//The data will be split to left and right data, which are stored to <left_data> and <right_data>.
	void divide_data_info(std::list<std::unique_ptr<DataType> >& data, std::list<DataType* >& left_data, std::list<DataType* >& right_data) {

		left_data.clear();;
		right_data.clear();

		for (const std::unique_ptr<DataType>& datum : data) {

			DataType* datum_ptr = datum.get();
			const generic_density_forest_vector& key_vector = *get_key_vector_(*datum_ptr);

			if (left_or_right(key_vector)) {
				right_data.push_back(datum_ptr);
			}
			else {
				left_data.push_back(datum_ptr);
			}
		}

	}

	//The data will be split to left and right data, which are stored to <left_data> and <right_data>.
	void divide_data(std::list<std::unique_ptr<DataType> >& data, std::list<std::unique_ptr<DataType> >& left_data, std::list<std::unique_ptr<DataType> >& right_data) {

		left_data.clear();;
		right_data.clear();

		while (data.size() > 0) {
			std::unique_ptr<DataType> datum = std::unique_ptr<DataType>(nullptr);
			datum.swap(data.back());
			data.pop_back();

			const generic_density_forest_vector& key_vect = *get_key_vector_(*datum);

			if (left_or_right(key_vect)) {
				right_data.push_back(std::move(datum));
			}
			else {
				left_data.push_back(std::move(datum));
			}
		}

		data.clear();

	}

	//The data will be split to left and right data, which are stored to <left_data> and <right_data>.
	void divide_data_sampled(std::list<std::unique_ptr<DataType> >& data, std::list<DataType* >& left_data, std::list<DataType* >& right_data, int nSamples) {

		left_data.clear();;
		right_data.clear();

		if (data.size() < 1) {
			return;
		}

		while ((int)(right_data.size() + left_data.size()) < nSamples) {

			int idx = AaltoGames::randInt(0, (int)data.size() - 1);

			std::list<std::unique_ptr<DataType> >::iterator iter = data.begin();

			int counter = 0;
			while (counter < idx && iter != data.end()) {
				iter++;
				counter++;
			}

			if (iter == data.end()) {
				continue;
			}

			std::unique_ptr<DataType>& datum = *iter;

			//try{
			const generic_density_forest_vector& key_vect = *get_key_vector_(*datum);

			if (left_or_right(key_vect)) {
				right_data.push_back(datum.get());
			}
			else {
				left_data.push_back(datum.get());
			}
			//}
			//catch (std::exception e){
			//	std::cout << data.size() << " " << left_data.size() << " " << right_data.size() << std::endl;
			//	std::cout << counter << std::endl;
			//	std::cout << datum << std::endl;
			//	std::cout << datum.get()->state_ << std::endl;
			//	std::cout << datum.get()->control_ << std::endl;
			//	std::cout << datum.get()->future_state_ << std::endl;
			//	std::cout << datum.get()->key_vector_ << std::endl;
			//}

		}

	}


};

template<typename DataType>
class GenericDensityTree {

	typedef GenericDensityNode<DataType>* NodePointer;
	typedef GenericDensityTree* TreePointer;
	typedef std::unique_ptr<GenericDensityNode<DataType> > UniqueNodePointer;


public:

	UniqueNodePointer root_;
	//std::list<DensityNode > mNodes; //This is just a container to hold the nodes.

	GenericDensityTree() {
		//mNodes.clear();
		root_ = UniqueNodePointer(new GenericDensityNode<DataType>());
	}

	GenericDensityTree(GenericDensityTree& otherTree) {

		root_ = UniqueNodePointer(new GenericDensityNode<DataType>(*(otherTree.root_)));

		std::vector<GenericDensityNode<DataType>* > nodesOther;
		nodesOther.push_back(&(*(otherTree.root_)));

		std::vector<GenericDensityNode<DataType>* > nodes;
		nodes.push_back(&(*root_));

		while (nodes.size() > 0) {
			NodePointer nodeOther = nodesOther.back();
			nodesOther.pop_back();

			NodePointer node = nodes.back();
			nodes.pop_back();

			UniqueNodePointer newLeft = UniqueNodePointer(nullptr);
			if (nodeOther->left_child_.get()) {
				newLeft = UniqueNodePointer(new GenericDensityNode<DataType>(*(nodeOther->left_child_.get())));
			}
			UniqueNodePointer newRight = UniqueNodePointer(nullptr);
			if (nodeOther->right_child_.get()) {
				newRight = UniqueNodePointer(new GenericDensityNode<DataType>(*(nodeOther->right_child_.get())));
			}

			node->setLeftChild(newLeft);
			node->setRightChild(newRight);

			if (nodeOther->left_child_.get()) {
				nodesOther.push_back(nodeOther->left_child_.get());
				nodes.push_back(node->left_child_.get());
			}

			if (nodeOther->right_child_.get()) {
				nodesOther.push_back(nodeOther->right_child_.get());
				nodes.push_back(node->right_child_.get());
			}

		}

	}

	GenericDensityTree operator=(GenericDensityTree& otherTree) {

		if (&otherTree == this) {
			return *this;
		}

		root_ = UniqueNodePointer(new GenericDensityNode<DataType>(*(otherTree.root_)));

		std::vector<GenericDensityNode<DataType>* > nodesOther;
		nodesOther.push_back(&(*(otherTree.root_)));

		std::vector<GenericDensityNode<DataType>* > nodes;
		nodes.push_back(&(*root_));

		while (nodes.size() > 0) {
			NodePointer nodeOther = nodesOther.back();
			nodesOther.pop_back();

			NodePointer node = nodes.back();
			nodes.pop_back();

			UniqueNodePointer newLeft = UniqueNodePointer(nullptr);
			if (nodeOther->left_child_.get()) {
				newLeft = UniqueNodePointer(new GenericDensityNode<DataType>(*(nodeOther->left_child_)));
			}
			UniqueNodePointer newRight = UniqueNodePointer(nullptr);
			if (nodeOther->right_child_.get()) {
				newRight = UniqueNodePointer(new GenericDensityNode<DataType>(*(nodeOther->right_child_)));
			}

			node->setLeftChild(newLeft);
			node->setRightChild(newRight);

			if (nodeOther->left_child_.get()) {
				nodesOther.push_back(nodeOther->left_child_.get());
				nodes.push_back(node->left_child_.get());
			}

			if (nodeOther->right_child_.get()) {
				nodesOther.push_back(nodeOther->right_child_.get());
				nodes.push_back(node->right_child_.get());
			}

		}

		return *this;
	}



	DataType* get_random() {

		NodePointer node = root_.get();
		NodePointer next = root_.get();

		while (next) {
			node = next;
			if (rand() % 2 == 0) {
				next = node->left_child_.get();
			}
			else {
				next = node->right_child_.get();
			}
		}

		DataType* nearest = nullptr;

		if (node->data_items_.size() > 0) {

			int rand_idx = rand() % node->data_items_.size();
			std::list<std::unique_ptr<DataType> >::iterator iter = node->data_items_.begin();

			while (rand_idx > 0 && iter != node->data_items_.end()) {
				rand_idx--;
				iter++;
			}

			if (iter == node->data_items_.end()) {
				iter--;
			}

			nearest = iter->get();

		}

		return nearest;

	}


	DataType* get_approximate_nearest(const DataType& datum) {

		NodePointer node = root_->get_leaf(datum);

		generic_density_forest_scalar nearest_dist = std::numeric_limits<generic_density_forest_scalar>::infinity();
		generic_density_forest_scalar current_dist = std::numeric_limits<generic_density_forest_scalar>::infinity();

		const generic_density_forest_vector& key_vector = *(node->get_key_vector_(datum));

		DataType* nearest = nullptr;

		for (const std::unique_ptr<DataType>& tmp : node->data_items_) {
			const generic_density_forest_vector& data_key_vector = *(node->get_key_vector_(*tmp));
			current_dist = (key_vector - data_key_vector).norm();

			if (current_dist < nearest_dist) {
				nearest_dist = current_dist;
				nearest = tmp.get();
			}
		}

		return nearest;

	}

	std::vector<DataType*> get_neighborhood(const DataType& datum) {

		NodePointer node = root_->get_leaf(datum);
		std::vector<DataType*> neighbors;


		for (const std::unique_ptr<DataType>& tmp : node->data_items_) {

			neighbors.push_back(tmp.get());

		}


		return neighbors;

	}


	int get_neighborhood(const DataType& datum, DataType* data_ptr_array[], int max_end_index) {

		NodePointer node = root_->get_leaf(datum);
		int end_idx = 0;

		if (node->data_storage_mode_ == Storage_mode::COPY) {
			for (const std::unique_ptr<DataType>& tmp : node->data_items_) {
				data_ptr_array[end_idx] = tmp.get();
				end_idx++;
				if (max_end_index == end_idx) {
					break;
				}
			}
		}

		return end_idx;

	}


	void rebuild_tree(int tries, int minimum_data_in_leaf, int remember_instances = -1) {
		std::list<std::unique_ptr<DataType> > data;

		std::vector<NodePointer> nodes_to_visit;
		nodes_to_visit.push_back(root_.get());

		while (nodes_to_visit.size() > 0) {
			NodePointer node = nodes_to_visit.back();
			nodes_to_visit.pop_back();
			if (node) {
				nodes_to_visit.push_back(node->left_child_.get());
				nodes_to_visit.push_back(node->right_child_.get());

				for (std::unique_ptr<DataType>& tmp : node->data_items_) {
					data.push_back(std::move(tmp));
				}
			}

		}

		if (remember_instances > 0) {
			while (remember_instances < (int)data.size()) {

				std::list<std::unique_ptr<DataType> >::iterator iter = data.begin();
				int rand_idx = rand() % data.size();

				while (rand_idx > 0) {
					rand_idx--;

					iter++;

					if (iter == data.end()) {
						break;
					}

				}
				if (iter == data.end()) {
					iter--;
				}


				data.erase(iter);


			}
		}

		const generic_density_forest_vector* (*key_vector_fun)(const DataType& datum) = root_->get_key_vector_;

		UniqueNodePointer tmp = UniqueNodePointer(new GenericDensityNode<DataType>());
		tmp->get_key_vector_ = key_vector_fun;
		root_.swap(tmp);

		//std::cout << "Tree has nodes: " << data.size() << std::endl;

		root_->build_tree(tries, minimum_data_in_leaf, std::move(data));

	}


	//Assuming that data_ptr_array has room for k pointers.
	void get_up_to_k_nearest(const DataType& datum, DataType* data_ptr_array[], int k) {
		generic_density_forest_vector key_vector = *(root_->get_key_vector_(datum));
		get_up_to_k_nearest(key_vector, data_ptr_array, k);
	}

	void get_up_to_k_nearest(const generic_density_forest_vector &key_vector, DataType* data_ptr_array[], int k, bool findInSibling = false) {

		NodePointer node = root_->get_leaf(key_vector);
		if (findInSibling)
		{
			if (node->parent_ != nullptr)
			{
				NodePointer left = node->parent_->left_child_.get();
				NodePointer right = node->parent_->right_child_.get();
				node = left == node ? right : left;
				node = node->get_leaf(key_vector);
			}
		}

		auto first_is_closer = [&](DataType* datum1, DataType* datum2) {
			if (!datum1 && !datum2) {
				return false;
			}
			if (datum1 && !datum2) {
				return true;
			}
			if (!datum1 && datum2) {
				return false;
			}

			const generic_density_forest_vector& key_vect1 = *(root_->get_key_vector_(*datum1));
			const generic_density_forest_vector& key_vect2 = *(root_->get_key_vector_(*datum2));

			float dist_to_1 = (key_vector - key_vect1).norm();
			float dist_to_2 = (key_vector - key_vect2).norm();
			return dist_to_1 < dist_to_2;
		};


		for (const std::unique_ptr<DataType>& tmp : node->data_items_) {
			DataType* ptr = tmp.get();

			bool is_in_array_already = false;
			for (int i = 0; i < k; i++) {
				if (data_ptr_array[i]) {
					if (*data_ptr_array[i] == *ptr) {
						is_in_array_already = true;
						break;
					}
				}
			}
			if (is_in_array_already) {
				continue;
			}

			if (first_is_closer(ptr, data_ptr_array[k - 1])) {
				data_ptr_array[k - 1] = ptr;
				std::sort(data_ptr_array, &(data_ptr_array[k]), first_is_closer);
			}

		}


	}



	//DataType* get_approximate_nearest(const DataType& datum){

	//	NodePointer node = get_leaf(datum);

	//	return &(node->data_items_[rand()%node->data_items_.size()]);

	//}



	void remove_empty_leaf(NodePointer node) {

		if (!(node->is_empty_leaf())) {
			return;
		}

		//Node is root
		if (!(node->parent_)) {
			return;
		}

		NodePointer grand_parent = nullptr;
		NodePointer parent = node->parent_;
		if (parent) {
			grand_parent = parent->parent_;
		}
		UniqueNodePointer sibling = nullptr;
		if (parent->left_child_.get() == node) {
			sibling = std::move(parent->right_child_);
		}
		else {
			sibling = std::move(parent->left_child_);
		}


		//The sibling should become the new root
		if (!grand_parent) {
			sibling->parent_ = nullptr;
			root_ = std::move(sibling);
			return;
		}

		assert(root_->hasChildren());

		//The sibling replaces parent
		if (grand_parent->left_child_.get() == parent) {
			grand_parent->setLeftChild(sibling);
			assert(grand_parent->hasChildren());
		}
		else {
			grand_parent->setRightChild(sibling);
			assert(grand_parent->hasChildren());
		}

	}


	std::vector<DataType*> find_data_item_in_tree_exhaustive_search(const DataType& datum) {

		std::vector<NodePointer> nodes;
		nodes.push_back(root_.get());

		std::vector<DataType*> data_items;

		while (nodes.size() > 0) {

			NodePointer node = nodes[nodes.size() - 1];
			nodes.pop_back();

			if (node) {
				NodePointer tmp_ptr = node->left_child_.get();
				if (tmp_ptr) {
					nodes.push_back(tmp_ptr);
				}

				tmp_ptr = node->right_child_.get();
				if (tmp_ptr) {
					nodes.push_back(tmp_ptr);
				}


			}

			std::list<std::unique_ptr<DataType> >::iterator iter = node->data_items_.begin();

			while (iter != node->data_items_.end()) {

				if ((**iter) == datum) {
					data_items.push_back((*iter).get());
				}

				iter++;

			}

		}

		return data_items;

	}


	void remove_data_point(const DataType& datum) {

		std::vector<NodePointer> nodes;
		nodes.push_back(root_.get());

		std::vector<NodePointer> leaves;

		while (nodes.size() > 0) {

			NodePointer node = nodes[nodes.size() - 1];
			nodes.pop_back();

			if (node) {
				bool has_child = false;
				NodePointer tmp_ptr = node->left_child_.get();
				if (tmp_ptr) {
					nodes.push_back(tmp_ptr);
					has_child = true;
				}

				tmp_ptr = node->right_child_.get();
				if (tmp_ptr) {
					nodes.push_back(tmp_ptr);
					has_child = true;
				}

				if (!has_child) {
					leaves.push_back(node);
				}


			}

			std::list<std::unique_ptr<DataType> >::iterator iter = node->data_items_.begin();

			while (iter != node->data_items_.end()) {

				if ((**iter) == datum) {
					iter = node->data_items_.erase(iter);
				}
				else {
					iter++;
				}
			}

		}

		for (NodePointer ptr : leaves) {
			remove_empty_leaf(ptr);
		}


	}

	void build_tree_copying(int tries, int minimum_data_in_leaf, std::vector<DataType >& data) {

		std::list<std::unique_ptr<DataType> > tmp;

		for (unsigned i = 0; i < data.size(); i++) {
			tmp.push_back(std::unique_ptr<DataType>(new DataType));
			*(tmp.back()) = data[i];
		}

		root_->build_tree(tries, minimum_data_in_leaf, std::move(tmp));

	}

	void add_sample(int tries, int minimum_data_in_leaf, int maximum_data_in_leaf, DataType datum) {

		NodePointer node = root_->get_leaf(datum);

		std::unique_ptr<DataType> datum_ptr = std::unique_ptr<DataType>(new DataType());
		*datum_ptr = datum;

		node->add_sample(tries, minimum_data_in_leaf, maximum_data_in_leaf, std::move(datum_ptr));

	}

	generic_density_forest_vector sample_conditioned(const generic_density_forest_vector& regressor) {

		generic_density_forest_vector key = root_->gaussian_model_.first*std::numeric_limits<generic_density_forest_scalar>::quiet_NaN();
		key.tail(regressor.size()) = regressor;

		NodePointer node = root_.get();

		while (node->hasChildren()) {



			key.head(node->regressand_mean_.size()) = node->regressand_mean_ + node->regression_*(regressor - node->regressor_mean_);
			key.head(node->regressand_mean_.size()) += node->regressand_cov_cholesky_*BoxMuller<generic_density_forest_scalar>(node->regressand_mean_.size());


			if (node->left_or_right(key)) {
				node = node->right_child_.get();
			}
			else {
				node = node->left_child_.get();
			}



		}

		return key;

	}


	void form_regression(int regressor_dim) {

		root_->compute_gaussian_models();

		std::vector<NodePointer> nodes;
		nodes.push_back(root_.get());

		while (nodes.size() > 0) {

			NodePointer node = nodes[nodes.size() - 1];
			nodes.pop_back();

			if (node) {
				node->form_regression_parameters(regressor_dim);
				nodes.push_back(node->left_child_.get());
				nodes.push_back(node->right_child_.get());
			}

		}

	}

};

template<typename DataType>
class GenericDensityForest {

	typedef GenericDensityNode<DataType>* NodePointer;
	typedef GenericDensityTree<DataType>* TreePointer;

private:

public:

	std::vector<GenericDensityTree<DataType> > forest_;

	GenericDensityForest() {
		forest_.push_back(GenericDensityTree<DataType>());
	}

	GenericDensityForest(unsigned int numberOfTrees) {
		for (unsigned int treeNum = 0; treeNum < numberOfTrees; treeNum++) {
			forest_.push_back(GenericDensityTree<DataType>());
		}
	}

	GenericDensityForest(GenericDensityForest& other) {
		forest_ = other.forest_;
	}

	GenericDensityForest<DataType> operator=(const GenericDensityForest<DataType>& other) {
		forest_ = other.forest_;
		return *this;
	}

	void set_key_vector_function(const generic_density_forest_vector* (*key_vector_function)(const DataType&)) {
		for (size_t i = 0; i < forest_.size(); i++) {
			forest_[i].root_->get_key_vector_ = key_vector_function;
		}
	}


	void set_split_optimization_mode(Split_Optimization_Mode mode) {
		for (size_t i = 0; i < forest_.size(); i++) {
			forest_[i].root_->split_mode = mode;
		}
	}

	void add_sample(int tries, int minimum_data_in_leaf, int maximum_data_in_leaf, DataType datum) {
		for (size_t i = 0; i < forest_.size(); i++) {
			forest_[i].add_sample(tries, minimum_data_in_leaf, maximum_data_in_leaf, datum);
		}
	}

	DataType* get_approximate_nearest(const DataType& datum) {

		std::vector<DataType*> nearest_ones(forest_.size(), nullptr);

		//#pragma omp parallel for
		for (size_t i = 0; i < (int)forest_.size(); i++) {
			nearest_ones[i] = forest_[i].get_approximate_nearest(datum);
		}

		NodePointer random_node = forest_[0].root_.get();
		const generic_density_forest_vector& key_vector = *(random_node->get_key_vector_(datum));

		int nearest_idx = 0;
		generic_density_forest_scalar nearest_dist = std::numeric_limits<generic_density_forest_scalar>::infinity();
		generic_density_forest_scalar current_dist = std::numeric_limits<generic_density_forest_scalar>::infinity();

		for (size_t i = 0; i < nearest_ones.size(); i++) {

			if (nearest_ones[i]) {

				const generic_density_forest_vector& comparison_key_vector = *(random_node->get_key_vector_(*(nearest_ones[i])));
				current_dist = (key_vector - comparison_key_vector).norm();

				if (current_dist < nearest_dist) {
					nearest_dist = current_dist;
					nearest_idx = i;
				}

			}

		}

		return nearest_ones[nearest_idx];


	}

	std::vector<DataType*> get_neighborhood(const DataType& datum) {

		std::vector<DataType*> nearest_ones;

		for (size_t i = 0; i < (int)forest_.size(); i++) {
			std::vector<DataType*> nearest_ones_tree;
			nearest_ones_tree = forest_[i].get_neighborhood(datum);
			for (size_t j = 0; j < nearest_ones_tree.size(); j++) {
				nearest_ones.push_back(nearest_ones_tree[j]);
			}
		}

		return nearest_ones;

	}

	//Assuming that data_ptr_array has room for k pointers.
	int get_up_to_k_nearest(const DataType& datum, DataType* data_ptr_array[], int k) {
		const generic_density_forest_vector& key_vector = *forest_[0].root_->get_key_vector_(datum);
		return get_up_to_k_nearest(key_vector, data_ptr_array, k);
	}

	int get_up_to_k_nearest(const generic_density_forest_vector &key_vector, DataType* data_ptr_array[], int k) {

		for (int i = 0; i < k; i++) {
			data_ptr_array[i] = nullptr;
		}

		if (k == 0) {
			return 0;
		}

		for (size_t i = 0; i < (int)forest_.size(); i++) {
			forest_[i].get_up_to_k_nearest(key_vector, data_ptr_array, k);
			forest_[i].get_up_to_k_nearest(key_vector, data_ptr_array, k, true);  //for robustness against splits that leave a child with only few samples, also search the sibling
		}

		int found_count = 0;

		for (int i = 0; i < k; i++) {
			if (data_ptr_array[i]) {
				found_count++;
			}
		}

		return found_count;

	}


	int get_neighborhood(const DataType& datum, DataType* nearest_ones[], int max_amount) {

		int end_idx = 0;

		for (size_t i = 0; i < (int)forest_.size(); i++) {
			end_idx += forest_[i].get_neighborhood(datum, &nearest_ones[end_idx], max_amount - end_idx);
		}

		return end_idx;

	}


	//DataType* get_approximate_nearest(const DataType& datum){


	//	return forest_[rand()%forest_.size()].get_approximate_nearest(datum);


	//}


	void build_forest(int tries, int minimum_data_in_leaf, std::vector<std::unique_ptr<DataType> > data) {
		for (int i = 0; i < (int)forest_.size(); i++) {
			GenericDensityTree<DataType>& tree = forest_[i];
			tree.root_->build_tree(tries, minimum_data_in_leaf, data);
		}
	}

	void build_forest_copying(int tries, int minimum_data_in_leaf, std::vector<DataType >& data) {
		for (int i = 0; i < (int)forest_.size(); i++) {
			GenericDensityTree<DataType>& tree = forest_[i];
			tree.build_tree_copying(tries, minimum_data_in_leaf, data);
		}
	}


	generic_density_forest_vector sample_conditioned(const generic_density_forest_vector& regressor) {

		return forest_[rand() % forest_.size()].sample_conditioned(regressor);

	}

	void form_regressions(int regressor_dim) {
		for (int i = 0; i < (int)forest_.size(); i++) {
			GenericDensityTree<DataType>& tree = forest_[i];
			tree.form_regression(regressor_dim);
		}

	}

	void claim_ownership_of_forest(GenericDensityForest<DataType>& other_forest) {
		forest_ = std::move(other_forest.forest_);
	}

	void claim_ownership_of_tree(GenericDensityTree<DataType>& tree, size_t number_of_trees_in_this_forest) {
		if (forest_.size() < number_of_trees_in_this_forest) {
			forest_.push_back(std::move(tree));
			return;
		}

		if (forest_.size() > number_of_trees_in_this_forest) {
			forest_.pop_back();
		}

		int swap_idx = rand() % forest_.size();
		//forest_[swap_idx].root_.reset();
		//forest_[swap_idx].root_ = std::move(tree.root_);
		forest_[swap_idx].root_.swap(tree.root_);
	}

	void remove_data_point(DataType datum) {
		for (int i = 0; i < (int)forest_.size(); i++) {
			GenericDensityTree<DataType>& tree = forest_[i];
			tree.remove_data_point(datum);
		}
	}

	DataType* get_random() {
		return forest_[rand() % forest_.size()].get_random();
	}

	bool forest_ready_to_use(void) {

		if (forest_.size() == 0) {
			return false;
		}

		if (forest_[0].root_->separating_normal_.size() == 0 && forest_[0].root_->data_items_.size() == 0) {
			return false;
		}

		return true;

	}

};

#endif