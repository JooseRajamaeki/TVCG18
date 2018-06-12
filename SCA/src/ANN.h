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

#ifndef ANN_NETWORK_HPP_
#define ANN_NETWORK_HPP_


#include <vector>
#include <utility>
#include <deque>
#include <list>
#include <exception>
#include <stdexcept>
#include <iomanip>
#include <iostream>
#include <thread>
#include <fstream>
#include <map>
#include <mutex>
#include <future>

#include "Eigen\Dense"
#include "MiscUtils.hpp"
#include "ProbUtils.hpp"

inline void sigmoid(float& num) {
	num = std::exp(-num);
	num = 1.0f / (1.0f + num);
}

inline float sigmoid_derivative(float num) {
	sigmoid(num);
	return num*(1.0f - num);
}

template<typename Scalar>
std::vector<Scalar*> vector_to_ptrs(std::vector<std::vector<Scalar> >& in_vec) {
	unsigned size = in_vec.size();

	std::vector<Scalar*> out(size,nullptr);

	for (unsigned i = 0; i < size; i++) {
		out[i] = in_vec[i].data();
	}

	return out;

}

std::vector<float*> vector_to_ptrs(std::vector<Eigen::VectorXf>& in_vec);
std::vector<float*> vector_to_ptrs(std::vector<std::unique_ptr<Eigen::VectorXf>>& in_vec);
std::vector<float*> vector_to_ptrs(std::vector<std::shared_ptr<Eigen::VectorXf>>& in_vec);


template<typename Scalar>
bool has_valid_nums(Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic>& matrix) {

	int rows = matrix.rows();
	int cols = matrix.cols();
	for (int row = 0; row < rows; row++) {
		for (int col = 0; col < cols; col++) {

			const Scalar& number = matrix(row, col);

			if (number - number != number - number) {
				return false;
			}

		}
	}

	return true;

}

template<typename Scalar>
bool has_valid_nums(Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& matrix) {

	int rows = matrix.size();
	for (int row = 0; row < rows; row++) {
			const Scalar& number = matrix(row);

			if (number - number != number - number) {
				return false;
			}

	}

	return true;

}


class Operation {
public:

	enum OperationType {
		TANH, ELU, SELU, BSELU, RELU, UNITY, EVOLVING_UNITY, IDENTITY, LINEAR_CLIPPED, SQUARE, LOG, SIGMOID, NEGATE, SOFTPLUS, MIRROR
	};

	Operation();
	virtual ~Operation();


	OperationType type_;
	int index_;
	bool ready_;
	int size_;

	Eigen::VectorXf activations_;
	Eigen::VectorXf outputs_;

	Eigen::VectorXf gradient_;
	Eigen::VectorXf deltas_;

	void zero_delta();
	void zero_activations();
	void set_activation(const float* in);

	virtual void init(unsigned dimension, int operation_index);
	virtual void perform_operation();
	virtual void get_gradients();
	virtual void prepare_forward();
	virtual void prepare_backward();

	void increase_dimension();

	void write_to_file(std::ofstream& filename);
	void load_from_file(std::ifstream& filename);

};

class IdentityOperation : public Operation {

public:

	IdentityOperation();
	virtual ~IdentityOperation();

	void perform_operation();
	void get_gradients();

};

class TanhOperation : public Operation {

public:

	TanhOperation();
	virtual ~TanhOperation();

	void perform_operation();
	void get_gradients();

};

class UnityOperation : public Operation {
public:

	UnityOperation();
	virtual ~UnityOperation();

	void perform_operation();
	void get_gradients();
};

class EvolvingUnityOperation : public Operation {
public:

	EvolvingUnityOperation();
	virtual ~EvolvingUnityOperation();

	void perform_operation();
	void get_gradients();
};

//Derivative is always one but the output is cosntrained between zero and one.
class LinearClippedOperation : public Operation {
public:

	LinearClippedOperation();
	virtual ~LinearClippedOperation();

	void perform_operation();
	void get_gradients();
};

class SquareOperation : public Operation {
public:

	SquareOperation();
	virtual ~SquareOperation();

	Eigen::VectorXf desired_;

	void perform_operation();
	void get_gradients();
	void init(unsigned dimension, int operation_index);
};

class LogOperation : public Operation {
public:
	LogOperation();
	virtual ~LogOperation();

	void perform_operation();
	void get_gradients();
};

class SigmoidOperation : public Operation {

public:
	SigmoidOperation();
	virtual ~SigmoidOperation();

	void perform_operation();
	void get_gradients();
};

class NegateOperation : public Operation {
public:
	NegateOperation();
	virtual ~NegateOperation();

	void perform_operation();
	void get_gradients();
};

class SoftPlusOperation : public Operation {
public:
	SoftPlusOperation();
	virtual ~SoftPlusOperation();

	void perform_operation();
	void get_gradients();
};

class ELUOperation : public Operation {
public:
	ELUOperation();
	virtual ~ELUOperation();

	void perform_operation();
	void get_gradients();
};

class SELUOperation : public Operation {
public:

	const float lambda_;

	SELUOperation();
	SELUOperation operator=(const SELUOperation&);
	virtual ~SELUOperation();

	void perform_operation();
	void get_gradients();
};

//Scaled ELU-operation with bipolar activation.
class BSELUOperation : public Operation {
public:

	const float lambda_;

	BSELUOperation();
	BSELUOperation operator=(const BSELUOperation&);
	virtual ~BSELUOperation();

	void perform_operation();
	void get_gradients();
};

class MirrorOperation : public Operation {
public:
	MirrorOperation();
	virtual ~MirrorOperation();

	void perform_operation();
	void get_gradients();
};

class Connection {

private:

public:
	void standard_init();
	Connection();
	Connection(int input_oper_idx, int output_oper_idx);
	Connection(const Connection& other);
	Connection operator=(const Connection& other);
	void copy_values(const Connection& other);
	int input_node_index_;
	int output_node_index_;

	bool forward_path_;
	bool backward_path_;
	bool fixed_connection_;
	bool identity_connection_;

	int input_dimension_;
	int output_dimension_;

	bool ready_;

	std::vector<Connection*> dependencies_forward_;
	std::vector<Connection*> dependencies_backward_;

	Eigen::VectorXf input_tmp_;
	Eigen::VectorXf output_tmp_;

	Operation* in_operation_;
	Operation* out_operation_;

	Eigen::MatrixXf weights_;
	Eigen::MatrixXf prev_weights_;
	Eigen::MatrixXf prev_prev_weights_;
	
	Eigen::MatrixXf gradients_;

	int gradient_samples_;
	Eigen::MatrixXf gradients_sum_;
	
	//Quantities needed for ADAM.
	Eigen::MatrixXf gradients_first_moment_;
	Eigen::MatrixXf gradients_second_moment_;
	int adam_sample_;

	//Quantities for RPROP
	Eigen::MatrixXf last_gradients_;
	Eigen::MatrixXf gradient_signs_;
	Eigen::MatrixXf update_values_;

	void set_identity();
	void connect_to_operations(const std::vector<Operation*>& operations);

	//Returns true if all the dependencies were ready and output was formed. Otherwise return false.
	bool form_output(float drop_out_stdev);
	void resize(int input_dimension = -1, int output_dimension = -1);

	void duplicate_output(int dimension);
	void duplicate_input(int dimension);

	bool compute_deltas(float error_drop_out_rate = 0.0f);
	void compute_gradients();

	void zero_accumulated_gradients();
	void accumulate_gradients(bool increase_counter = true);
	void apply_gradients(const float& learning_rate = 0.01);
	void randomize_weights(const float min_val, const float max_val);
	void randomize_weights_gaussian(const float stdev);

	void apply_adam(const float learning_rate = 0.01, const float first_moment_smoothing = 0.9, const float second_moment_smoothing = 0.99, const float epsilon = 0.0000001);
	void apply_adamax(const float learning_rate = 0.01, const float first_moment_smoothing = 0.9, const float second_moment_smoothing = 0.99, const float epsilon = 0.0000001);

	void apply_rprop(float min_val = 0.0000000001f, float max_val = 2.0f);
	void apply_rmsprop(float learning_rate);

	void clamp_weights(float min_weight, float max_weight);

	void write_matrix_to_file(std::ofstream& filename, Eigen::MatrixXf& matrix_to_write);
	Eigen::MatrixXf load_matrix_from_file(std::ifstream& filename);

	void write_map_to_file(std::ofstream& filename, std::map<int,int>& map_to_write);
	std::map<int, int> load_map_from_file(std::ifstream& filename);
	
	void write_to_file(std::ofstream& filename);
	void load_from_file(std::ifstream& filename);

};

class MultiLayerPerceptron {

public:

	std::vector<std::unique_ptr<Operation> > operations_;
	std::vector<Connection> connections_;

	Operation* input_operation_;
	Operation* output_operation_;

	Eigen::VectorXf input_mean_;
	Eigen::VectorXf input_stdev_;

	float learning_rate_;

	float min_weight_;
	float max_weight_;

	float adam_first_moment_smoothing_;
	float adam_second_moment_smoothing_;

	float mse_;
	float prev_mse_;

	float max_gradient_norm_;
	float drop_out_stdev_;
	float error_drop_out_prob_;

	//Indicate if we are training. This is mainly for drop out purposes
	float training_;



	MultiLayerPerceptron();
	MultiLayerPerceptron(const MultiLayerPerceptron& other);
	void copy(const MultiLayerPerceptron& other);
	MultiLayerPerceptron operator=(const MultiLayerPerceptron& other);

	void copy_operations(const std::vector<std::unique_ptr<Operation>>& operations);
	//First element in <layer_widths> is the input dimension and last element is the output dimension.
	void build_dependencies();
	void reset_training();
	void build_network(const std::vector<unsigned>& layer_widths);
	void build_elu_network(const std::vector<unsigned>& layer_widths);

	void run(const float* in, float* out = nullptr);
	float mse(const float** in, const float** out, unsigned data_points);

	void backpropagate_deltas(const float* in, const float* out, bool is_gradient = false, const float* output_scales = nullptr);
	//<is_gradient> tells if <out> contains the gradient. If <is_gradient> is false <out> is the target value.
	void compute_gradients(const float* in, const float* out, bool is_gradient = false, const float* output_scales = nullptr);
	void compute_gradients_log_bernoulli(const float* in, const float* out);

	void train_back_prop(const float** in, const float** out, unsigned data_points, unsigned minibatch_size, bool is_gradient = false, const float* output_scales = nullptr);
	void train_adam(const float** in, const float** out, unsigned data_points, unsigned minibatch_size, bool is_gradient = false, const float* output_scales = nullptr);
	void train_adamax(const float** in, const float** out, unsigned data_points, unsigned minibatch_size, bool is_gradient = false, const float* output_scales = nullptr);
	void train_rprop(const float** in, const float** out, unsigned data_points, bool is_gradient = false, const float* output_scales = nullptr);
	void train_rmsprop(const float** in, const float** out, unsigned data_points, unsigned minibatch_size, bool is_gradient = false, const float* output_scales = nullptr);
	void randomize_weights(float min_val, float max_val, float bias_val = 0.0f);
	void randomize_weights();
	void randomize_weights_and_biases(float min_val, float max_val);
	void randomize_random_layer(float min_val, float max_val);
	void train_input_normalization(const float** in, unsigned data_points);
	void train_bernoulli_classifier_adam(const float** in, const float** out, unsigned data_points, unsigned minibatch_size);
	void incremental_matching(const float** noise, const float** data, unsigned conditioning_dim, unsigned noise_dim, unsigned data_points, unsigned minibatch_size, int supervised_minibatch_size, float (*distance_measure)(const float* pt1,const float* pt2) = nullptr, void (*distance_measure_gradient)(const float* prediction, const float* true_val, float* gradient) = nullptr);
	void hausdorff_matching(const float** noise, const float** data, unsigned conditioning_dim, unsigned noise_dim, unsigned data_points, unsigned minibatch_size, int supervised_minibatch_size, float(*distance_measure)(const float* pt1, const float* pt2) = nullptr, void(*distance_measure_gradient)(const float* prediction, const float* true_val, float* gradient) = nullptr);
	float incremental_matching_error(const float** noise, const float** data, unsigned conditioning_dim, unsigned noise_dim, unsigned data_points, float(*distance_measure)(const float* pt1, const float* pt2) = nullptr);

	void write_to_file(std::string filename);
	void load_from_file(std::string filename);
	std::string read_operations(std::ifstream& input_file);
	std::string read_connections(std::ifstream& input_file, std::string line);
	void perform_flow(std::deque<Connection*>& flow_queue);
	int get_amount_parameters();
	std::vector<float> get_parameters_as_one_vector();
	void set_parameters(const std::vector<float>& parameters);
	std::vector<Connection*> get_connections_in_order();



};

#endif
