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

#include "SCAController.h"

using namespace Eigen;

namespace AaltoGames
{


	bool valid_float(const float& num) {

		if (num <= std::numeric_limits<float>::max() && num >= std::numeric_limits<float>::lowest()) {
			return true;
		}

		return false;
	}

	void SCAControl::init_neural_net(int input_dim, int output_dim, MultiLayerPerceptron& net) {

		unsigned seed = (unsigned)time(nullptr);
		srand(seed);

		const int layer_width = 100;


		std::vector<unsigned> layers;
		layers.push_back(input_dim);
		for (int i = 0; i < amount_network_layers_; i++) {
			layers.push_back(layer_width);
		}
		layers.push_back(output_dim);



		switch (network_type_)
		{
		case AaltoGames::SCAControl::ELU:
			net.build_elu_network(layers);
			break;
		case AaltoGames::SCAControl::BSELU:
			net.build_network(layers);
			break;
		default:
			net.build_network(layers);
			break;
		}


		net.max_gradient_norm_ = 0.1f;
		net.learning_rate_ = 0.0001f;
		net.drop_out_stdev_ = drop_out_stdev_;
		net.min_weight_ = std::numeric_limits<float>::lowest();
		net.max_weight_ = std::numeric_limits<float>::max();
		net.adam_first_moment_smoothing_ = 0.9f;
		net.adam_second_moment_smoothing_ = 0.99f;


		float weight_min = -0.05f;
		float weight_max = 0.05f;
		net.randomize_weights(weight_min, weight_max);

	}

	inline static void form_key_vector(SCAControl* optimizer, const Eigen::VectorXf& state, const Eigen::VectorXf& control, Eigen::VectorXf& key_vector) {

		int key_vector_size = state.size();
		if (key_vector.size() != key_vector_size) {
			key_vector.resize(key_vector_size);
		}

		key_vector.head(optimizer->state_dimension_) = state;
	}

	int SCAControl::getCurrentStep(void) {
		return current_step_;
	}

	static float distance_metric(const SCAControl::TeachingSample& datum1, const SCAControl::TeachingSample& datum2) {

		return (datum1.state_ - datum2.state_).norm();

	}

	SCAControl::TeachingSample::Scalar distance_metric_key_vector(SCAControl::TeachingSample& datum1, SCAControl::TeachingSample& datum2) {
		return (datum1.key_vector_ - datum2.key_vector_).norm();
	}

	const generic_density_forest_vector* teaching_sample_to_key_vector(const SCAControl::TeachingSample& sample) {


		if (sample.key_vector_.size() == 0) {
			std::cout << "Key vector size zero. Msg from key vect fun.";
		}

		const generic_density_forest_vector* return_value = &(sample.key_vector_);

		return return_value;

	}

	void SCAControl::init(int nSamples, int nSteps, int nStateDimensions, int control_dimension_, const float *controlMinValues, const float *controlMaxValues, const float *controlMean, const float *controlPriorStd, const float *controlDiffPriorStd, const float controlMutationStdScale, bool _useMirroring)
	{

		this->state_dimension_ = nStateDimensions;
		this->control_dimension_ = control_dimension_;

		this->control_min_.resize(control_dimension_);
		this->control_max_.resize(control_dimension_);
		memcpy(&this->control_min_[0], controlMinValues, sizeof(float)*control_dimension_);
		memcpy(&this->control_max_[0], controlMaxValues, sizeof(float)*control_dimension_);

		unsigned seed = (unsigned)time(nullptr);
		srand(seed);


		resample_threshold_ = 0;
		Eigen::initParallel();
		time_t timer;
		struct tm y2k = { 0 };
		double seconds;

		y2k.tm_hour = 0;   y2k.tm_min = 0; y2k.tm_sec = 0;
		y2k.tm_year = 100; y2k.tm_mon = 0; y2k.tm_mday = 1;

		time(&timer);





		seconds = difftime(timer, mktime(&y2k));
		iteration_idx_ = 0;
		experience_.resize(nSteps + 1);
		previous_experience_.resize(nSteps + 1);
		this->amount_samples_ = nSamples;
		this->max_samples_ = nSamples;
		this->steps_ = nSteps;
		this->max_steps_ = nSteps;

		this->control_mean_.resize(control_dimension_);
		this->control_std_.resize(control_dimension_);
		this->control_diff_std_.resize(control_dimension_);




		old_best_.resize(nSteps + 1);
		if (controlMean != NULL)
		{
			memcpy(&this->control_mean_[0], controlMean, sizeof(float)*control_dimension_);
		}
		else
		{
			this->control_mean_.setZero();
		}
		for (int step = 0; step <= nSteps; step++)
		{
			old_best_[step].init(nStateDimensions, control_dimension_);
			experience_[step].resize(nSamples);

			for (int i = 0; i < nSamples; i++)
			{
				experience_[step][i].init(nStateDimensions, control_dimension_);
			}
		}

		best_cost_ = FLT_MAX;  //not yet found any solution
		time_advanced_ = false;


		setSamplingParams(controlPriorStd, controlDiffPriorStd, controlMutationStdScale);
		best_full_sample_idx_ = 0;


		learning_ = true;

	}

	SCAControl::SCAControl()
	{

		auto dummy = []() {
			return;
		};

		training_neural_net_ = std::async(std::launch::async, dummy);
		building_forest_ = std::async(std::launch::async, dummy);


		machine_learning_mse_ = 0.0f;
		machine_learning_and_chosen_control_discrepancy_ = 0.0f;
		input_noise_dim_ = 20;

		network_type_ = NetworkType::BSELU;
		amount_network_layers_ = 5;


		use_sampling_ = true;
		force_old_best_valid_ = false;

		sampling_mode_ = SamplingMode::MINIMIZING_INFORMATION_LOSS;
		use_machine_learning_ = true;
		use_forests_ = true;
		amount_recent_ = 100;


		amount_data_in_tree_ = 2500;

		nn_trajectories_ = 3;
		machine_learning_samples_ = 3;
		noisy_machine_learning_samples_ = 3;
		zero_samples_ = 0;



		stored_sample_percentage_ = 25;

		no_prior_trajectory_portion_ = 0.25f;
		regularization_noise_ = 0.0f;
		drop_out_stdev_ = 0.1;


		setParams(0.5f, true, 0);

		validation_fraction_ = 0.1f;


		learning_budget_ = 2000;

		number_of_data_in_leaf_ = 10;
		number_of_hyperplane_tries_ = 25;
		number_of_nearest_neighbor_trees_ = 5;



		ann_forest_ = GenericDensityForest<TeachingSample>(number_of_nearest_neighbor_trees_);
		ann_forest_.set_key_vector_function(teaching_sample_to_key_vector);
		tree_under_construction_ = GenericDensityTree<TeachingSample>();
		tree_under_construction_.root_->get_key_vector_ = teaching_sample_to_key_vector;


	}

	SCAControl::~SCAControl()
	{
		building_forest_.wait();
		training_neural_net_.wait();
	}



	void __stdcall SCAControl::setSamplingParams(const float *controlPriorStd, const float *controlDiffPriorStd, float controlMutationStdScale)
	{
		memcpy(&this->control_std_[0], controlPriorStd, sizeof(float)*control_dimension_);
		memcpy(&this->control_diff_std_[0], controlDiffPriorStd, sizeof(float)*control_dimension_);
		control_variation_std_ = controlMutationStdScale*this->control_std_;
		static_prior_.resize(1, control_dimension_);
		static_prior_.mean[0] = control_mean_;
		static_prior_.setStds(this->control_std_);
		static_prior_.weights[0] = 1;
		static_prior_.weightsUpdated();

	}

	void SCAControl::resize_marginal(void) {
		while ((int)experience_.size() > steps_ + 1) {
			experience_.pop_back();
		}

		while ((int)experience_.size() < steps_ + 1) {
			experience_.push_back(std::vector<MarginalSample>());
		}


		for (int step = 0; step < (int)experience_.size(); step++) {
			std::vector<MarginalSample>& marginal = experience_[step];
			while ((int)marginal.size() > amount_samples_) {
				marginal.pop_back();
			}

			while ((int)marginal.size() < amount_samples_) {
				marginal.push_back(MarginalSample());
			}

		}

		for (int step = 0; step < (int)experience_.size(); step++) {
			for (int particle = 0; particle < (int)experience_[0].size(); particle++) {
				MarginalSample& sample = experience_[step][particle];
				sample.init(state_dimension_, control_dimension_);
			}
		}
	}

	void __stdcall SCAControl::startIteration(bool advanceTime, const float *initialState)
	{

		DiagonalGaussian sample_dist;
		sample_dist.first = Eigen::VectorXf::Zero(control_dimension_);
		sample_dist.second = Eigen::VectorXf::Zero(control_dimension_);

		const int max_distributions = 20;

		std::vector<DiagonalGaussian> dists_tmp(max_distributions, sample_dist);
		gaussian_distributions_ = std::vector<std::vector<DiagonalGaussian> >(amount_samples_, dists_tmp);
		sampling_distributions_ = std::vector<DiagonalGaussian>(amount_samples_, sample_dist);

		Eigen::Map<const Eigen::VectorXf> init_state_map(initialState, state_dimension_);

		float state_discrepancy_debug = 0.0f;
		if (old_best_.size() > 1 && old_best_[1].previousState.size() == state_dimension_) {
			state_discrepancy_debug = (init_state_map - old_best_[1].state).norm();
		}


		resize_marginal();


		for (int i = 0; i < amount_samples_; i++)
		{
			MarginalSample &sample = experience_[0][i];
			memcpy(&sample.state[0], initialState, sizeof(float)*state_dimension_);
			sample.control.setZero();
			sample.previousControl.setZero();
			sample.previousPreviousControl.setZero();
			sample.forwardBelief = 1;
			sample.fwdMessage = 1;
			sample.belief = 1;
			sample.fullCost = 0;
			sample.stateCost = 0;
			sample.controlCost = 0;
			sample.previousMarginalSampleIdx = i;
			sample.bestStateCost = FLT_MAX;
			sample.costSoFar = 0;
			sample.nForks = 0;
		}


		time_advanced_ = advanceTime;


		old_best_valid_ = false;

		if (iteration_idx_ > 0) {
			old_best_valid_ = true;
		}

		const MarginalSample& performed_transition = old_best_[1];

		if (advanceTime)
		{


			if (iteration_idx_ > 0) {

				old_best_valid_ = false;

				float dist = 0.0f;
				if (performed_transition.state.size() > 0) {
					for (int i = 0; i < state_dimension_; i++) {
						float diff = performed_transition.state[i] - initialState[i];
						dist += std::abs(diff);
					}
				}

				const float tolerance = 0.00001f;

				if (dist <= tolerance) {
					old_best_valid_ = true;
				}



				if (force_old_best_valid_) {
					old_best_valid_ = true;
				}

			}
			else {
				old_best_valid_ = false;
			}


			if (old_best_valid_) {
				if (old_best_.size() > 0) {
					old_best_.pop_front();
				}
				if (previous_experience_.size() > 0) {
					previous_experience_.pop_front();
				}

				for (int i = 0; i < amount_samples_; i++)
				{
					experience_[0][i].control = performed_transition.control;
					experience_[0][i].previousControl = performed_transition.previousControl;
					experience_[0][i].previousPreviousControl = performed_transition.previousPreviousControl;
				}
			}
			else {
				old_best_.clear();
				previous_experience_.clear();

				for (int i = 0; i < amount_samples_; i++)
				{
					experience_[0][i].control.setZero();
					experience_[0][i].previousControl.setZero();
					experience_[0][i].previousPreviousControl.setZero();
				}
			}

		}
		else {


			if (iteration_idx_ > 0) {

				old_best_valid_ = false;

				float dist = 0.0f;
				if (performed_transition.previousState.size() > 0) {
					for (int i = 0; i < state_dimension_; i++) {
						float diff = performed_transition.previousState[i] - initialState[i];
						dist += std::abs(diff);
					}
				}

				const float tolerance = 0.00001f;

				if (dist <= tolerance) {
					old_best_valid_ = true;
				}


				if (force_old_best_valid_) {
					old_best_valid_ = true;
				}

			}
			else {
				old_best_valid_ = false;
			}

			if (!old_best_valid_) {
				old_best_.clear();
				previous_experience_.clear();

				for (int i = 0; i < amount_samples_; i++)
				{
					experience_[0][i].control.setZero();
					experience_[0][i].previousControl.setZero();
					experience_[0][i].previousPreviousControl.setZero();
				}
			}
		}

	}

	void __stdcall SCAControl::startPlanningStep(int step)
	{



		if (keys_.size() != amount_samples_) {
			keys_.resize(amount_samples_);
		}

		if (priors_.size() != amount_samples_) {

			priors_.clear();

			DiagonalGMM prior;
			prior.resize(1, control_dimension_);
			prior.weights[0] = 1;
			prior.weightsUpdated();
			for (int i = 0; i < amount_samples_; i++) {
				priors_[i] = std::unique_ptr<DiagonalGMM>(new DiagonalGMM(prior));
			}

		}

		if (proposals_.size() != amount_samples_) {

			proposals_.clear();

			DiagonalGMM proposal = DiagonalGMM();
			proposal.resize(1, control_dimension_);

			for (int i = 0; i < amount_samples_; i++) {
				proposals_[i] = std::unique_ptr<DiagonalGMM>(new DiagonalGMM(proposal));
			}

		}


		current_step_ = step;
		next_step_ = current_step_ + 1;


		int particle_idx = 0;
		if (particle_idx < (int)experience_[next_step_].size()) {
			experience_[next_step_][particle_idx].particleRole = ParticleRole::OLD_BEST;
			particle_idx++;
			experience_[next_step_][particle_idx].priorSampleIdx = 0;
		}


		for (int i = 0; i < zero_samples_; i++)
		{
			if (particle_idx < (int)experience_[next_step_].size()) {
				experience_[next_step_][particle_idx].particleRole = ParticleRole::ZERO_SAMPLE;
				particle_idx++;
			}

		}


		if (use_machine_learning_ && false) {
			if (particle_idx < (int)experience_[next_step_].size()) {
				experience_[next_step_][particle_idx].particleRole = ParticleRole::MACHINE_LEARNING_NO_RESAMPLING;
				particle_idx++;
			}
		}

		if (use_machine_learning_) {
			for (int i = 0; i < machine_learning_samples_; i++)
			{
				if (particle_idx < (int)experience_[next_step_].size()) {
					experience_[next_step_][particle_idx].particleRole = ParticleRole::MACHINE_LEARNING_NO_VARIATION;
					particle_idx++;
				}

			}
		}


		if (use_machine_learning_) {
			for (int i = 0; i < noisy_machine_learning_samples_; i++)
			{
				if (particle_idx < (int)experience_[next_step_].size()) {
					experience_[next_step_][particle_idx].particleRole = ParticleRole::MACHINE_LEARNING;
					particle_idx++;
				}

			}
		}


		int nFree = (int)(no_prior_trajectory_portion_*(float)amount_samples_);
		for (int i = 0; i < nFree; i++) {
			if (particle_idx < (int)experience_[next_step_].size()) {
				experience_[next_step_][particle_idx].particleRole = ParticleRole::FREE;
				particle_idx++;
			}
		}


		if (iteration_idx_ > 0)
		{
			for (int i = 0; i < nn_trajectories_; i++)
			{
				if (particle_idx < (int)experience_[next_step_].size()) {
					experience_[next_step_][particle_idx].particleRole = ParticleRole::NEAREST_NEIGHBOR;
					particle_idx++;
				}
			}
		}




		if (iteration_idx_ > 0) {
			for (; particle_idx < (int)experience_[next_step_].size(); particle_idx++) {
				if (particle_idx < (int)experience_[next_step_].size()) {
					experience_[next_step_][particle_idx].particleRole = ParticleRole::PREVIOUS_FRAME_PRIOR;
				}
			}
		}



		{

			if (step > 0)
			{

				float bestCost = FLT_MAX;
				for (int sampleIdx = 0; sampleIdx < amount_samples_; sampleIdx++)
				{
					bestCost = std::min(bestCost, (float)experience_[step][sampleIdx].fullCost);
				}


				int forkedTrajectories[1024];
				int prunedTrajectories[1024];
				int nForked = 0;
				int nPruned = 0;
				float costThreshold = bestCost + (float)resample_threshold_;
				const float tolerance = 0.01f;
				costThreshold += tolerance;

				for (int sampleIdx = 0; sampleIdx < amount_samples_; sampleIdx++)
				{
					experience_[next_step_][sampleIdx].previousMarginalSampleIdx = sampleIdx;
				}

				for (int sampleIdx = 0; sampleIdx < amount_samples_; sampleIdx++)
				{
					//All trajectories with cost greater than costThreshold and no special role are pruned
					MarginalSample &nextSample = experience_[next_step_][sampleIdx];

					if ((experience_[step][sampleIdx].fullCost > costThreshold)
						&& (nextSample.particleRole != ParticleRole::OLD_BEST)
						&& (nextSample.particleRole != ParticleRole::MACHINE_LEARNING_NO_RESAMPLING))
					{
						prunedTrajectories[nPruned] = sampleIdx;
						nPruned++;
					}

					//The pruned trajectories will be reallocated to the trajectories with cost below threshold (i.e., forks created)
					else if (experience_[step][sampleIdx].fullCost <= costThreshold)
					{
						forkedTrajectories[nForked] = sampleIdx;
						nForked++;
					}

				}
				//prune others and select a random continued one
				for (int i = 0; i < nPruned; i++)
				{
					int sampleIdx = prunedTrajectories[i];
					MarginalSample &nextSample = experience_[next_step_][sampleIdx];
					nextSample.previousMarginalSampleIdx = forkedTrajectories[randInt(0, nForked - 1)];
					experience_[step][nextSample.previousMarginalSampleIdx].nForks++;
				}
			}
		}

		//remember the number of forks
		for (int sampleIdx = 0; sampleIdx < amount_samples_; sampleIdx++)
		{
			MarginalSample &nextSample = experience_[next_step_][sampleIdx];
			MarginalSample &sample = experience_[step][nextSample.previousMarginalSampleIdx];
			nextSample.nForks = sample.nForks;
		}

	}

	double SCAControl::getBestTrajectoryCost()
	{
		return best_cost_;
	}

	int __stdcall SCAControl::getBestSampleLastIdx()
	{
		return best_full_sample_idx_;
	}

	float SCAControl::getMachineLearningMSE()
	{
		std::lock_guard<std::mutex> lock(mse_changing_mutex_);
		return machine_learning_mse_;
	}

	float SCAControl::getMachineLearningChosenControlDiscrepancy()
	{
		machine_learning_and_chosen_control_discrepancy_ = 0.0f;

		if (actor_.get()) {

			Eigen::VectorXf ml_control;
			ml_control.resize(control_dimension_);
			ml_control.setZero();

			getMachineLearningControl(old_best_[1].state.data(), ml_control.data());

			machine_learning_and_chosen_control_discrepancy_ = (ml_control.tail(control_dimension_) - old_best_[1].control).norm();

		}

		return machine_learning_and_chosen_control_discrepancy_;
	}

	void __stdcall SCAControl::updateResults(int sampleIdx, const float *finalControl, const float *newState, double cost, const float *priorMean, const float *priorStd)
	{
		MarginalSample &nextSample = experience_[next_step_][sampleIdx];


		memcpy(nextSample.state.data(), newState, sizeof(float)*state_dimension_);
		memcpy(nextSample.control.data(), finalControl, sizeof(float)*control_dimension_);


		nextSample.stateCost = cost;
		nextSample.originalStateCostFromClient = cost;
		MarginalSample &previousSample = experience_[current_step_][nextSample.previousMarginalSampleIdx];


		nextSample.controlCost = 0.0f;
		nextSample.fullCost = previousSample.fullCost + nextSample.originalStateCostFromClient + nextSample.controlCost;


	}

	void __stdcall SCAControl::endPlanningStep(int stepIdx)
	{

	}


	void SCAControl::long_term_learning()
	{


		std::deque<std::shared_ptr<TeachingSample>> transition_data;
		std::deque<std::shared_ptr<TeachingSample>> validation_data;
		{
			std::lock_guard<std::mutex> lock(copying_transition_data_);
			transition_data = transitions_;
			validation_data = validation_data_;
		}

		if (transition_data.size() < 100) {
			return;
		}

		unsigned minibatch_size = transition_data.size() / 10;
		minibatch_size = std::min(minibatch_size, (unsigned)100);
		if (minibatch_size == 0) {
			return;
		}


		const unsigned input_dim = state_dimension_;
		unsigned output_dim = control_dimension_;


		if (!actor_in_training_.get()) {
			actor_in_training_ = std::unique_ptr<MultiLayerPerceptron>(new MultiLayerPerceptron());
			init_neural_net(input_dim, output_dim, *actor_in_training_);
		}




		Eigen::VectorXf state_stdev = Eigen::VectorXf::Zero(state_dimension_);

		if (regularization_noise_ > 0.0f)
		{
			Eigen::VectorXf state_mean = Eigen::VectorXf::Zero(state_dimension_);
			int amount = 0;
			for (std::shared_ptr<TeachingSample> sample : transitions_) {

				state_mean += sample->state_;
				amount++;

			}
			state_mean /= (float)amount;

			for (std::shared_ptr<TeachingSample> sample : transitions_) {

				state_stdev += (sample->state_ - state_mean).cwiseAbs2();

			}
			state_stdev /= (float)amount;
			state_stdev = state_stdev.cwiseSqrt();
		}





		std::vector<float*> inputs;
		std::vector<float*> outputs;

		inputs.reserve(transition_data.size());
		outputs.reserve(transition_data.size());




		auto form_training_data = [&](std::deque<std::shared_ptr<TeachingSample>>& data_set) {

			inputs.clear();
			outputs.clear();

			if (regularization_noise_ > 0.0f) {

				Eigen::VectorXf noise = Eigen::VectorXf::Zero(state_dimension_);
				Eigen::VectorXf noise_scaling = state_stdev*regularization_noise_;

				for (const std::shared_ptr<TeachingSample>& datum : data_set) {

					BoxMuller(noise);
					noise.array() *= noise_scaling.array();

					datum->input_for_learner_ = datum->state_ + noise;

					inputs.push_back(datum->input_for_learner_.data());
					outputs.push_back(datum->control_.data());

				}
			}
			else {
				for (const std::shared_ptr<TeachingSample>& datum : data_set) {
					inputs.push_back(datum->state_.data());
					outputs.push_back(datum->control_.data());
				}
			}

		};


		form_training_data(transition_data);


		int max_epochs = 5;
		int epoch = 0;
		while (epoch < max_epochs) {


			actor_in_training_->train_adam((const float**)inputs.data(), (const float**)outputs.data(), inputs.size(), minibatch_size);

			epoch++;
		}


		state_stdev.setZero();
		form_training_data(validation_data);


		float mse = std::numeric_limits<float>::infinity();
		if (actor_in_training_.get()) {
			mse = actor_in_training_->mse((const float**)inputs.data(), (const float**)outputs.data(), inputs.size());

			std::lock_guard<std::mutex> mse_lock(mse_changing_mutex_);
			machine_learning_mse_ = mse;
		}

		if (!(mse >= 0.0f && mse < std::numeric_limits<float>::max())) {

			int input_size = actor_in_training_->input_operation_->size_;
			int output_size = actor_in_training_->output_operation_->size_;

			actor_in_training_ = std::unique_ptr<MultiLayerPerceptron>(new MultiLayerPerceptron);
			init_neural_net(input_size, output_size, *actor_in_training_);
		}


		std::cout << "Trained actor. MSE: " << mse << " Data: " << transition_data.size() << std::endl;

		actor_copy_ = std::unique_ptr<MultiLayerPerceptron>(new MultiLayerPerceptron(*actor_in_training_));


	}




	void SCAControl::long_term_learning_distribution()
	{
		std::deque<std::shared_ptr<TeachingSample>> transition_data;
		std::deque<std::shared_ptr<TeachingSample>> validation_data;
		{
			std::lock_guard<std::mutex> lock(copying_transition_data_);
			transition_data = transitions_;
			validation_data = validation_data_;
		}




		if (transition_data.size() < 100) {
			return;
		}

		unsigned minibatch_size = transition_data.size();
		if (minibatch_size == 0) {
			return;
		}

		unsigned supervised_minibatch_size = minibatch_size / 10;
		supervised_minibatch_size = std::min(supervised_minibatch_size, (unsigned)100);
		if (supervised_minibatch_size == 0) {
			return;
		}




		const unsigned input_dim = state_dimension_ + input_noise_dim_;
		unsigned output_dim = state_dimension_ + control_dimension_;


		if (!actor_in_training_.get()) {
			actor_in_training_ = std::unique_ptr<MultiLayerPerceptron>(new MultiLayerPerceptron());
			init_neural_net(input_dim, output_dim, *actor_in_training_);
		}



		std::vector<float*> data_ptrs;
		data_ptrs.reserve(transition_data.size());





		auto form_training_data = [&](std::deque<std::shared_ptr<TeachingSample>>& data_set) {

			data_ptrs.clear();


			int joint_dim = data_set[0]->state_.size() + data_set[0]->control_.size();

			for (const std::shared_ptr<TeachingSample>& datum : data_set) {

				datum->input_for_learner_.resize(joint_dim);
				datum->input_for_learner_.head(state_dimension_) = datum->state_;
				datum->input_for_learner_.tail(control_dimension_) = datum->control_;

				data_ptrs.push_back(datum->input_for_learner_.data());
			}

		};


		form_training_data(transition_data);

		int max_epochs = 5;
		int epoch = 0;
		while (epoch < max_epochs) {

			std::vector<float> noise_in;
			noise_in.resize(input_noise_dim_);
			std::vector<std::vector<float>> noise_vectors;

			for (int i = 0; i < data_ptrs.size(); ++i) {
				sample_noise(noise_in.data(), input_noise_dim_);
				noise_vectors.push_back(noise_in);
			}

			std::vector<float*> noise_ptrs = vector_to_ptrs(noise_vectors);
			actor_in_training_->incremental_matching((const float**)noise_ptrs.data(), (const float**)data_ptrs.data(), state_dimension_, input_noise_dim_, data_ptrs.size(), minibatch_size, supervised_minibatch_size);

			epoch++;
		}


#if 1
		form_training_data(validation_data);


		float mse = std::numeric_limits<float>::infinity();
		if (actor_in_training_.get()) {
			std::vector<float> noise_in;
			noise_in.resize(input_noise_dim_);
			std::vector<std::vector<float>> noise_vectors;

			for (int i = 0; i < data_ptrs.size(); ++i) {
				sample_noise(noise_in.data(), input_noise_dim_);
				noise_vectors.push_back(noise_in);
			}

			std::vector<float*> noise_ptrs = vector_to_ptrs(noise_vectors);
			mse = actor_in_training_->incremental_matching_error((const float**)noise_ptrs.data(), (const float**)data_ptrs.data(), state_dimension_, input_noise_dim_, data_ptrs.size());


			std::lock_guard<std::mutex> mse_lock(mse_changing_mutex_);
			machine_learning_mse_ = mse;
		}


		if (!(mse >= 0.0f && mse < std::numeric_limits<float>::max())) {

			int input_size = actor_in_training_->input_operation_->size_;
			int output_size = actor_in_training_->output_operation_->size_;

			actor_in_training_ = std::unique_ptr<MultiLayerPerceptron>(new MultiLayerPerceptron);
			init_neural_net(input_size, output_size, *actor_in_training_);
		}
#endif


		std::cout << "Trained distribution actor. MSE: " << mse << " Data: " << transition_data.size() << std::endl;

		actor_copy_ = std::unique_ptr<MultiLayerPerceptron>(new MultiLayerPerceptron(*actor_in_training_));

	}



	std::deque<std::string> SCAControl::get_settings()
	{
		std::deque<std::string> settings;

		settings.push_back("Number of trajectories: " + std::to_string(amount_samples_));
		settings.push_back("Number of time steps: " + std::to_string(steps_));
		settings.push_back("Learning: " + std::to_string(learning_));

		switch (sampling_mode_)
		{
		case AaltoGames::SCAControl::PRODUCT_OF_GAUSSIANS:
			settings.push_back("Sampling scheme: Product of distributions");
			break;
		case AaltoGames::SCAControl::MINIMIZING_INFORMATION_LOSS:
			settings.push_back("Sampling scheme: Maximum entropy");
			break;
		default:
			settings.push_back("Sampling scheme: Maximum entropy");
			break;
		}

		settings.push_back("Number nearest neighbor trajectories: " + std::to_string(nn_trajectories_));
		settings.push_back("Number machine learning trajectories: " + std::to_string(machine_learning_samples_));
		settings.push_back("Number zero samples: " + std::to_string(zero_samples_));
		settings.push_back("Number of noisy machine learning trajectories: " + std::to_string(noisy_machine_learning_samples_));
		settings.push_back("Input noise dimension: " + std::to_string(input_noise_dim_));

		switch (network_type_)
		{
		case AaltoGames::SCAControl::ELU:
			settings.push_back("Network type: ELU");
			break;
		case AaltoGames::SCAControl::BSELU:
			settings.push_back("Network type: BSELU");
			break;
		default:
			settings.push_back("Network type: BSELU");
			break;
		}

		settings.push_back("Hidden layers in neural network: " + std::to_string(amount_network_layers_));

		settings.push_back("Use machine learning: " + std::to_string(use_machine_learning_));
		settings.push_back("Regularization noise: " + std::to_string(regularization_noise_));
		settings.push_back("Drop out stdev: " + std::to_string(drop_out_stdev_));
		settings.push_back("Validation fraction: " + std::to_string(validation_fraction_));

		settings.push_back("Learning budget: " + std::to_string(learning_budget_));

		settings.push_back("Amount recently used linear search samples: " + std::to_string(amount_recent_));
		settings.push_back("Use nearest neighbor forests: " + std::to_string(use_forests_));
		settings.push_back("Amount of data in nearest neighbor trees: " + std::to_string(amount_data_in_tree_));
		settings.push_back("Number of trees in forest: " + std::to_string(ann_forest_.forest_.size()));

		settings.push_back("Stored previous frame sample percentage: " + std::to_string(stored_sample_percentage_));
		settings.push_back("Free particle amount: " + std::to_string(no_prior_trajectory_portion_));
		settings.push_back("Resample threshold: " + std::to_string(resample_threshold_));

		return settings;
	}

	void __stdcall SCAControl::endIteration()
	{



		{

			best_cost_ = DBL_MAX;
			int bestIdx = 0;
			for (int i = 0; i < amount_samples_; i++)
			{
				MarginalSample &sample = experience_[steps_][i];
				sample.costToGo = 0;
				if (sample.fullCost < best_cost_)
				{
					best_cost_ = sample.fullCost;
					bestIdx = i;
				}
				sample.fullSampleIdx = i;
			}


			double maxCost = FLT_MAX / 100.0f / (float)(steps_ + 1); //can't use DBL_MAX, as the costs might get summed and result in INF
			for (int step = steps_ - 1; step >= 0; step--)
			{
				//backward propagation
				for (int i = 0; i < amount_samples_; i++)
				{
					MarginalSample &sample = experience_[step][i];
					sample.costToGo = maxCost;
					sample.fullSampleIdx = -1;  //denotes that the trajectory has been pruned
				}

				for (int nextIdx = 0; nextIdx < amount_samples_; nextIdx++)
				{
					MarginalSample &nextSample = experience_[step + 1][nextIdx];
					MarginalSample &sample = experience_[step][nextSample.previousMarginalSampleIdx];

					//fullCost + costToGo should always sum to the full cost at the step==steps_
					sample.costToGo = _min(sample.costToGo, nextSample.costToGo + nextSample.stateCost + nextSample.controlCost);
					sample.fullSampleIdx = nextSample.fullSampleIdx;
				}
			}


			old_best_.resize(steps_ + 1);
			//recover the best trajectory, store for using at next iteration
			best_full_sample_idx_ = bestIdx;	//index of the last marginal sample of the best full sample			
			for (int step = steps_; step >= 0; step--)
			{
				old_best_[step] = experience_[step][bestIdx];
				bestIdx = experience_[step][bestIdx].previousMarginalSampleIdx;
			}
		}


		///////////////////////////////////////////////////////////////////////////


		//Store the stored_sample_percentage_ samples to guide sampling in the next frame
		{
			previous_experience_.resize(steps_ + 1);
			for (int step = 0; step <= steps_; step++)
			{
				int nPruned = ((int)(experience_[step].size()*stored_sample_percentage_)) / 100;
				if (nPruned != (int)previous_experience_[step].size())
					previous_experience_[step].resize(nPruned);
			}


			std::vector<int> best_index(experience_[0].size(), 0);
			for (unsigned i = 0; i < best_index.size(); i++) {
				best_index[i] = i;
			}

			auto cost_sorting_lambda = [&](int i, int j) {

				int last_index = experience_.size() - 1;

				if (experience_[last_index][i].fullCost < experience_[last_index][j].fullCost) {
					return true;
				}
				else {
					return false;
				}

			};

			std::sort(best_index.begin(), best_index.end(), cost_sorting_lambda);

			for (unsigned sample = 0; sample < previous_experience_[0].size(); sample++) {

				int back_track_sample = best_index[sample];

				for (int step = experience_.size() - 1; step >= 0; step--) {

					previous_experience_[step][sample] = experience_[step][back_track_sample];
					previous_experience_[step][sample].previousMarginalSampleIdx = sample;
					back_track_sample = experience_[step][back_track_sample].previousMarginalSampleIdx;

				}

			}


		}

		///////////////////////////////////////////////////////////////////////////

		//Learning
		if (learning_ && time_advanced_)
		{



			//Form the teaching sample
			std::unique_ptr<TeachingSample> teachingSample = std::unique_ptr<TeachingSample>(new TeachingSample(old_best_[1]));



			if (use_forests_) {
				bool add_samples_to_forest = true;
				if (add_samples_to_forest) {
					ann_forest_.add_sample(number_of_hyperplane_tries_, number_of_data_in_leaf_, 12, *teachingSample);
					adding_buffer_.push_back(*teachingSample);
				}
			}



			recent_samples_.push_back(std::unique_ptr<TeachingSample>(new TeachingSample(*teachingSample)));
			while ((int)recent_samples_.size() > amount_recent_) {
				recent_samples_.pop_front();
			}



			//if tree  rebuilder thread finished, launch another one
			const bool rebuild_tree = true;
			if (rebuild_tree && use_forests_)
			{

				if (building_forest_.wait_for(std::chrono::microseconds(1)) == std::future_status::ready) {


					for (TeachingSample& sample : adding_buffer_) {
						tree_under_construction_.add_sample(number_of_hyperplane_tries_, number_of_data_in_leaf_, 40, sample);
					}
					adding_buffer_.clear();

					int swap_idx = rand() % ann_forest_.forest_.size();
					ann_forest_.forest_[swap_idx].root_.swap(tree_under_construction_.root_);

					auto rebuild_tree = [&] {
						std::clock_t building_start_time = std::clock();

						tree_under_construction_.rebuild_tree(number_of_hyperplane_tries_, number_of_data_in_leaf_, amount_data_in_tree_);

						std::clock_t build_time = std::clock() - building_start_time;
						float build_time_sec = (float)build_time / (float)CLOCKS_PER_SEC;
					};

					building_forest_ = std::async(std::launch::async, rebuild_tree);
				}
			}



			transitions_buffer_.push_back(std::move(teachingSample));



			if (use_machine_learning_) {

				if (training_neural_net_.wait_for(std::chrono::microseconds(1)) == std::future_status::ready) {

					auto train_function = [&]() {
						if (input_noise_dim_ == 0) {
							long_term_learning();
						}
						else {
							long_term_learning_distribution();
						}
					};


					for (std::shared_ptr<TeachingSample> sample : transitions_buffer_) {
						float u = sampleUniform<float>();

						if (u < validation_fraction_) {
							validation_data_.push_back(sample);
						}
						else {

							transitions_.push_back(sample);
						}

					}

					while (transitions_.size() > learning_budget_) {
						transitions_.pop_front();
					}

					while (validation_data_.size() > learning_budget_) {
						validation_data_.pop_front();
					}

					transitions_buffer_.clear();

					if (actor_copy_.get()) {
						actor_.swap(actor_copy_);
						actor_copy_.reset();
					}

					training_neural_net_ = std::async(std::launch::async, train_function);
				}

			}

		}


		iteration_idx_++;
	}

	void SCAControl::setParams(float resampleThreshold, bool learning, int nTrajectories)
	{
		resample_threshold_ = resampleThreshold;
		learning_ = learning;
		amount_samples_ = nTrajectories;
	}

	void __stdcall SCAControl::getBestControl(int timeStep, float *out_control)
	{
		//the +1 because old_best_ stores marginal states which store "incoming" controls instead of "outgoing"
		memcpy(out_control, old_best_[timeStep + 1].control.data(), control_dimension_ * sizeof(float));
	}

	const float* SCAControl::getRecentControl(float * state, float * out_control, int thread)
	{

		Eigen::Map<Eigen::VectorXf> state_vec(state, state_dimension_);
		Eigen::Map<Eigen::VectorXf> control_out_vec(out_control, control_dimension_);
		control_out_vec.setZero();


		TeachingSample* teaching_sample = nullptr;
		if (use_forests_) {
			TeachingSample& key = keys_[thread];
			key.key_vector_ = Eigen::Map<Eigen::VectorXf>(state, state_dimension_);
			teaching_sample = ann_forest_.get_approximate_nearest(key);
		}


		float dist = std::numeric_limits<float>::infinity();
		if (teaching_sample) {
			dist = (teaching_sample->state_ - state_vec).cwiseAbs().sum();
		}


		for (std::unique_ptr<TeachingSample>& sample : recent_samples_) {

			float current_dist = (sample->state_ - state_vec).cwiseAbs().sum();

			if (current_dist < dist) {
				dist = current_dist;
				teaching_sample = sample.get();
			}

		}

		if (teaching_sample) {
			control_out_vec = teaching_sample->control_;
			return teaching_sample->state_.data();
		}

		return nullptr;
	}

	void __stdcall SCAControl::getBestControlState(int timeStep, float *out_state)
	{
		//the +1 because old_best_ stores marginal states which store "incoming" controls instead of "outgoing"
		memcpy(out_state, old_best_[timeStep + 1].state.data(), state_dimension_ * sizeof(float));
	}

	double __stdcall SCAControl::getBestTrajectoryOriginalStateCost(int timeStep)
	{
		//the +1 because old_best_ stores marginal states which store "incoming" controls instead of "outgoing"
		return old_best_[timeStep + 1].originalStateCostFromClient;
	}

	int __stdcall SCAControl::getPreviousSampleIdx(int sampleIdx, int timeStep)
	{

		if (timeStep < 0) {
			timeStep = next_step_;
		}

		if ((int)experience_.size() > timeStep && (int)experience_[timeStep].size() > sampleIdx) {
			MarginalSample &nextSample = experience_[timeStep][sampleIdx];
			return nextSample.previousMarginalSampleIdx;
		}
		else {
			return 0;
		}
	}

	void __stdcall SCAControl::getAssumedStartingState(int sampleIdx, float *out_state) {
		MarginalSample &nextSample = experience_[next_step_][sampleIdx];
		MarginalSample& currentSample = experience_[current_step_][nextSample.previousMarginalSampleIdx];

		if (out_state) {
			for (int i = 0; i < currentSample.state.size(); i++) {
				out_state[i] = currentSample.state[i];
			}
		}
	}

	bool __stdcall SCAControl::getMachineLearningControl(float *state, float* out_control) {

		std::lock_guard<std::mutex> lock(actor_mutex_);

		if (!actor_.get()) {
			for (int i = 0; i < control_dimension_; i++) {
				out_control[i] = 0.0f;
			}
			return false;
		}


		if (input_noise_dim_ == 0) {

			actor_->run(state);

			for (int i = 0; i < control_dimension_; i++) {

				float& output = actor_->output_operation_->outputs_[i];

				if (output - output != output - output) {
					output = 0.0f;
				}

				output = std::min(output, control_max_[i]);
				output = std::max(output, control_min_[i]);

				out_control[i] = output;
			}

		}
		else {



			float input[1000];

			int idx = 0;
			for (int i = 0; i < state_dimension_; ++i) {
				input[idx] = state[i];
				++idx;
			}

			sample_noise(&input[idx], input_noise_dim_);

			actor_->run(input);

			idx = state_dimension_;
			for (int i = 0; i < control_dimension_; ++i) {

				float& output = actor_->output_operation_->outputs_[idx];

				if (output - output != output - output) {
					output = 0.0f;
				}

				output = std::min(output, control_max_[i]);
				output = std::max(output, control_min_[i]);

				out_control[i] = output;
				++idx;
			}

		}





		return true;

	}

	int SCAControl::get_up_to_k_nearest(float * state, int k, int thread, SCAControl::TeachingSample** samples)
	{
		Eigen::Map<Eigen::VectorXf> state_vec(state, state_dimension_);
		for (int i = 0; i < k; i++) {
			samples[i] = nullptr;
		}


		int found = 0;
		if (use_forests_) {
			TeachingSample& key = keys_[thread];
			key.state_ = Eigen::Map<Eigen::VectorXf>(state, state_dimension_);
			found = ann_forest_.get_up_to_k_nearest(key.state_, samples, k);
		}

		auto is_closer = [&](TeachingSample* first, TeachingSample* second) {


			if (!first && !second) {
				return false;
			}
			if (first && !second) {
				return true;
			}
			if (!first && second) {
				return false;
			}


			float dist_first = (first->state_ - state_vec).norm();
			float dist_second = (second->state_ - state_vec).norm();

			if (dist_first < dist_second) {
				return true;
			}

			return false;

		};

		auto remove_duplicates = [&]() {
			for (int i = 0; i < k; i++) {
				for (int j = i + 1; j < k; j++) {
					if (samples[i] && samples[j]) {
						if (*samples[i] == *samples[j]) {
							samples[j] = nullptr;
						}
					}
				}
			}
		};

		for (std::unique_ptr<TeachingSample>& sample : recent_samples_) {

			if (is_closer(sample.get(), samples[k - 1])) {

				remove_duplicates();

				for (int i = 0; i < k; i++) {
					if (!samples[i]) {
						samples[i] = sample.get();
						break;
					}
				}

				std::sort(&samples[0], &samples[k], is_closer);
			}

		}

		remove_duplicates();
		std::sort(&samples[0], &samples[k], is_closer);

		int num_samples = 0;
		while (samples[num_samples] != nullptr && num_samples < k) {
			num_samples++;
		}

		return num_samples;
	}


	void form_kl_distribution(SCAControl::DiagonalGaussian& compromise, std::vector<SCAControl::DiagonalGaussian>& distributions, int amount_dists, float* probs = nullptr) {

		compromise.first.setZero();
		compromise.second.setZero();

		const int max_distributions_ever = 100;
		float tmp_array[max_distributions_ever];


		float weight_tmp = 1.0f / (float)amount_dists;
		if (probs) {
			float wSum = 0.0f;

			for (int i = 0; i < amount_dists; i++) {
				wSum += probs[i];
			}

			bool all_clear = true;
			for (int i = 0; i < amount_dists; i++) {
				probs[i] /= wSum;
				all_clear &= valid_float(probs[i]);
			}

			if (!all_clear) {
				for (int i = 0; i < amount_dists; i++) {
					probs[i] = weight_tmp;
				}
			}


		}
		else {
			probs = tmp_array;
			for (int i = 0; i < amount_dists; i++) {
				probs[i] = weight_tmp;
			}
		}

		//Mean
		for (int i = 0; i < amount_dists; i++) {
			const float& weight = probs[i];
			compromise.first += weight*distributions[i].first;
		}

		//Stdev
		for (int i = 0; i < amount_dists; i++) {
			const float& weight = probs[i];
			Eigen::VectorXf& stdev = compromise.second;
			int dims = stdev.size();
			for (int dim = 0; dim < dims; dim++) {

				const float& sigma = distributions[i].second[dim];
				const float& mu = distributions[i].first[dim];

				float diff = compromise.first[dim] - mu;

				stdev[dim] += weight*(diff*diff + sigma*sigma);

			}
		}

		Eigen::VectorXf& stdev = compromise.second;
		int dims = stdev.size();
		for (int dim = 0; dim < dims; dim++) {
			stdev[dim] = std::sqrt(stdev[dim]);
		}

	}


	void __stdcall SCAControl::getControl(int sampleIdx, float *out_control, const float *priorMean, const float *priorStd)
	{
		switch (sampling_mode_)
		{
		case AaltoGames::SCAControl::PRODUCT_OF_GAUSSIANS:
			getControlL1(sampleIdx, out_control, priorMean, priorStd);
			break;
		case AaltoGames::SCAControl::MINIMIZING_INFORMATION_LOSS:
			getControlKL(sampleIdx, out_control, priorMean, priorStd);
			break;
		default:
			getControlKL(sampleIdx, out_control, priorMean, priorStd);
			break;
		}
	}

	void SCAControl::getControlL1(int sampleIdx, float * out_control, const float * priorMean, const float * priorStd)
	{
		//link the marginal samples to each other so that full-dimensional samples can be recovered
		MarginalSample &nextSample = experience_[next_step_][sampleIdx];
		MarginalSample &currentSample = experience_[current_step_][nextSample.previousMarginalSampleIdx];
		Eigen::VectorXf& control = nextSample.control;

		bool processed = false;
		//special processing for old best trajectory
		if (nextSample.particleRole == ParticleRole::OLD_BEST)
		{
			if (next_step_ < (int)old_best_.size()) {

				//next_step_ as index because the control is always stored to the next sample ("control that brought me to this state")
				control = old_best_[next_step_].control;

				processed = true;
			}
			else {
				nextSample.particleRole = ParticleRole::MACHINE_LEARNING;
			}
		}

		if (nextSample.particleRole == ParticleRole::MACHINE_LEARNING_NO_VARIATION || nextSample.particleRole == ParticleRole::MACHINE_LEARNING_NO_RESAMPLING)
		{
			if (!actor_.get()) {
				nextSample.particleRole = ParticleRole::MACHINE_LEARNING;
			}
			else {
				getMachineLearningControl(currentSample.state.data(), control.data());
				processed = true;
			}
		}

		if (nextSample.particleRole == ParticleRole::ZERO_SAMPLE)
		{
			control = static_prior_.mean[0];
			processed = true;
		}




		if (!processed)
		{


			if (nextSample.particleRole != ParticleRole::MACHINE_LEARNING &&
				nextSample.particleRole != ParticleRole::NEAREST_NEIGHBOR &&
				nextSample.particleRole != ParticleRole::PREVIOUS_FRAME_PRIOR &&
				nextSample.particleRole != ParticleRole::FREE) {
				nextSample.particleRole = ParticleRole::MACHINE_LEARNING;
			}


			if (nextSample.particleRole == ParticleRole::MACHINE_LEARNING) {
				nextSample.previous_frame_prior_ = true;
				nextSample.nearest_neighbor_prior_ = false;
				nextSample.machine_learning_prior_ = true;
				if (!actor_.get()) {
					nextSample.particleRole = ParticleRole::NEAREST_NEIGHBOR;
				}
			}

			if (nextSample.particleRole == ParticleRole::NEAREST_NEIGHBOR) {
				nextSample.previous_frame_prior_ = true;
				nextSample.nearest_neighbor_prior_ = true;
				nextSample.machine_learning_prior_ = false;
				if (!use_forests_) {
					nextSample.particleRole = ParticleRole::PREVIOUS_FRAME_PRIOR;
				}
			}

			if (nextSample.particleRole == ParticleRole::PREVIOUS_FRAME_PRIOR) {
				nextSample.previous_frame_prior_ = true;
				nextSample.nearest_neighbor_prior_ = false;
				nextSample.machine_learning_prior_ = false;
			}

			if (nextSample.particleRole == ParticleRole::FREE) {
				nextSample.previous_frame_prior_ = false;
				nextSample.nearest_neighbor_prior_ = false;
				nextSample.machine_learning_prior_ = false;
			}



			DiagonalGMM *proposal = proposals_[sampleIdx].get();
			DiagonalGMM *prior = priors_[sampleIdx].get();


			prior->mean[0] = currentSample.control;
			prior->std[0] = control_diff_std_;

			//multiply the difference prior and static prior to yield the proposal
			if (current_step_ == 0 && (!time_advanced_ || iteration_idx_ == 0))
			{
				proposal->copyFrom(static_prior_);
			}
			else
			{
				DiagonalGMM::multiply(static_prior_, *prior, *proposal);
			}


			//the optional prior passed as argument
			if (priorMean != NULL && priorStd != NULL)
			{

				Eigen::Map<const Eigen::VectorXf> externPriorMeanMap(priorMean, control_dimension_);
				Eigen::Map<const Eigen::VectorXf> externPriorStdMap(priorStd, control_dimension_);

				prior->mean[0] = externPriorMeanMap;
				prior->std[0] = externPriorStdMap;

				for (int dim = 0; dim < control_dimension_; dim++) {
					float& num = prior->mean[0][dim];

					if (num < control_min_[dim]) {
						num = control_min_[dim];
					}

					if (num > control_max_[dim]) {
						num = control_max_[dim];
					}

				}

				DiagonalGMM::multiply(*prior, *proposal, *proposal);	//ok to have same source and destination if both have only 1 gaussian
			}




			const MarginalSample* nearest = nullptr;
			float nearest_dist = std::numeric_limits<float>::infinity();

			//use previous frame trajectories as a prior. 
			if (nextSample.previous_frame_prior_ && next_step_ < previous_experience_.size()) {

				int search_time = next_step_;
				const std::vector<MarginalSample>& marginal = previous_experience_[search_time];

				for (unsigned i = 0; i < marginal.size(); i++) {
					float current_dist = (marginal[i].previousState - currentSample.state).cwiseAbs().sum();

					if (current_dist < nearest_dist) {
						nearest_dist = current_dist;
						nearest = &marginal[i];
					}

				}

			}


			const float* recent_state = nullptr;

			if (nextSample.nearest_neighbor_prior_) {
				recent_state = getRecentControl(currentSample.state.data(), control.data(), sampleIdx);
			}

			if (recent_state) {


				const bool force_multiply = false;


				if (force_multiply) {
					for (int i = 0; i < control.size(); i++) {
						productNormalDist(nearest->control[i], control_variation_std_[i], proposal->mean[0](i), proposal->std[0](i), proposal->mean[0](i), proposal->std[0](i));
						productNormalDist(control[i], control_variation_std_[i], proposal->mean[0](i), proposal->std[0](i), proposal->mean[0](i), proposal->std[0](i));
					}
				}
				else {
					Eigen::Map<const Eigen::VectorXf> state_vec(recent_state, state_dimension_);

					float current_dist = (state_vec - currentSample.state).cwiseAbs().sum();
					if (current_dist < nearest_dist) {

						for (int i = 0; i < control.size(); i++) {
							productNormalDist(control[i], control_variation_std_[i], proposal->mean[0](i), proposal->std[0](i), proposal->mean[0](i), proposal->std[0](i));
						}

					}
					else {
						if (nearest) {
							for (int i = 0; i < nearest->control.size(); i++) {
								productNormalDist(nearest->control[i], control_variation_std_[i], proposal->mean[0](i), proposal->std[0](i), proposal->mean[0](i), proposal->std[0](i));
							}
						}
					}
				}

			}
			else {
				if (nearest) {
					for (int i = 0; i < nearest->control.size(); i++) {
						productNormalDist(nearest->control[i], control_variation_std_[i], proposal->mean[0](i), proposal->std[0](i), proposal->mean[0](i), proposal->std[0](i));
					}
				}
			}



			if (nextSample.machine_learning_prior_)
			{

				getMachineLearningControl(currentSample.state.data(), control.data());

				for (int i = 0; i < control.size(); i++) {
					productNormalDist(control[i], control_variation_std_[i], proposal->mean[0](i), proposal->std[0](i), proposal->mean[0](i), proposal->std[0](i));
				}

			}


			for (int i = 0; i < control.size(); i++) {
				control[i] = (float)sample_clipped_gaussian(proposal->mean[0](i), proposal->std[0](i), control_min_(i), control_max_(i));
			}


		}


		//Clamping the control to the bounds
		for (int i = 0; i < control.size(); i++) {
			control[i] = std::min(std::max(control[i], control_min_[i]), control_max_[i]);
			out_control[i] = control[i];
		}


		nextSample.previousState = experience_[current_step_][nextSample.previousMarginalSampleIdx].state;
		nextSample.previousControl = experience_[current_step_][nextSample.previousMarginalSampleIdx].control;
		nextSample.previousPreviousControl = experience_[current_step_][nextSample.previousMarginalSampleIdx].previousControl;

		if (current_step_ > 0)
		{
			MarginalSample &previousSample = experience_[current_step_ - 1][currentSample.previousMarginalSampleIdx];
			nextSample.previousPreviousState = previousSample.state;
		}
	}

	void SCAControl::getControlKL(int sampleIdx, float * out_control, const float * priorMean, const float * priorStd)
	{
		MarginalSample &nextSample = experience_[next_step_][sampleIdx];
		MarginalSample &currentSample = experience_[current_step_][nextSample.previousMarginalSampleIdx];
		Eigen::VectorXf& control = nextSample.control;


		nextSample.previousState = currentSample.state;
		nextSample.previousPreviousState = currentSample.previousState;

		nextSample.previousControl = currentSample.control;
		nextSample.previousPreviousControl = currentSample.previousControl;


		bool processed = false;
		//special processing for old best trajectory
		if (nextSample.particleRole == ParticleRole::OLD_BEST)
		{
			if (next_step_ < (int)old_best_.size()) {
				//next_step_ as index because the control is always stored to the next sample ("control that brought me to this state")
				control = old_best_[next_step_].control;

				processed = true;
			}
			else {
				nextSample.particleRole = ParticleRole::MACHINE_LEARNING;
			}
		}

		if (nextSample.particleRole == ParticleRole::MACHINE_LEARNING_NO_VARIATION || nextSample.particleRole == ParticleRole::MACHINE_LEARNING_NO_RESAMPLING)
		{
			if (!actor_.get()) {
				nextSample.particleRole = ParticleRole::MACHINE_LEARNING;
			}
			else {
				getMachineLearningControl(currentSample.state.data(), control.data());
				processed = true;
			}
		}

		if (nextSample.particleRole == ParticleRole::ZERO_SAMPLE)
		{
			control = static_prior_.mean[0];
			processed = true;
		}



		if (!processed)
		{

			if (nextSample.particleRole != ParticleRole::MACHINE_LEARNING &&
				nextSample.particleRole != ParticleRole::NEAREST_NEIGHBOR &&
				nextSample.particleRole != ParticleRole::PREVIOUS_FRAME_PRIOR &&
				nextSample.particleRole != ParticleRole::FREE) {
				nextSample.particleRole = ParticleRole::MACHINE_LEARNING;
			}


			if (nextSample.particleRole == ParticleRole::MACHINE_LEARNING) {
				nextSample.previous_frame_prior_ = true;
				nextSample.nearest_neighbor_prior_ = true;
				nextSample.machine_learning_prior_ = true;
				if (!actor_.get()) {
					nextSample.particleRole = ParticleRole::NEAREST_NEIGHBOR;
				}
			}

			if (nextSample.particleRole == ParticleRole::NEAREST_NEIGHBOR) {
				nextSample.previous_frame_prior_ = true;
				nextSample.nearest_neighbor_prior_ = true;
				nextSample.machine_learning_prior_ = false;
				if (!use_forests_) {
					nextSample.particleRole = ParticleRole::PREVIOUS_FRAME_PRIOR;
				}
			}

			if (nextSample.particleRole == ParticleRole::PREVIOUS_FRAME_PRIOR) {
				nextSample.previous_frame_prior_ = true;
				nextSample.nearest_neighbor_prior_ = false;
				nextSample.machine_learning_prior_ = false;
			}

			if (nextSample.particleRole == ParticleRole::FREE) {
				nextSample.previous_frame_prior_ = false;
				nextSample.nearest_neighbor_prior_ = false;
				nextSample.machine_learning_prior_ = false;
			}


			const bool zero_variances = true;


			float probs[1000];
			int distribution_idx = 0;

			int previous_time_distribution = -1;

			if (!old_best_valid_)
			{
				gaussian_distributions_[sampleIdx][distribution_idx].first = static_prior_.mean[0];
				gaussian_distributions_[sampleIdx][distribution_idx].second = static_prior_.std[0];
				probs[distribution_idx] = 1.0f;
				distribution_idx++;
			}
			else {

				gaussian_distributions_[sampleIdx][distribution_idx].first = currentSample.control;
				gaussian_distributions_[sampleIdx][distribution_idx].second = control_variation_std_;
				probs[distribution_idx] = 1.0f;
				previous_time_distribution = distribution_idx;
				distribution_idx++;

			}


			//the optional prior passed as argument
			if (priorMean != NULL && priorStd != NULL)
			{
				Eigen::VectorXf& mean = gaussian_distributions_[sampleIdx][distribution_idx].first;
				Eigen::VectorXf& std = gaussian_distributions_[sampleIdx][distribution_idx].second;
				probs[distribution_idx] = 0.2f;
				distribution_idx++;

				for (int dim = 0; dim < control_dimension_; dim++) {
					mean[dim] = priorMean[dim];
					std[dim] = priorStd[dim];
				}

				if (zero_variances) {
					std.setZero();
				}
			}

			const MarginalSample* nearest = nullptr;
			float nearest_dist = std::numeric_limits<float>::infinity();
			//use previous frame trajectories as prior. 
			if (nextSample.previous_frame_prior_) {

				if (previous_experience_.size() > next_step_) {
					const std::vector<MarginalSample>& marginal = previous_experience_[next_step_];
					float current_dist = 0.0f;

					for (unsigned i = 0; i < marginal.size(); i++) {
						current_dist = (marginal[i].previousState - currentSample.state).cwiseAbs().sum();

						if (current_dist < nearest_dist) {
							nearest_dist = current_dist;
							nearest = &marginal[i];
						}

					}



					if (nearest) {
						gaussian_distributions_[sampleIdx][distribution_idx].first = nearest->control;
						gaussian_distributions_[sampleIdx][distribution_idx].second = control_variation_std_;
						probs[distribution_idx] = 5.0f;

						if (zero_variances) {
							gaussian_distributions_[sampleIdx][distribution_idx].second.setZero();
						}

						distribution_idx++;



					}
				}
			}


			if (nextSample.nearest_neighbor_prior_) {

				const int k = 3;
				TeachingSample* samples[k + 1];
				int num_samples = get_up_to_k_nearest(currentSample.state.data(), k, sampleIdx, samples);


				if (num_samples > 0) {


					for (int idx = 0; idx < num_samples; idx++) {
						const TeachingSample* sample = samples[idx];

#if 0
						float current_dist = (sample->state_ - currentSample.state).norm();

						if (current_dist > nearest_dist) {
							continue;
						}
#endif

						gaussian_distributions_[sampleIdx][distribution_idx].first = sample->control_;
						gaussian_distributions_[sampleIdx][distribution_idx].second = control_variation_std_;
						probs[distribution_idx] = 5.0f;

						if (zero_variances) {
							gaussian_distributions_[sampleIdx][distribution_idx].second.setZero();
						}

						distribution_idx++;
					}


					if (zero_variances) {
						for (int dist_idx = 0; dist_idx < distribution_idx; dist_idx++) {
							gaussian_distributions_[sampleIdx][dist_idx].second.setZero();
						}
					}

				}

			}


			if (nextSample.machine_learning_prior_)
			{
				bool control_available = false;
				control_available = getMachineLearningControl(currentSample.state.data(), control.data());

				gaussian_distributions_[sampleIdx][distribution_idx].first = control;
				gaussian_distributions_[sampleIdx][distribution_idx].second = control_variation_std_;


				if (control_available) {

					probs[distribution_idx] = 10.0f;
					distribution_idx++;
				}




				if (zero_variances) {
					for (int dist_idx = 0; dist_idx < distribution_idx; dist_idx++) {
						gaussian_distributions_[sampleIdx][dist_idx].second.setZero();
					}
				}



			}


			form_kl_distribution(sampling_distributions_[sampleIdx], gaussian_distributions_[sampleIdx], distribution_idx, probs);


			for (int i = 0; i < control.size(); i++) {
				control[i] = (float)sample_clipped_gaussian(sampling_distributions_[sampleIdx].first[i], sampling_distributions_[sampleIdx].second[i], control_min_(i), control_max_(i));
			}

		}



		//Clamping the control to the bounds
		for (int i = 0; i < control.size(); i++) {

			float& control_val = control[i];

			if (!(control_val > std::numeric_limits<float>::lowest() && control_val < std::numeric_limits<float>::max())) {
				control_val = 0.0f;
			}

			control[i] = std::min(std::max(control[i], control_min_[i]), control_max_[i]);
			out_control[i] = control[i];
		}


		nextSample.previousState = experience_[current_step_][nextSample.previousMarginalSampleIdx].state;
		nextSample.previousControl = experience_[current_step_][nextSample.previousMarginalSampleIdx].control;
		nextSample.previousPreviousControl = experience_[current_step_][nextSample.previousMarginalSampleIdx].previousControl;

		if (current_step_ > 0)
		{
			MarginalSample &previousSample = experience_[current_step_ - 1][currentSample.previousMarginalSampleIdx];
			nextSample.previousPreviousState = previousSample.state;
		}
	}



	int SCAControl::getNumTrajectories()
	{
		return amount_samples_;
	}


	int SCAControl::getNumSteps()
	{
		return steps_;
	}

	void SCAControl::restart()
	{
		iteration_idx_ = 0;
	}


	SCAControl::TeachingSample::TeachingSample(const MarginalSample & marginal_sample)
	{
		state_ = marginal_sample.previousState;
		future_state_ = marginal_sample.state;
		control_ = marginal_sample.control;

		cost_to_go_ = 0.0f;
		instantaneous_cost_ = (float)marginal_sample.originalStateCostFromClient + (float)marginal_sample.controlCost;

		key_vector_ = state_;

		state_control_ = Eigen::VectorXf::Zero(state_.size() + control_.size());
		state_control_.head(state_.size()) = state_;
		state_control_.tail(control_.size()) = control_;

		input_for_learner_ = state_;

	}

	void sample_noise(float* noise, const int noise_dim)
	{
		//Uniform noise from -1 to 1.
		for (int i = 0; i < noise_dim; ++i) {
			noise[i] = (float)rand() / (float)RAND_MAX;
			noise[i] *= 2.0f;
			noise[i] += -1.0f;
		}
	}

} //AaltoGames;
