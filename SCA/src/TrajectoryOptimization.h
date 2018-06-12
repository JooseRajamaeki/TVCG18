#ifndef TRAJECTORY_OPTIMIZATION_H
#define TRAJECTORY_OPTIMIZATION_H
#include <Eigen/Eigen> 
#include <deque>

namespace AaltoGames
{

#ifdef SWIG
#define __stdcall  //SWIG doesn't understand __stdcall, but fortunately c# assumes it for virtual calls
#endif

	enum ParticleRole{
		OLD_BEST, POLICY, MACHINE_LEARNING_NO_VARIATION, GAUSSIAN_BACK_PROB, VALIDATION, PREVIOUS_FRAME_PRIOR, PREVIOUS_FRAME_PRIOR_WITH_SMOOTHING, SMOOTHED, FREE, INFORMED, INFORMED_WITH_VARIATION, PREVIOUS_FRAME_CONTROL, MACHINE_LEARNING, SPLICED, MACHINE_LEARNING_SMOOTHED, MACHINE_LEARNING_NO_RESAMPLING, MACHINE_LEARNING_FORWARD, MACHINE_LEARNING_BACKWARD, NEAREST_NEIGHBOR, NEAREST_NEIGHBOR_AND_MACHINE_LEARNING, DEBUG_ML, POLICY_SEARCH, REINFORCEMENT_LEARNING, EVOLUTIONARY, OLD_ACTOR_NO_VARIATION, ZERO_SAMPLE
	};

	class ITrajectoryOptimization
	{
	public:
		//returns total cost, including both control and state cost
		virtual double __stdcall getBestTrajectoryCost()=0;
		virtual void __stdcall getBestControl(int timeStep, float *out_control)=0;
		virtual void __stdcall startIteration(bool advanceTime, const float *initialState)=0;
		virtual void __stdcall startPlanningStep(int stepIdx)=0;
		//typically, this just returns sampleIdx. However, if there's been a resampling operation, multiple new samples may link to the same previous sample (and corresponding state)
		virtual int __stdcall getPreviousSampleIdx(int sampleIdx)=0;
		//samples a new control, considering an optional gaussian prior with diagonal covariance
		virtual void __stdcall getControl(int sampleIdx, float *out_control, const float *priorMean=0, const float *priorStd=0)=0;
		virtual void __stdcall updateResults(int sampleIdx, const float *finalControl, const float *newState, double stateCost, const float *priorMean=0, const float *priorStd=0, float control_cost = std::numeric_limits<float>::quiet_NaN())=0;
		virtual void __stdcall endPlanningStep(int stepIdx)=0;
		virtual void __stdcall endIteration()=0;
	};


	class MarginalSample
	{
	public:
		Eigen::VectorXf state,mirroredState;
		Eigen::VectorXf physicsState;
		Eigen::VectorXf previousState;
		Eigen::VectorXf previousPreviousState;
		Eigen::VectorXf control,mirroredControl;
		Eigen::VectorXf previousControl;
		Eigen::VectorXf previousPreviousControl;
		Eigen::VectorXf targetStateInFuture;  //used for predictive models
		std::deque<Eigen::VectorXf> control_sequence;

		bool targetStateValid;
		ParticleRole particleRole;
		int validationIdx;
		//double weight;
		double forwardBelief;
		double belief;
		double fwdMessage;
		double bwdMessage;
		double stateCost;
		double originalStateCostFromClient;
		double statePotential;
		double fullCost;
		double costToGo;
		double fullControlCost;
		double controlCost;
		double controlPotential;
		double bestStateCost;
		double stateDeviationCost;
		double costSoFar; //argmin_i cost so far of segment i in previous step + transition cost (i.e., log of Gaussian kernel) + this segment's cost. Forward equivalent of costToGo
		double priorProbability;
		int fullSampleIdx;
		int previousMarginalSampleIdx;
		//only used in the no kernels mode
		int priorSampleIdx; 
		int nForks;
		bool physicsStateValid;
		bool pruned;
		bool is_valid_;

		bool nearest_neighbor_prior_;
		bool machine_learning_prior_;
		bool previous_frame_prior_;

		void init(int nStateDimensions,int nControlDimensions){
			state.resize(nStateDimensions);
			state.setZero();
			mirroredState.resize(nStateDimensions);
			mirroredState.setZero();
			previousState.resize(nStateDimensions);
			previousState.setZero();
			previousPreviousState.resize(nStateDimensions);
			previousPreviousState.setZero();
			control.resize(nControlDimensions);
			control.setZero();
			mirroredControl.resize(nControlDimensions);
			mirroredControl.setZero();
			previousControl.resize(nControlDimensions);
			previousControl.setZero();
			//weight=1.0;
			fullCost=0.0;
			stateCost=0;
			originalStateCostFromClient=0;
			statePotential=1;
			fullControlCost=0;
			bestStateCost=0;
			controlCost=0;
			stateDeviationCost=0;
			belief=fwdMessage=bwdMessage=1;
			forwardBelief=1.0;
			nForks=0;
			targetStateValid=false;
			particleRole = ParticleRole::FREE;
			validationIdx = 0;
			pruned=false;
			previousMarginalSampleIdx = 0;
			is_valid_ = false;
		}
	};

} //AaltoGames


#endif //safeguard