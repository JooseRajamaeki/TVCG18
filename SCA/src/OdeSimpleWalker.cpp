

#include <stdlib.h>
#include <cmath>
#include <vector>
#include <queue>
#include <chrono>
#include <cstdlib>
#include <cstdio>
#include <Eigen/Geometry>

#include "RenderClient.h"
#include "RenderCommands.h"
#include "UnityOde.h"
#include "UnityOde_internal.h"

#include "mathutils.h"
#include "SCAController.h"
#include "FileUtils.h"



using namespace std::chrono;
using namespace AaltoGames;
using namespace Eigen;
#include "Biped.h"
#include "BipedLegsOnly.h"
#include "Quadruped.h"
#include "Monoped.h"



#define MONOPED 1
#define LEGS 2
#define HUMANOID 3
#define QUADRUPED 4


#define CHARACTER HUMANOID


#if CHARACTER == MONOPED
//Monoped-specific parameters
typedef Monoped TestRig;
static const float controlAccSd = deg2rad*10.0f;
static const float angleSd = deg2rad*20.0f;  //cost function
static const int nTrajectories = 256;
static const int nRealtimeTrajectories = 64;
int non_groung_contact_bones[] = { 1, 0 };
int feet_bones[] = { 2 };
#endif

#if CHARACTER == LEGS
//Legs only biped specific parameters
typedef BipedLegsOnly TestRig;
static const float controlAccSd = deg2rad*10.0f;
static const float angleSd = deg2rad*20.0f;  //cost function
static const int nTrajectories = 64;
static const int nRealtimeTrajectories = 16;
int non_groung_contact_bones[] = { (int)TestRig::Bones::bLeftThigh,(int)TestRig::Bones::bRightThigh,(int)TestRig::Bones::bLeftShin,(int)TestRig::Bones::bRightShin };
int feet_bones[] = { (int)TestRig::Bones::bLeftFoot,(int)TestRig::Bones::bRightFoot };
#endif

#if CHARACTER == HUMANOID
//Full humanoid biped specific parameters
typedef Biped TestRig;
static const float controlAccSd = deg2rad*10.0f;
static const float angleSd = deg2rad*20.0f;  //cost function
static const int nTrajectories = 64;
static const int nRealtimeTrajectories = 16;
int non_groung_contact_bones[] = { (int)TestRig::Bones::bPelvis,(int)TestRig::Bones::bLeftThigh,(int)TestRig::Bones::bLeftShin,(int)TestRig::Bones::bRightThigh,(int)TestRig::Bones::bRightShin,(int)TestRig::Bones::bSpine,(int)TestRig::Bones::bHead,(int)TestRig::Bones::bLeftUpperArm,(int)TestRig::Bones::bLeftForeArm,(int)TestRig::Bones::bRightUpperArm,(int)TestRig::Bones::bRightForeArm };
int feet_bones[] = { (int)TestRig::Bones::bLeftFoot,(int)TestRig::Bones::bRightFoot };
#endif

#if CHARACTER == QUADRUPED
//Quadruped-specific parameters
typedef Quadruped TestRig;
static const float controlAccSd = deg2rad*10.0f;
static const float angleSd = deg2rad*20.0f;  //cost function
static const int nTrajectories = 64;
static const int nRealtimeTrajectories = 16;
int non_groung_contact_bones[] = { (int)TestRig::Bones::bPelvis,(int)TestRig::Bones::bLeftThigh,(int)TestRig::Bones::bRightThigh,(int)TestRig::Bones::bLeftUpperArm,(int)TestRig::Bones::bRightUpperArm };
int feet_bones[] = { (int)TestRig::Bones::bLeftForeArm,(int)TestRig::Bones::bRightForeArm,(int)TestRig::Bones::bLeftShin,(int)TestRig::Bones::bRightShin };
#endif

//Common parameters
static int num_motor_angles = 0;
static const float maxDistanceFromOrigin = 5.0f;
static const float rad2deg = 1.0f / deg2rad;
static const bool rigTestMode = false;
static const float poseSpringConstant = 10.0f;
static TestRig character;
static VectorXf controlMin, controlMax, controlRange, controlMean, controlSd, controlDiffSd;
static SCAControl *flc;

static const float planningHorizonSeconds = 1.2f;
static int nTimeSteps = (int)(planningHorizonSeconds / timeStep);
static int nPhysicsPerStep = 1;
static const int fps = (int)(0.5f + 1.0f / timeStep);
static const bool useThreads = true;
static int resetSaveSlot = nTrajectories + 1;
static int masterContext;
static const float kmh2ms = 1.0f / 3.6f;
static float targetSpeed = 1.0f;
static const float ikSd = 0.05f;
static const bool scoreAngles = true;
static const float velSd = 0.05f;
static const bool useContactVelCost = false;
static const float contactVelSd = 0.2f;
static float comDiffSd = 0.025f;
static const float controlSdRelToRange = 8.0f;


static const bool useFastStep = false;
static const float angleSamplingSd = deg2rad*25.0f;  //sampling
static const float resampleThreshold = 2.0f;
static const float mutationSd = 0.1f;
static const float poseTorqueRelSd = 0.5f;
static const float poseTorqueK = maxControlTorque / (1.0f*PI); //we exert maximum torque towards default pose if the angle displacement is 90 degrees 
static const float friction = 0.5f;
static int frameIdx = 0;
static int randomSeed = 2;
static const bool multiTask = false;
static bool enableRealTimeMode = true;
static bool realtimeMode = false;
static int stateDim;
static const bool useErrorLearning = false;
static const bool test_real_time_mode = false;

//video capturing
static bool captureVideo = true;
static const int startRealtimeModeAt = 9000;
static int autoExitAt = 5000;
static const int nInitialCapturedFrames = autoExitAt;
static const int nFramesToCaptureEveryMinute = 15 * fps;

//launched spheres
static const bool useSpheres = false;
static const int sphereInterval = 5000;
static int lastSphereLaunchTime = 0;

//acceleration
static const bool useAcceleration = false;
static const float acceleration = 1.0f / 10.0f; //1 m/s in 10 seconds
static const float acceleratedMaxSpeed = 100.0f; //practically inf (meters per second)

static bool use_external_prior = true;
//random impulses 
static bool useRandomImpulses = false;
static const float randomImpulseMagnitude = 100.0f;
static const int randomImpulseInterval = 200;

//targeted walking
static bool useWalkTargets = false;
static Vector3f walkTarget = Vector3f(-1000.0f, 0, 0);
static int walk_time = 200;

static enum TestType
{
	STRAIGHT_WALK = 0, GET_UP
};

static const bool read_settings = false;
static TestType test = TestType::STRAIGHT_WALK;

//get up test
static int start_get_up_test_at = 2000;
const std::string fallen_filename = "fallen_position.txt";


//recovery mode
static const bool enableRecoveryMode = false;
static const bool includeRecoveryModeInState = true; //Has to be included as long as the mode directly affects fmax and spring kp
static float recoveryModePoseSdMult = 5.0f;
static float recoveryModeAccSdMult = 1.0f;
static float recoveryModeFmax = defaultFmax*2.0f;
static bool inRecoveryMode = false;
static const float recoveryModeAngleLimit = 30.0f;
static const float recoveryModeSpringKp = springKp*5.0f;

//misc
static int groundPlane = -1;

std::deque<Eigen::VectorXf> control_sequence;
std::deque<Eigen::VectorXf> machine_learning_control_sequence;


static bool run_on_neural_network = false;


static std::string started_at_time_string = "";
static std::vector<std::vector<float>> costs;
static std::vector<std::string> comments;
static bool no_settings_exit = false;

static Eigen::VectorXf minControls;
static Eigen::VectorXf maxControls;

static void write_vector_to_file(const std::string& filename, const std::vector<std::vector<float> >& data, const std::vector<std::string> comments = std::vector<std::string>()) {

	std::ofstream myfile;
	myfile.open(filename);
	myfile.clear();

	for (const std::string& comment_line : comments) {
		myfile << "//" << comment_line << std::endl;
	}

	for (const std::vector<float>& measurement : data) {
		for (unsigned i = 0; i < measurement.size(); i++) {
			myfile << measurement[i];
			if (i < (int)measurement.size() - 1) {
				myfile << ",";
			}
		}
		myfile << std::endl;
	}

	myfile.close();

}

static void write_deque_of_eigen_to_file(const std::string& filename, const std::deque<Eigen::VectorXf >& data, const std::vector<std::string> comments = std::vector<std::string>()) {

	std::ofstream myfile;
	myfile.open(filename);
	myfile.clear();

	for (const std::string& comment_line : comments) {
		myfile << "//" << comment_line << std::endl;
	}

	for (const Eigen::VectorXf& measurement : data) {
		for (unsigned i = 0; i < measurement.size(); i++) {
			myfile << measurement[i];
			if (i < (int)measurement.size() - 1) {
				myfile << ",";
			}
		}
		myfile << std::endl;
	}

	myfile.close();

}


//timing
high_resolution_clock::time_point t1;
void startPerfCount()
{
	t1 = high_resolution_clock::now();
}
int getDurationMs()
{
	high_resolution_clock::time_point t2 = high_resolution_clock::now();

	std::chrono::duration<double> time_span = duration_cast<std::chrono::duration<double>>(t2 - t1);
	return (int)(time_span.count()*1000.0);
}

class SphereData
{
public:
	int body, geom, spawnFrame;
};
static std::vector<SphereData> spheres;


class SimulationContext
{
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;  //needed as we have fixed size eigen matrices as members
	VectorXf stateFeatures;
	VectorXf control;
	VectorXf priorMean, priorSd;
	VectorXf angleRates;
	Vector3f initialPosition, resultPosition;
	float stateCost;
	float controlCost;
	int trajectoryIdx;
};

static SimulationContext contexts[nTrajectories + 2];

Vector3f get_target_dir(Vector3f& com) {

	Vector3f dir = walkTarget - com;
	dir.normalize();
	return dir;
}

//state vector seen by the controller, including both character's physical state and task information
int computeStateVector(float *out_state)
{

	Vector3f com;
	character.computeCOM(com);

	Vector3f target_dir = get_target_dir(com);
	const Vector3f initialDir(-1, 0, 0);
	Quaternionf targetRotation = Quaternionf::FromTwoVectors(initialDir, target_dir);
	Quaternionf stateRotation = targetRotation.inverse();

	const bool use_motor_angles = false;

	int nState = 0;
	if (!use_motor_angles) {
		nState = character.computeStateVector(out_state, stateRotation);

		//task variables scaled larger so that they dominate the state distances
		out_state[nState++] = targetSpeed*10.0f;

	}

	if (use_motor_angles) {

		//since ground contacts are highly significant, add foot bone tip y pos and vel with greater weight
		for (int i = 0; i <= 1; i++)
		{
			Vector3f footPos = character.getFootPos((OdeRig::BodySides)i);
			out_state[nState] = footPos.z();
			nState++;
		}

		Vector3f tmp = com;
		out_state[nState] = tmp.z();
		nState++;

		character.computeMeanVel(tmp);
		out_state[nState] = tmp.z();
		nState++;



		Quaternionf q(odeBodyGetQuaternion(character.bones[0]->body));
		Quaternionf root_rotation = stateRotation*q;

		out_state[nState] = root_rotation.x();
		nState++;
		out_state[nState] = root_rotation.y();
		nState++;
		out_state[nState] = root_rotation.z();
		nState++;
		out_state[nState] = root_rotation.w();
		nState++;

		if (num_motor_angles == 0) {
			for (auto joint_ptr : character.joints) {
				num_motor_angles += joint_ptr->nMotorDof;
			}
		}

		float motor_angles[100];
		character.getCurrentMotorAngles(motor_angles);

		for (int motor_angle = 0; motor_angle < num_motor_angles; motor_angle++) {
			out_state[nState] = motor_angles[motor_angle];
			nState++;
		}

		character.getCurrentAngleRates(motor_angles);

		for (int motor_angle = 0; motor_angle < num_motor_angles; motor_angle++) {
			out_state[nState] = motor_angles[motor_angle];
			nState++;
		}

		//task variables scaled larger so that they dominate the state distances
		out_state[nState++] = target_dir.norm()*10.0f;

	}

	if (enableRecoveryMode && includeRecoveryModeInState  && flc->sampling_mode_ == SCAControl::SamplingMode::PRODUCT_OF_GAUSSIANS)
	{
		out_state[nState++] = 10.0f*(inRecoveryMode ? 1.0f : 0);  //needed as state cost computed differently, and recovery mode affects the state,action -> next state mapping (through spring constants and fmax)
	}

	if (useSpheres)
	{


		bool pushZeros = true;
		if (spheres.size() > 0)
		{
			//add the relative position of the last launched sphere
			SphereData &sd = spheres.back();
			Vector3f spherePos(odeBodyGetPosition(sd.body));
			Vector3f sphereVel(odeBodyGetLinearVel(sd.body));
			Vector3f pos(odeBodyGetPosition(character.bones[0]->body));
			Vector3f relPos = stateRotation*(spherePos - pos);
			Vector3f relVel = stateRotation*sphereVel;
			//if sphere flying towards us, add it to the state
			if (relPos.norm() < 5.0f && relVel.dot(relPos.normalized()) < -1.0f)
			{
				character.pushStateVector3f(nState, out_state, relPos);
				character.pushStateVector3f(nState, out_state, relVel);
				pushZeros = false;
			}
		}
		if (pushZeros)
		{
			character.pushStateVector3f(nState, out_state, Vector3f::Zero());
			character.pushStateVector3f(nState, out_state, Vector3f::Zero());
		}
	}
	return nState;
}

std::chrono::time_point<std::chrono::system_clock> start, end;

Eigen::VectorXf init_motor_angles;

static int character_root_bone = -1;

static int orig_ml_trajectories;
static int orig_noisy_ml_trajectories;
static int orig_nearest_neighbor_trajectories;


void print_sampling_mode(void) {
	switch (flc->sampling_mode_)
	{
	case AaltoGames::SCAControl::PRODUCT_OF_GAUSSIANS:
		std::cout << "Sampling scheme: Product of distributions" << std::endl;
		break;
	case AaltoGames::SCAControl::MINIMIZING_INFORMATION_LOSS:
		std::cout << "Sampling scheme: Maximum entropy" << std::endl;
		break;
	default:
		std::cout << "Error in sampling scheme." << std::endl;
		break;
	}
}

void print_nonlinearity(void) {

	switch (flc->network_type_)
	{
	case AaltoGames::SCAControl::BSELU:
		std::cout << "BSELU" << std::endl;
		break;
	case AaltoGames::SCAControl::ELU:
		std::cout << "ELU" << std::endl;
		break;
	default:
		std::cout << "Error in network type." << std::endl;
		break;
	}

}

void EXPORT_API rcInit()
{

#if CHARACTER == HUMANOID
	character_root_bone = (int)TestRig::Bones::bSpine;
#endif

#if CHARACTER == MONOPED
	comDiffSd *= 100000.0f;
#endif


	flc = new SCAControl();

	
	if (read_settings) {

		std::ifstream infile("settings.txt");

		std::string comment = "//";

		std::deque<std::string> settings;

		for (std::string line; getline(infile, line); )
		{

			if (line.find(comment) != std::string::npos) {
				continue;
			}

			settings.push_back(line);

		}

		infile.close();


		if (settings.size() < 1) {
			autoExitAt = 0;
			return;
		}


		for (std::string setting : settings) {

			std::stringstream stream(setting);
			std::string key = "";
			std::string value = "";
			std::getline(stream, key, ':');
			std::getline(stream, value, '\n');


			if ("MACHINE_LEARNING" == key) {
				std::stringstream value_stream(value);
				value_stream >> flc->use_machine_learning_;
			}

			if ("NEAREST_NEIGHBOR" == key) {
				std::stringstream value_stream(value);
				value_stream >> flc->use_forests_;
			}


			if ("SAMPLING_MODE" == key) {

				if (value.compare("kl") == 0) {
					flc->sampling_mode_ = SCAControl::SamplingMode::MINIMIZING_INFORMATION_LOSS;
				}

				if (value.compare("l1") == 0) {
					flc->sampling_mode_ = SCAControl::SamplingMode::PRODUCT_OF_GAUSSIANS;
				}
			}

			if ("NETWORK_TYPE" == key) {
				if (value.compare("bselu") == 0) {
					flc->network_type_ = SCAControl::BSELU;
				}

				if (value.compare("elu") == 0) {
					flc->network_type_ = SCAControl::ELU;
				}
			}


			
			if ("TEST_TYPE" == key) {	

				test = TestType::STRAIGHT_WALK;

				int comparison = value.compare("get_up");
			
				if (comparison == 0) {
					test = TestType::GET_UP;
					std::cout << "Get up test chosen." << std::endl;
				}

			}


			if ("LAYERS" == key) {
				std::stringstream value_stream(value);
				value_stream >> flc->amount_network_layers_;
			}

			if ("DROPOUT" == key) {
				std::stringstream value_stream(value);
				value_stream >> flc->drop_out_stdev_;
			}

			if ("STOCHASTICITY_DIM" == key) {
				std::stringstream value_stream(value);
				value_stream >> flc->input_noise_dim_;
			}

			if ("ZERO_SAMPLES" == key) {
				std::stringstream value_stream(value);
				value_stream >> flc->zero_samples_;
			}



		}





		std::ofstream outfile("settings.txt", std::ofstream::out);
		outfile.clear();
		outfile.close();

		captureVideo = false;
		useWalkTargets = false;
		flc->force_old_best_valid_ = true;

		print_sampling_mode();
		print_nonlinearity();


	};



	if (test == TestType::STRAIGHT_WALK) {
		std::cout << "Straight walk test." << std::endl;
	}
	if (test == TestType::GET_UP) {
		std::cout << "Get up test." << std::endl;
	}



	if (test_real_time_mode) {
		enableRealTimeMode = true;
	}

	started_at_time_string = get_time_string();
	unsigned seed = time(nullptr);
	srand(seed);



	costs.clear();
	comments.clear();

	initOde(nTrajectories + 2);
	setCurrentOdeContext(ALLTHREADS);
	allocateODEDataForThread();
	odeRandSetSeed(randomSeed);
	// create world
	odeWorldSetGravity(0, 0, -9.81f);
	odeWorldSetCFM(1e-5);
	odeSetFrictionCoefficient(friction);

	odeWorldSetContactMaxCorrectingVel(5);
	odeWorldSetContactSurfaceLayer(0.01f);
	groundPlane = odeCreatePlane(0, 0, 0, 1, 0);
	odeGeomSetCategoryBits(groundPlane, 0xf0000000);
	odeGeomSetCollideBits(groundPlane, 0xffffffff);
	character.init(false, rigTestMode);
	int nLegs = character.numberOfLegs();

	//controller init (motor target velocities are the controlled variables, one per joint)
	float tmp[1024];
	stateDim = computeStateVector(tmp);
	controlMin = Eigen::VectorXf::Zero(character.controlDim);
	controlMax = Eigen::VectorXf::Zero(character.controlDim);
	controlMean = Eigen::VectorXf::Zero(character.controlDim);
	controlSd = Eigen::VectorXf::Zero(character.controlDim);
	controlDiffSd = Eigen::VectorXf::Zero(character.controlDim);

	int controlVarIdx = 0;
	for (size_t i = 0; i < character.joints.size(); i++)
	{
		OdeRig::Joint *j = character.joints[i];
		for (int dofIdx = 0; dofIdx < j->nMotorDof; dofIdx++)
		{


			controlMin[controlVarIdx] = j->angleMin[dofIdx];
			controlMax[controlVarIdx] = j->angleMax[dofIdx];
			controlSd[controlVarIdx] = angleSamplingSd;


			controlMean[controlVarIdx] = 0;
			controlDiffSd[controlVarIdx] = controlSd[controlVarIdx];
			controlVarIdx++;
		}
	}


	controlRange = controlMax - controlMin;

	flc->amount_data_in_tree_ = 2 * 60 * 30; //2 minutes


	flc->init(nTrajectories, nTimeSteps / nPhysicsPerStep, stateDim, character.controlDim, controlMin.data(), controlMax.data(), controlMean.data(), controlSd.data(), controlDiffSd.data(), mutationSd, false);
	flc->no_prior_trajectory_portion_ = 0.25f;
	flc->learning_budget_ = 2000;
	if (useWalkTargets) {
		flc->learning_budget_ = 10000;
	}

	minControls = flc->control_min_;
	maxControls = flc->control_max_;

	for (int i = 0; i < nTrajectories + 1; i++)	//+1 because of master context
	{
		contexts[i].stateFeatures.resize(stateDim);
		contexts[i].control.resize(character.controlDim);
		contexts[i].trajectoryIdx = i;
		contexts[i].priorMean.resize(character.controlDim);
		contexts[i].priorSd.resize(character.controlDim);
		contexts[i].angleRates.resize(character.controlDim);
	}

	masterContext = nTrajectories;
	setCurrentOdeContext(masterContext);
	stepOde(timeStep, true); //allow time for joint limits to set
	stepOde(timeStep, true); //allow time for joint limits to set
	stepOde(timeStep, true); //allow time for joint limits to set
	character.saveInitialPose();
	saveOdeState(resetSaveSlot, masterContext);
	saveOdeState(masterContext, masterContext);

	float tmp_angles[1000];
	int amount_motor_angles = character.getCurrentMotorAngles(tmp_angles);

	init_motor_angles.resize(amount_motor_angles);
	for (int i = 0; i < amount_motor_angles; ++i) {
		init_motor_angles[i] = tmp_angles[i];
	}


	orig_ml_trajectories = flc->machine_learning_samples_;
	orig_noisy_ml_trajectories = flc->noisy_machine_learning_samples_;
	orig_nearest_neighbor_trajectories = flc->nn_trajectories_;


	start = std::chrono::system_clock::now();

}

void EXPORT_API rcUninit()
{
	delete flc;
	uninitOde();
}

void EXPORT_API rcGetClientData(RenderClientData &data)
{
	data.physicsTimeStep = timeStep*(float)nPhysicsPerStep;
	data.maxAllowedTimeStep = data.physicsTimeStep; //render every step
	data.defaultMouseControlsEnabled = true;
}

static int dofIdx = 0;
static float rigTestControl[32] = { 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 };
static bool paused = false;
static bool applyImpulse = false;

void throwSphere()
{
	//launch a ball
	setCurrentOdeContext(ALLTHREADS);
	float r = 0.33f;
	SphereData sd;
	sd.geom = odeCreateSphere(r);
	sd.body = odeBodyCreate();
	sd.spawnFrame = frameIdx;
	odeGeomSetBody(sd.geom, sd.body);
	dMass mass;
	odeMassSetSphereTotal(sd.body, 50.0f, r);
	setCurrentOdeContext(masterContext);
	restoreOdeState(masterContext);
	Vector3f characterPos(odeBodyGetPosition(character.bones[0]->body));
	Vector3f throwDir(randomf() - 0.5f, randomf() - 0.5f, 0);
	throwDir.normalize();
	Vector3f vel;
	character.computeMeanVel(vel);
	Vector3f spawnPos = characterPos + vel - 5.0f*throwDir;  //Add vel, as the ball will hit in 1 second
	spawnPos.z() = 2.0f;
	odeBodySetPosition(sd.body, spawnPos.x(), spawnPos.y(), spawnPos.z());
	Vector3f spawnVel = throwDir*5.0f;
	spawnVel.z() = 3.5f;
	odeBodySetLinearVel(sd.body, spawnVel.x(), spawnVel.y(), spawnVel.z());
	saveOdeState(masterContext, masterContext);
	spheres.push_back(sd);
	lastSphereLaunchTime = frameIdx;
}

static void write_vectors_to_file(const std::string& filename, const std::deque<Eigen::VectorXf>& data, const std::vector<std::string> comments = std::vector<std::string>()) {

	std::ofstream myfile;
	myfile.open(filename);

	for (const std::string& comment_line : comments) {
		myfile << "//" << comment_line << std::endl;
	}

	for (const Eigen::VectorXf& datum : data) {
		int size = datum.size();
		for (int i = 0; i < size; i++) {
			myfile << datum[i];
			if (i < size - 1) {
				myfile << ",";
			}
		}
		myfile << std::endl;
	}

	myfile.close();

}

static bool switchTargets = false; //if true, toggles target switching at next rcUpdate()
void EXPORT_API rcOnKeyDown(int key)
{
	if (key == 'p')
		paused = !paused;

	if (key == '2')
	{
		dofIdx = (dofIdx + 1) % character.nTotalDofs;
		printf("Adjusting dof %d\n", dofIdx);
	}

	if (key == '1')
	{
		dofIdx = (dofIdx - 1 + character.nTotalDofs) % character.nTotalDofs;
		printf("Adjusting dof %d\n", dofIdx);
	}

	if (key == '3')
	{
		rigTestControl[dofIdx] = -maxControlSpeed;
	}

	if (key == '4')
	{
		rigTestControl[dofIdx] = 0;
	}

	if (key == '5')
	{
		rigTestControl[dofIdx] = maxControlSpeed;
	}


	if (key == '6') {
		saveOdeStateToFile(fallen_filename.data(), masterContext, 0, nullptr);
	}

	if (key == '7') {
		loadOdeStateFromFile(fallen_filename.data(), masterContext, 0, nullptr);
	}

	if (key == 'r')
	{
		saveOdeState(masterContext, resetSaveSlot);
		if (useAcceleration)
			targetSpeed *= 0;
	}

	if (key == 't')
	{
		realtimeMode = !realtimeMode;
	}

	if (key == ' ')
	{
		throwSphere();
	}

	if (key == 'i')
	{
		applyImpulse = true;
	}

	if (key == 'w')
	{
		switchTargets = true;
	}

	if (key == 'e') {
		autoExitAt = frameIdx;
	}

	if (key == 'n') {
		run_on_neural_network = true;
	}

	if (key == 'm') {
		run_on_neural_network = false;
	}

	if (key == 'o') {
		write_vectors_to_file("controls.txt", control_sequence);
		write_vectors_to_file("ml_controls.txt", machine_learning_control_sequence);
		std::system("python density_plotter.py");
	}

	if (key == 'l') {
		write_vectors_to_file("controls.txt", control_sequence);
		std::system("python sequence_plotter.py");
	}
}

void EXPORT_API rcOnKeyUp(int key)
{
}

float computeStateCost(const TestRig &character, bool debugOutput = false)
{
	float result = 0;
	Vector3f com;
	character.computeCOM(com);

	Vector3f target_dir = get_target_dir(com);
	const Vector3f initialDir(-1, 0, 0);
	Quaternionf targetRotation = Quaternionf::FromTwoVectors(initialDir, target_dir);




	//bone angle diff from initial
	for (size_t i = 0; i < character.bones.size(); i++)
	{
		OdeRig::Bone *b = character.bones[i];
		Quaternionf q = ode2eigenq(odeBodyGetQuaternion(b->body));
		float weight = 1.0f;
		float sdMult = 1.0f;
		result += weight*squared(q.angularDistance(targetRotation*b->initialRotation) / (angleSd));
	}


	//com over feet
	if (scoreAngles)
	{
		Vector3f meanFeet = Vector3f::Zero();
		int num_feet = 0;

		float foot_max_height = -1000.0f;

		for (int i : feet_bones)
		{

			const Eigen::Vector3f footPos(odeBodyGetPosition(character.bones[i]->body));

			foot_max_height = std::max(foot_max_height, footPos.z());

			meanFeet += footPos;
			++num_feet;
		}

		meanFeet /= (float)num_feet;
		meanFeet.z() = foot_max_height;

		Vector3f comDiff = com - meanFeet;
		comDiff.z() = std::min(comDiff.z(), 0.0f); //only penalize vertical difference if a foot higher than com (prevents the cheat where character lies down and holds legs up over com)
		result += comDiff.squaredNorm() / squared(comDiffSd);
	}

	//COM vel difference
	Vector3f vel;
	character.computeMeanVel(vel);
	Vector3f targetVel = targetSpeed*target_dir;
	Vector3f velDiff = vel - targetVel;
	velDiff.z() = std::min(velDiff.z(), 0.0f);  //don't penalize upwards deviation
	result += velDiff.squaredNorm() / squared(velSd);


	return result;
}


bool fallen()
{
	for (int bone : non_groung_contact_bones) {

		int bone_geom_id = character.bones[bone]->geoms[0];

		Vector3f pos, normal, vel;
		pos.setZero();
		normal.setZero();
		vel.setZero();
		bool contact = odeGetGeomContact(bone_geom_id, groundPlane, pos.data(), normal.data(), vel.data());

		if (contact) {
			return true;
		}

	}

	return false;
}

bool recovered()
{
	return !fallen();
}


void applyControl(const float *control)
{


	float currentAngles[256];
	float control2[256];
	memcpy(control2, control, sizeof(float)*character.controlDim);
	character.getCurrentMotorAngles(currentAngles);
	for (int i = 0; i < character.nTotalDofs; i++)
	{
		control2[i] = (control[i] - currentAngles[i]) *  poseSpringConstant;
	}
	character.applyControl(control2);


	character.setFmaxForAllMotors(defaultFmax);
	character.setMotorSpringConstants(springKp, springDamping);

	if (enableRecoveryMode)
	{
		bool allow_changing_spring_constants = true;
		bool allow_changing_fmax = true;

		if (flc->sampling_mode_ == SCAControl::SamplingMode::MINIMIZING_INFORMATION_LOSS) {
			allow_changing_spring_constants = false;
			allow_changing_fmax = false;
		}

		if (inRecoveryMode) {
			if (allow_changing_fmax) {
				character.setFmaxForAllMotors(recoveryModeFmax);
			}
			if (allow_changing_spring_constants) {
				character.setMotorSpringConstants(recoveryModeSpringKp, springDamping);
			}
		}
	}


}

static float previousCost = 10000.0f;

void EXPORT_API rcOnMouse(float, float, float, float, float, float, int, int, int) {

}

int total_simulations = 0;
int cumulative_controller_update_time = 0;

void EXPORT_API rcUpdate()
{

	if (flc->sampling_mode_ == SCAControl::SamplingMode::MINIMIZING_INFORMATION_LOSS) {
		use_external_prior = false;
	}

	if (test == TestType::GET_UP) {
		if (frameIdx == start_get_up_test_at) {
			loadOdeStateFromFile(fallen_filename.data(), masterContext, 0, nullptr);
		}
	}



	//accelerated target velocity
	if (useAcceleration)
	{
		if (frameIdx == 0)
			targetSpeed *= 0;
		targetSpeed += acceleration*timeStep;
	}

	//walk targets
	if (useWalkTargets)
	{


		//pick next target location and velocity
		Vector3f com;
		character.computeCOM(com);
		com.z() = 0;

		auto random_walk_target = []() {
			Eigen::Vector3f target;

			target.setRandom();

			float dist_from_origin = 5.0f;

			target.z() = 0;
			target.normalize();
			target *= dist_from_origin;

			return target;

		};

		if (frameIdx % walk_time == 0 || frameIdx == 0 || switchTargets) {
			walkTarget = random_walk_target();
			switchTargets = false;
		}

		float distToTarget = (walkTarget - com).norm();
		while (distToTarget < 1.5f)
		{
			walkTarget = random_walk_target();
			distToTarget = (walkTarget - com).norm();
		}


	}

	//visualize target as a "pole"
	dMatrix3 R;
	dRSetIdentity(R);
	rcSetColor(0.5f, 1, 0.5f);
	rcDrawCapsule(walkTarget.data(), R, 20.0f, 0.02f);
	rcSetColor(1, 1, 1);

	//multitasking with random velocity
	static float timeOnTask = 0;
	timeOnTask += timeStep;
	if (multiTask && (timeOnTask > 5.0f))
	{
		float r = (float)randInt(-1, 1);
		targetSpeed = r;
		printf("New target vel %f\n", targetSpeed);
		timeOnTask = 0;
	}

	//setup current character state
	setCurrentOdeContext(masterContext);
	restoreOdeState(masterContext);
	VectorXf startState(stateDim);
	computeStateVector(&startState[0]);

	int currentTrajectories = nTrajectories;
	int learning = true;

	if (realtimeMode) {
		currentTrajectories = nRealtimeTrajectories;
		learning = false;

		flc->machine_learning_samples_ = 2;
		flc->noisy_machine_learning_samples_ = 2;
		flc->nn_trajectories_ = 2;

	}
	else {

		flc->machine_learning_samples_ = orig_ml_trajectories;
		flc->noisy_machine_learning_samples_ = orig_noisy_ml_trajectories;
		flc->nn_trajectories_ = orig_nearest_neighbor_trajectories;

	}

	flc->setParams(previousCost*resampleThreshold, learning, currentTrajectories);

	flc->control_diff_std_ = controlSd;
	flc->static_prior_.mean[0] = init_motor_angles;


	end = std::chrono::system_clock::now();

	std::chrono::duration<double> duration = end - start;
	int seconds = (int)duration.count() % 60;
	int minutes = (int)duration.count() / 60;

	VectorXf scaledControlSd = controlSd;
	controlDiffSd = controlSd;

	if (inRecoveryMode)
	{
		scaledControlSd *= recoveryModePoseSdMult;
		controlDiffSd *= recoveryModePoseSdMult;
	}
	else {
		flc->control_min_ = minControls;
		flc->control_max_ = maxControls;
	}

	flc->setSamplingParams(scaledControlSd.data(), controlDiffSd.data(), mutationSd);

	startPerfCount();
	int masterContext = nTrajectories;



	bool willFall = false;
	if (!rigTestMode)
	{

		if (!run_on_neural_network) {

			float total_computation_seconds = (float)cumulative_controller_update_time / 1000.0f;

			rcPrintString("Number of trajectories: %d %s", currentTrajectories, realtimeMode ? "(Realtime-mode)" : "");
			rcPrintString("Planning horizon: %.1f seconds", planningHorizonSeconds);
			rcPrintString("Simulation time %02d:%02d:%2d (%d frames), total computing time %.2f s", frameIdx / (fps * 60), (frameIdx / fps) % 60, frameIdx % fps, frameIdx, total_computation_seconds);
			rcPrintString("Pruning threshold %.1f", previousCost*resampleThreshold);



			flc->startIteration(true, &startState[0]);

			for (int step = 0; step < nTimeSteps / nPhysicsPerStep; step++)
			{
				int nUsedTrajectories = flc->getNumTrajectories();
				flc->startPlanningStep(step);
				if (step == 0)
				{
					for (int i = 0; i < nUsedTrajectories; i++)
					{
						saveOdeState(contexts[i].trajectoryIdx, masterContext);
					}
				}
				else
				{
					for (int i = 0; i < nUsedTrajectories; i++)
					{
						saveOdeState(contexts[i].trajectoryIdx, contexts[i].trajectoryIdx);
					}

				}

				std::deque<std::future<void>> workers;

				for (int t = nUsedTrajectories - 1; t >= 0; t--)
				{
					//lambda to be executed in the thread of the simulation context
					auto controlStep = [step](int data) {
						if (frameIdx == 0 && step == 0 && useThreads)
						{
							allocateODEDataForThread();
							odeRandSetSeed(randomSeed);
						}
						SimulationContext &c = contexts[data];
						setCurrentOdeContext(c.trajectoryIdx);
						restoreOdeState(flc->getPreviousSampleIdx(c.trajectoryIdx));


						if (use_external_prior) {
							//pose prior (towards zero angles)
							character.getCurrentMotorAngles(c.priorMean.data());

							//In pose-based control we use the extra prior to limit acceleration.
							//We first compute the predicted pose based on current pose and motor angular velocities, and then
							//set the prior there.
							c.angleRates.setZero();
							character.getCurrentAngleRates(c.angleRates.data());
							c.priorMean = c.priorMean + c.angleRates / poseSpringConstant;
							c.priorMean = c.priorMean.cwiseMax(controlMin);
							c.priorMean = c.priorMean.cwiseMin(controlMax);
							c.priorSd.setConstant(controlAccSd * (inRecoveryMode ? recoveryModeAccSdMult : 1.0f));


						}

						//sample control
						if (use_external_prior) {
							flc->getControl(c.trajectoryIdx, &c.control[0], c.priorMean.data(), c.priorSd.data());
						}
						else {
							flc->getControl(c.trajectoryIdx, &c.control[0], nullptr, nullptr);
						}


						if (character_root_bone >= 0) {
							c.initialPosition = Eigen::Vector3f(odeBodyGetPosition(character.bones[character_root_bone]->body));
						}
						else {
							character.computeCOM(c.initialPosition);
						}

						bool broken = false;

						//step physics
						float controlCost = 0;
						for (int k = 0; k < nPhysicsPerStep; k++)
						{
							applyControl(&c.control[0]);
							if (useFastStep)
								broken = !stepOdeFast(timeStep, false);
							else
								broken = !stepOde(timeStep, false);
							if (broken)
							{
								restoreOdeState(flc->getPreviousSampleIdx(c.trajectoryIdx));
								break;
							}


							controlCost += character.getAppliedSqJointTorques();
							controlCost /= squared(defaultFmax);

						}



						if (character_root_bone >= 0) {
							c.resultPosition = Eigen::Vector3f(odeBodyGetPosition(character.bones[character_root_bone]->body));
						}
						else {
							character.computeCOM(c.resultPosition);
						}

						if (!broken)
						{
							float brokenDistanceThreshold = 0.25f;
							if ((c.resultPosition - c.initialPosition).norm() > brokenDistanceThreshold)
							{
								restoreOdeState(flc->getPreviousSampleIdx(c.trajectoryIdx));
								c.resultPosition = c.initialPosition;
								broken = true;
							}
						}

						//evaluate state cost
						float stateCost = computeStateCost(character);
						if (broken)
							stateCost += 1000000.0f;
						computeStateVector(&c.stateFeatures[0]);

						flc->updateResults(c.trajectoryIdx, c.control.data(), c.stateFeatures.data(), stateCost + controlCost);
						c.stateCost = stateCost;
						c.controlCost = controlCost;
					};
					if (!useThreads)
						controlStep(t);
					else
						workers.push_back(std::async(std::launch::async, controlStep, t));
				}
				if (useThreads)
				{
					for (std::future<void>& worker : workers) {
						worker.wait();
					}
				}

				flc->endPlanningStep(step);

				bool plot_trajectories = true;

				if (plot_trajectories) {
					//debug visualization
					for (int t = nUsedTrajectories - 1; t >= 0; t--)
					{
						SimulationContext &c = contexts[t];
						if (flc->experience_[step + 1][t].particleRole == ParticleRole::OLD_BEST) {
							rcSetColor(0, 1, 0, 1);
						}
						else if (flc->experience_[step + 1][t].particleRole == ParticleRole::NEAREST_NEIGHBOR)
						{
							rcSetColor(0, 0, 1, 1);
						}
						else
						{
							rcSetColor(1, 1, 1, 1.0f);
						}
						rcDrawLine(c.initialPosition.x(), c.initialPosition.y(), c.initialPosition.z(), c.resultPosition.x(), c.resultPosition.y(), c.resultPosition.z());
					}
				}
				rcSetColor(1, 1, 1, 1);
			}

			flc->endIteration();

			//print profiling info
			int controllerUpdateMs = getDurationMs();
			rcPrintString("Controller update time: %d ms", controllerUpdateMs);
			cumulative_controller_update_time += controllerUpdateMs;

			//check whether best trajectory will fall
			setCurrentOdeContext(contexts[flc->getBestSampleLastIdx()].trajectoryIdx);

			willFall = fallen();

		}


		if (useAcceleration)
			rcPrintString("Target speed: %.2f m/s", fabs(targetSpeed));


		//step master context
		setCurrentOdeContext(masterContext);
		restoreOdeState(masterContext);


		Eigen::VectorXf machine_learning_control = contexts[masterContext].control;

		computeStateVector(contexts[masterContext].stateFeatures.data());
		flc->getMachineLearningControl(contexts[masterContext].stateFeatures.data(), machine_learning_control.data());

		float stochasticity_check = 0.0f;
		if (flc->input_noise_dim_ > 0) {
			Eigen::VectorXf tmp_control = machine_learning_control;
			flc->getMachineLearningControl(contexts[masterContext].stateFeatures.data(), machine_learning_control.data());
			stochasticity_check = (tmp_control - machine_learning_control).norm();
		}

		if (run_on_neural_network) {
			contexts[masterContext].control = machine_learning_control;
		}
		else {
			flc->getBestControl(0, contexts[masterContext].control.data());
		}

		machine_learning_control_sequence.push_back(machine_learning_control);
		control_sequence.push_back(contexts[masterContext].control);

		float best_trajectory_cost = (float)flc->getBestTrajectoryCost();
		for (int k = 0; k < nPhysicsPerStep; k++)
		{
			//apply control (and random impulses)
			applyControl(contexts[masterContext].control.data());
			if (applyImpulse || (useRandomImpulses && !realtimeMode && ((frameIdx % randomImpulseInterval) == 0)))
			{
				applyImpulse = false;
				float impulse[3] = { randomf()*randomImpulseMagnitude,randomf()*randomImpulseMagnitude,randomf()*randomImpulseMagnitude };
				odeBodyAddForce(character.bones[0]->body, impulse);
			}
			if (useFastStep)
				stepOdeFast(timeStep, false);
			else
				stepOde(timeStep, false);
		}
		
		saveOdeState(masterContext, masterContext);

		float controlCost = character.getAppliedSqJointTorques();
		controlCost /= squared(defaultFmax);

		previousCost = contexts[flc->getBestSampleLastIdx()].stateCost;
		float state_cost = computeStateCost(character);
		rcPrintString("state cost %.2f, current control cost %.2f, end state cost %.2f", state_cost, controlCost, previousCost);
		//rcPrintString("Stochasticity check %.9f\n", stochasticity_check);

		total_simulations += flc->amount_samples_*flc->steps_;

		rcPrintString("Total training simulation steps: %d", total_simulations);

		std::vector<float> cost;
		cost.push_back(best_trajectory_cost);
		cost.push_back(state_cost + controlCost);

		flc->getMachineLearningChosenControlDiscrepancy();
		cost.push_back(flc->machine_learning_and_chosen_control_discrepancy_);

		float ml_mse = flc->getMachineLearningMSE();
		cost.push_back(ml_mse);

		bool on_the_ground = fallen();
		cost.push_back((float)on_the_ground);

		costs.push_back(cost);


	} //!rigTestMode
	setCurrentOdeContext(masterContext);

	//in simple forward walking without targets, if moved too far from origin, move back
	if (!useWalkTargets)
	{
		Vector3f com;
		character.computeCOM(com);
		if (captureVideo && com.norm() > maxDistanceFromOrigin)
		{
			Vector3f disp = -com;
			disp.z() = 0;
			for (size_t i = 0; i < character.bones.size(); i++)
			{
				Vector3f pos(odeBodyGetPosition(character.bones[i]->body));
				pos += disp;
				odeBodySetPosition(character.bones[i]->body, pos.x(), pos.y(), pos.z());
			}
			saveOdeState(masterContext, masterContext);
		}
	}


	bool reset_on_fall = true;

	if (test == TestType::GET_UP && frameIdx >= start_get_up_test_at) {
		reset_on_fall = false;
	}

	if (reset_on_fall) {
		bool on_the_ground = fallen();

		//if (captureVideo) {
		//	on_the_ground = false;
		//}

		if (on_the_ground) {

			saveOdeState(masterContext, resetSaveSlot);
			run_on_neural_network = false;

			for (int i = 0; i <= nTrajectories; i++)
			{
				setCurrentOdeContext(contexts[i].trajectoryIdx);
				restoreOdeState(resetSaveSlot);
				saveOdeState(contexts[i].trajectoryIdx, contexts[i].trajectoryIdx);
			}

	
			if (useAcceleration)
				targetSpeed *= 0;
		}
	}


	if (rigTestMode)
	{
		applyControl(rigTestControl);
		if (useFastStep)
			stepOdeFast(timeStep);
		else
			stepOde(timeStep, false);
		saveOdeState(masterContext, masterContext);
	}
	static bool setVP = true;
	if (setVP)
	{
		setVP = false;
		rcSetViewPoint(-(2.0f + maxDistanceFromOrigin), -3, 1.5f, -(maxDistanceFromOrigin - 1), 0, 0.9f);
		rcSetLightPosition(-0, -10, 10);
	}
	rcDrawAllObjects((dxSpace *)odeGetSpace());

	character.debugVisualize();

	//state transitions
	if (willFall)
	{
		rcPrintString("Falling predicted!");
		if (enableRecoveryMode)
			inRecoveryMode = true;
	}
	if (!willFall && recovered())
		inRecoveryMode = false;
	if (inRecoveryMode) {
		if (flc->sampling_mode_ == SCAControl::SamplingMode::PRODUCT_OF_GAUSSIANS) {
			rcPrintString("Recovery mode with stronger movements and wider search distributions.");
		}

		if (flc->sampling_mode_ == SCAControl::SamplingMode::MINIMIZING_INFORMATION_LOSS) {
			rcPrintString("Recovery mode with wider search distributions.");
		}
	}

	//turn on real-time mode automatically when learned enough
	if (frameIdx > startRealtimeModeAt)
		realtimeMode = true;

	if (no_settings_exit) {
		exit(0);
	}

	//quit 
	if (frameIdx > autoExitAt)
	{
		std::deque<std::string> settings = flc->get_settings();

		for (std::string setting : settings) {
			comments.push_back(setting);
		}

		comments.push_back("Turns: " + std::to_string(useWalkTargets));

		std::string file_time_string = get_time_string();

		std::string cost_file_name = file_time_string + "_costs.csv";
		write_vector_to_file(cost_file_name, costs, comments);

		std::string control_file_name = file_time_string + "_controls.csv";
		write_deque_of_eigen_to_file(control_file_name, control_sequence, comments);

		control_file_name = file_time_string + "_controls_machine_learning.csv";
		write_deque_of_eigen_to_file(control_file_name, machine_learning_control_sequence, comments);

		if (fileExists("out.mp4"))
			remove("out.mp4");
		system("screencaps2mp4.bat");
		
		exit(0);
	}

	if (captureVideo)
	{
		//During learning, capture video at the beginning and every now and then.
		if (realtimeMode || (frameIdx < nInitialCapturedFrames || (frameIdx % (60 * fps) < nFramesToCaptureEveryMinute)))
		{
			rcTakeScreenShot();
		}
	}

	//spheres
	if (useSpheres && (frameIdx > lastSphereLaunchTime + sphereInterval))
	{
		Vector3f com;
		character.computeCOM(com);
		if (useWalkTargets || (com.x() > -maxDistanceFromOrigin + 2.0f) && (com.x() < -0.5f))  //don't throw a sphere right before character is about to be teleported
			throwSphere();
	}

	for (size_t i = 0; i < spheres.size(); i++)
	{
		SphereData &sd = spheres[i];
		if (sd.spawnFrame < frameIdx - 5 * fps)
		{
			setCurrentOdeContext(ALLTHREADS);
			odeGeomDestroy(sd.geom);
			odeBodyDestroy(sd.body);
			saveOdeState(masterContext, masterContext);
			spheres[i] = spheres[spheres.size() - 1];
			spheres.resize(spheres.size() - 1);
		}
	}

	frameIdx++;

}
