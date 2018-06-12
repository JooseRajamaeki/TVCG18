#include "UnityOde.h"

#include <unordered_map>
#include <map>

#ifndef UNITYODE_INTERNAL_H
#define UNITYODE_INTERNAL_H

typedef std::unordered_map<int,dBodyID> BodyContainer;
typedef std::unordered_map<int,dJointID> JointContainer;
typedef std::unordered_map<int,dJointGroupID> JointGroupContainer;
typedef std::unordered_map<int,dGeomID> GeomContainer;
typedef std::unordered_map<int,dJointID> AMotorContainer;
typedef std::unordered_map<int,dContact> ContactContainer;
typedef std::unordered_map<int,dJointFeedback> JointFeedbackContainer;

#define NUM_CONTACTS 10

class OdeThreadContext
{
public:
	OdeThreadContext()
		: frictionCoefficient(1.0f)
	{
	}
	dWorldID world;
	dSpaceID space;
	dJointGroupID contactGroup;
	BodyContainer bodies;
	JointContainer joints;
	JointGroupContainer jointGroups;
	GeomContainer geoms;
	AMotorContainer amotors;
	ContactContainer contacts;
	JointFeedbackContainer jointFeedbacks;
	float contactSoftCFM;
	float frictionCoefficient;
	bool errorOccurred;
};

struct BodyState
{
	dReal location[3];
	dReal velocity[3];
	dReal angularVelocity[3];
	dReal quaternion[4];
};

struct AMotorState
{
	dReal angle[3];
	dVector3 axis[3];
	dxJointLimitMotor limot[3]; // limit+motor info for axes
};

struct JointState
{
	dReal fmax[3];
};

struct ContactState
{
	dVector3 position;
	dVector3 normal;
	dVector3 vel;
	int geomId1;
	int geomId2;
};

struct SaveState
{
	std::vector<BodyState> bodies;
	std::vector<AMotorState> amotors;
	std::vector<float> hingeAngles;
	std::vector<JointState> joints;
	std::vector<ContactState> contacts;
	float contactSoftCFM;
	float frictionCoefficient;
	unsigned long seed;
};


OdeThreadContext *getOdeThreadContext(int idx);
void erase_contacts(int idx);

#endif