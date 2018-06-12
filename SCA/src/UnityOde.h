// UnityOde is a C-api ODE wrapper that hides the ODE implementation so that the API can easily be
// used in managed code by marshaling the API. This is used, e.g., with Unity 3D game engine C# scripts. 
// Most of the C# wrapping is generated automatically using SWIG.
// 
// The wrapper also encapsulates threaded simulation. 

#pragma once
#include <stdint.h>

#include <ode/ode.h>
#include <ode/src/joints/joint.h>

#if _MSC_VER // this is defined when compiling with Visual Studio
#define EXPORT_API __declspec(dllexport) // Visual Studio needs annotating exported functions with this
#else
#define EXPORT_API // XCode does not need annotating exported functions, so define is empty
#endif // _MSC_VER
static const int ALLTHREADS=-1;

// Define our own types for SWIG. This is needed because arrays (such as dQuaternion) can't be returned from functions
typedef dReal * OdeQuaternion;
typedef dReal * OdeVector;
typedef const dReal * ConstOdeQuaternion;
typedef const dReal * ConstOdeVector;
typedef const int * BodyIDList;

// Sets the current thread context that is manipulated. For setters and creators, one can use ALLTHREADS to make the operations apply for all contexts.
// Note that getters will return properties from the context 0 if ALLTHREADS is specified.
void EXPORT_API setCurrentOdeContext(int threadIdx);
int EXPORT_API getCurrentOdeContext();

//Sets the number of threads (or simulation contexts) used. All bodies etc. will be duplicated for each thread
bool EXPORT_API initOde(int numThreads);
bool EXPORT_API uninitOde();

EXPORT_API bool initialized();

void EXPORT_API odeRandSetSeed(unsigned long s);

void EXPORT_API allocateODEDataForThread();

EXPORT_API void enableJointFeedback(bool enable);
//supply -1 for jointType to get results from all joints
EXPORT_API void odeBodyGetAccumulatedForce(int bodyId, int jointType, OdeVector result);
EXPORT_API void odeBodyGetAccumulatedTorque(int bodyId, int jointType, OdeVector result);
EXPORT_API ConstOdeVector odeJointGetAccumulatedTorque(int jointId, int bodyIdx); //bodyIdx either 0 or 1, denoting the connected bodies
EXPORT_API ConstOdeVector odeJointGetAccumulatedForce(int jointId, int bodyIdx); //bodyIdx either 0 or 1, denoting the connected bodies

bool EXPORT_API stepOde(float stepSize, bool breakOnErrors);

bool EXPORT_API stepOdeFast(float stepSize, bool breakOnErrors=true);

//saves the master thread state
bool EXPORT_API saveOdeState(int slot = 0, int sourceContext=0);
//restores the thread state from the last save
bool EXPORT_API restoreOdeState(int slot = 0, bool breakOnErrors = false);

bool EXPORT_API saveOdeStateToFile(const char *filename, int slot, int extraFloatsAmount, float *extraFloats);
bool EXPORT_API loadOdeStateFromFile(const char *filename, int slot, int extraFloatsAmount, float *extraFloats);

void EXPORT_API odeFixUnityRotation(int geomId);

void EXPORT_API odeSetContactSoftCFM(float cfm);

void EXPORT_API odeSetFrictionCoefficient(float mu);

// Body: Creating and Destroying Bodies

int  EXPORT_API odeBodyCreate();
void EXPORT_API odeBodyDestroy(int bodyId);

// Body: Position and orientation

void EXPORT_API odeBodySetPosition(int bodyId, float x, float y, float z);
//bool EXPORT_API odeBodySetRotation(int bodyId, vmml::mat3f *m);
bool EXPORT_API odeBodySetQuaternion(int bodyId, ConstOdeQuaternion q, bool breakOnErrors);
void EXPORT_API odeBodySetLinearVel(int bodyId, float x, float y, float z);
void EXPORT_API odeBodySetAngularVel(int bodyId, float x, float y, float z);

EXPORT_API ConstOdeVector odeBodyGetPosition(int bodyId);
#ifndef SWIG
//return value points to ODE's internal 12 element rotation matrix - swig won't convert to a Unity Vector3
const dReal *odeBodyGetRotation (int bodyId);
void EXPORT_API odeBodySetRotation(int bodyId, dReal* R);
#endif
//note that the returned value points to a 12 element dReal array instead of the usual 3 element vector
EXPORT_API ConstOdeQuaternion odeBodyGetQuaternion(int bodyId);
EXPORT_API ConstOdeVector odeBodyGetLinearVel(int bodyId);
EXPORT_API ConstOdeVector odeBodyGetAngularVel(int bodyId);

// Body: Mass and force

// TODO: expose dMass struct
void EXPORT_API odeBodySetMass(int bodyId, float mass);
float EXPORT_API odeBodyGetMass(int bodyId);

EXPORT_API void odeBodyAddForce(int bodyId, ConstOdeVector f);
EXPORT_API void odeBodyAddTorque(int bodyId, ConstOdeVector f);
EXPORT_API void odeBodyAddRelForce(int bodyId, ConstOdeVector f);
EXPORT_API void odeBodyAddRelTorque(int bodyId, ConstOdeVector f);
EXPORT_API void odeBodyAddForceAtPos(int bodyId, ConstOdeVector f, ConstOdeVector p);
EXPORT_API void odeBodyAddForceAtRelPos(int bodyId, ConstOdeVector f, ConstOdeVector p);
EXPORT_API void odeBodyAddRelForceAtPos(int bodyId, ConstOdeVector f, ConstOdeVector p);
EXPORT_API void odeBodyAddRelForceAtRelPos(int bodyId, ConstOdeVector f, ConstOdeVector p);

EXPORT_API void odeBodySetForce(int bodyId, ConstOdeVector f);
EXPORT_API void odeBodySetTorque(int bodyId, ConstOdeVector f);
EXPORT_API ConstOdeVector odeBodyGetForce(int bodyId);
EXPORT_API ConstOdeVector odeBodyGetTorque(int bodyId);

// Body: Kinematic State

bool EXPORT_API odeBodySetDynamic(int bodyId);
bool EXPORT_API odeBodySetKinematic(int bodyId);
bool EXPORT_API odeBodyIsKinematic(int bodyId);

// Body: Utility


// Geom: General geom functions

EXPORT_API void odeGeomDestroy(int geomId);

bool EXPORT_API odeGeomSetBody(int geomId, int bodyId);
int EXPORT_API odeGeomGetBody(int geomId);

void EXPORT_API odeGeomSetPosition(int geomId, float x, float y, float z);
#ifndef SWIG //don't support rotation matrices, only quaternions through SWIG
void EXPORT_API odeGeomSetRotation(int geomId, dReal* R);
#endif

void EXPORT_API odeGeomSetQuaternion(int geomId, ConstOdeQuaternion q);

EXPORT_API ConstOdeVector odeGeomGetPosition(int geomId);
#ifndef SWIG
//return value points to ODE's internal 12 element rotation matrix - swig won't convert to a Unity Vector3
const dReal *odeGeomGetRotation (int geomId);
#endif
//const dReal * dGeomGetRotation (dGeomID);
EXPORT_API void odeGeomGetQuaternion(int geomId, OdeQuaternion result);

void EXPORT_API odeGeomSetOffsetWorldPosition(int geomId, float x, float y, float z);

EXPORT_API void odeGeomSetCategoryBits (int geomId, unsigned long bits);
EXPORT_API void odeGeomSetCollideBits (int geomId, unsigned long bits);
EXPORT_API unsigned long odeGeomGetCategoryBits (int geomId);
EXPORT_API unsigned long odeGeomGetCollideBits (int geomId);

// Geometry classes

// Geom: Sphere class
EXPORT_API int odeCreateSphere(float radius);
EXPORT_API void odeGeomSphereSetRadius(int geomId, float radius);
EXPORT_API float odeGeomSphereGetRadius(int geomId);
EXPORT_API float dGeomSpherePointDepth(int geomId, float x, float y, float z);

// Geom: Box class

EXPORT_API int odeCreateBox(float lx, float ly, float lz);
EXPORT_API void odeGeomBoxSetLengths(int geomId, float lx, float ly, float lz);
EXPORT_API void odeGeomBoxGetLengths(int geomId, float &lx, float &ly, float &lz);
EXPORT_API float dGeomBoxPointDepth(int geomId, float x, float y, float z);

// Geom: Plane class

int   EXPORT_API odeCreatePlane(int spaceId, float a, float b, float c, float d);
void  EXPORT_API odeGeomPlaneSetParams(int geomId, float a, float b, float c, float d);
//OdeVec4 EXPORT_API odeGeomPlaneGetParams(int geomId);
float EXPORT_API odeGeomPlanePointDepth(int geomId, float x, float y, float z);

// Geom: Heightfield class
int   EXPORT_API odeCreateHeightfield(const float *heightData, float width, float depth, int widthSamples, int depthSamples, float scale, float offset, float thickness, int wrap);

// Geom: Capsule class

int   EXPORT_API odeCreateCapsule(int spaceId, float radius, float length);
void  EXPORT_API odeGeomCapsuleSetParams(int geomId, float radius, float length);
void  EXPORT_API odeGeomCapsuleGetParams(int geomId, float &radius, float &length);
float EXPORT_API odeGeomCapsulePointDepth(int geomId, float x, float y, float z);



///////////////////////////////////////////////////////////////////////////////
// Joints
///////////////////////////////////////////////////////////////////////////////

// Joint: Creating and Destroying Joints

int EXPORT_API odeJointCreateBall();
int EXPORT_API odeJointCreateHinge();
int EXPORT_API odeJointCreateSlider();
//dJointID dJointCreateContact (dWorldID, dJointGroupID, const dContact *);
int EXPORT_API odeJointCreateUniversal();
int EXPORT_API odeJointCreateHinge2();
int EXPORT_API odeJointCreatePR();
int EXPORT_API odeJointCreatePU();
int EXPORT_API odeJointCreatePiston();
int EXPORT_API odeJointCreateFixed();
int EXPORT_API odeJointCreateAMotor();
int EXPORT_API odeJointCreateLMotor();
int EXPORT_API odeJointCreatePlane2D();

// Joint: Miscellaneous Joint Functions

void EXPORT_API odeJointAttach(int jointId, int bodyId1, int bodyId2);
int EXPORT_API odeJointGetType(int jointId);
int EXPORT_API odeJointGetBody(int jointId, int index);

// Joint: Joint parameter setting functions

// Joint: Ball and Socket parameters

void EXPORT_API odeJointSetBallAnchor(int jointId, float x, float y, float z);
void EXPORT_API odeJointGetBallAnchor(int jointId, float &x, float &y, float &z);
void EXPORT_API odeJointGetBallAnchor2(int jointId, float &x, float &y, float &z);

// Joint: Hinge parameters

void  EXPORT_API odeJointSetHingeAnchor(int jointId, float x, float y, float z);
void  EXPORT_API odeJointSetHingeAxis(int jointId, float x, float y, float z);
void EXPORT_API odeJointSetHinge2Anchor(int jointId, float x, float y, float z);
void  EXPORT_API odeJointSetHinge2Axis(int jointId, int axis, float x, float y, float z);
void EXPORT_API odeJointGetHingeAnchor(int jointId, float &x, float &y, float &z);
void EXPORT_API odeJointGetHingeAnchor2(int jointId, float &x, float &y, float &z);
void EXPORT_API odeJointGetHingeAxis(int jointId, float &x, float &y, float &z);
float EXPORT_API odeJointGetHingeAngle(int jointId);
float EXPORT_API odeJointGetHingeAngleRate(int jointId);
float EXPORT_API odeJointGetHingeAngleFromBodyRotations(int jointId, ConstOdeQuaternion q1, ConstOdeQuaternion q2);

// Joint: AMotor parameters

void  EXPORT_API odeJointSetAMotorMode(int jointId, int mode);
int   EXPORT_API odeJointGetAMotorMode(int jointId);
void  EXPORT_API odeJointSetAMotorNumAxes(int jointId, int num);
int   EXPORT_API odeJointGetAMotorNumAxes(int jointId);
void  EXPORT_API odeJointSetAMotorAxis(int jointId, int anum, int rel, float x, float y, float z);
void  EXPORT_API odeJointGetAMotorAxis(int jointId, int anum, float &x, float &y, float &z);
int   EXPORT_API odeJointGetAMotorAxisRel(int jointId, int anum);
void  EXPORT_API odeJointSetAMotorAngle(int jointId, int anum, float angle);
float EXPORT_API odeJointGetAMotorAngle(int jointId, int anum);
float EXPORT_API odeJointGetAMotorAngleRate(int jointId, int anum);
//maps body rotations to angular motor angles
void EXPORT_API odeJointGetAMotorAnglesFromBodyRotations(int jointId, ConstOdeQuaternion q1, ConstOdeQuaternion q2, OdeVector result);
// Joint: Parameter functions

void  EXPORT_API odeJointSetBallParam(int jointId, int parameter, float value);
void  EXPORT_API odeJointSetHingeParam(int jointId, int parameter, float value);
void  EXPORT_API odeJointSetSliderParam(int jointId, int parameter, float value);
void  EXPORT_API odeJointSetHinge2Param(int jointId, int parameter, float value);
void  EXPORT_API odeJointSetUniversalParam(int jointId, int parameter, float value);
void  EXPORT_API odeJointSetAMotorParam(int jointId, int parameter, float value);
void  EXPORT_API odeJointSetLMotorParam(int jointId, int parameter, float value);
void  EXPORT_API odeJointSetPRParam(int jointId, int parameter, float value);
void  EXPORT_API odeJointSetPUParam(int jointId, int parameter, float value);
void  EXPORT_API odeJointSetPistonParam(int jointId, int parameter, float value);
float EXPORT_API odeJointGetBallParam(int jointId, int parameter);
float EXPORT_API odeJointGetHingeParam(int jointId, int parameter);
float EXPORT_API odeJointGetSliderParam(int jointId, int parameter);
float EXPORT_API odeJointGetHinge2Param(int jointId, int parameter);
float EXPORT_API odeJointGetUniversalParam(int jointId, int parameter);
float EXPORT_API odeJointGetAMotorParam(int jointId, int parameter);
float EXPORT_API odeJointGetLMotorParam(int jointId, int parameter);
float EXPORT_API odeJointGetPRParam(int jointId, int parameter);
float EXPORT_API odeJointGetPUParam(int jointId, int parameter);
float EXPORT_API odeJointGetPistonParam(int jointId, int parameter);


// Joint: Setting Joint Torques/Forces Directly

EXPORT_API void odeJointAddHingeTorque(int jointId, float torque);
EXPORT_API void odeJointAddUniversalTorques(int jointId, float torque1, float torque2);
EXPORT_API void odeJointAddSliderForce(int jointId, float force);
EXPORT_API void odeJointAddHinge2Torques(int jointId, float torque1, float torque2);
EXPORT_API void odeJointAddAMotorTorques(int jointId, float torque1, float torque2, float torque3);

//Some shortcuts to avoid separate calls for, e.g., setting/getting motor angles and fmax
//For hinges, will use
EXPORT_API void odeJointSetFmax(int jointId, float fmax1, float fmax2, float fmax3);
EXPORT_API void odeJointSetFixed(int jointId);
EXPORT_API void odeJointGetMotorAnglesDegrees(int jointId, OdeVector result);
EXPORT_API void odeJointSetAMotorVelocitiesDegreesPerSecond(int jointId, float vel1, float vel2, float vel3);
EXPORT_API void odeJointSetAMotorVelocitiesRadiansPerSecond(int jointId, float vel1, float vel2, float vel3);
//Checks if the joint at a stop, and clamps the corresponding proposed motor velocity (inOut) so that if it's applied to the motor, it doesn't drive the joint against the limit.
EXPORT_API void clampControlVelocitiesAtStops(int nJoints, int *jointIds, float *velocities);
//sets several motor velocities at once, given as a float array with 1 float per hinge, 3 floats per amotor. Assumes degrees per seconds.
EXPORT_API void setMotorVelocities(int nJoints, int *jointIds, const float *velocities);

///////////////////////////////////////////////////////////////////////////////
// Support Functions
///////////////////////////////////////////////////////////////////////////////

// Support: Mass functions

EXPORT_API void odeMassSetSphere(int bodyId, float density, float radius);
EXPORT_API void odeMassSetSphereTotal(int bodyId, float total_mass, float radius);

EXPORT_API void odeMassSetCapsule(int bodyId, float density, float radius, float length);
EXPORT_API void odeMassSetCapsuleTotal(int bodyId, float total_mass, float radius, float length);

EXPORT_API void odeMassSetCylinder(int bodyId, float density, float radius, float length);
EXPORT_API void odeMassSetCylinderTotal(int bodyId, float total_mass, float radius, float length);

EXPORT_API void odeMassSetBox(int bodyId, float density, float lx, float ly, float lz);
EXPORT_API void odeMassSetBoxTotal(int bodyId, float total_mass, float lx, float ly, float lz);

// Returns the maximum contact speed of the body, -1 if the body is in no contact
EXPORT_API float odeGetMaxContactSpeed(int bodyId);
///////////////////////////////////////////////////////////////////////////////
// World
///////////////////////////////////////////////////////////////////////////////

void EXPORT_API odeWorldSetGravity(float x,float y, float z);
void EXPORT_API odeWorldSetCFM(float cfm);
void EXPORT_API odeWorldSetERP(float erp);
void EXPORT_API odeWorldSetContactMaxCorrectingVel(float vel);
void EXPORT_API odeWorldSetContactSurfaceLayer (float depth);

int EXPORT_API odeGetContactCount();
void EXPORT_API odeGetContactInfo(int index, int &body1Id, int &body2Id,  OdeVector out_pos, OdeVector out_normal, OdeVector out_vel);

int EXPORT_API odeRaycast(float px, float py, float pz, float dx, float dy, float dz, float length, OdeVector out_pos, float & out_depth, unsigned long collideBits = 0xFFFFFFFF, unsigned long categoryBits = 0xFFFFFFFF);
//version of the raycast that doesn't return the body identifier, simply returns true if a geom hit
bool EXPORT_API odeRaycastGeom(float px, float py, float pz, float dx, float dy, float dz, float length, OdeVector out_pos, float & out_depth, unsigned long collideBits = 0xFFFFFFFF, unsigned long categoryBits = 0xFFFFFFFF);
//get contact between bodies. returns false if no contact
bool EXPORT_API odeGetContact(int body1Id, int body2Id,  OdeVector out_pos, OdeVector out_normal, OdeVector out_vel);
//get contact between geoms. returns false if no contact
bool EXPORT_API odeGetGeomContact(int geom1Id, int geom2Id,  OdeVector out_pos, OdeVector out_normal, OdeVector out_vel);
void *odeGetSpace();