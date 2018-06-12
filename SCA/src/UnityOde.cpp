

#include <algorithm>
#include <vector>
#include <csignal>
#include <ctime>
#include <map>
#include <iostream>
#include <fstream>
#include <ode/ode.h>
//#pragma warning(disable: 4003)
#include "UnityOde.h"
#include "MathUtils.h"
#include <assert.h>
#include "ode/src/joints/amotor.h"
#include "ode/src/joints/ball.h"
#include "UnityOde_internal.h"
//#include "Debug.h"
#include "error_internal.h"

using namespace AaltoGames;

static std::map<int, SaveState> saves;

static std::vector<OdeThreadContext> contexts;
static int counter = 1;  //start from 1 so that 0 denotes no object
static bool s_initialized = false;

//thread local storage: every thread knows which context it is manipulating and getting data from
__declspec(thread) static int s_setIdx = ALLTHREADS;
__declspec(thread) static int s_getIdx = 0;

#define ITERATE_THREADS(threadIdx) int firstThread=s_setIdx,lastThread=s_setIdx; \
	if (firstThread<0){firstThread=0; lastThread=(int)contexts.size()-1;} \
	for (int threadIdx=firstThread; threadIdx<=lastThread; threadIdx++)

OdeThreadContext *getOdeThreadContext(int idx)
{
	return &contexts[idx];
}

void erase_contacts(int idx)
{
	SaveState & save = ::saves[idx];
	save.contacts.clear();
}




// this is called by dSpaceCollide when two objects in space are
// potentially colliding.

static void nearCallback(void *data, dGeomID o1, dGeomID o2)
{
	int i, n;

	OdeThreadContext *context = (OdeThreadContext *)data;


	dContact contact[NUM_CONTACTS];

	// exit without doing anything if the two bodies are connected by a joint 
	dBodyID b1, b2;
	b1 = dGeomGetBody(o1);
	b2 = dGeomGetBody(o2);
	if (b1 && b2 && dAreConnected(b1, b2)) return;

	//get collision points
	n = dCollide(o1, o2, NUM_CONTACTS, &contact[0].geom, sizeof(dContact));

	if (n > 0) {
		for (i = 0; i<n; i++) {

			int cl1 = dGeomGetClass(contact[i].geom.g1);
			int cl2 = dGeomGetClass(contact[i].geom.g2);
			if ((cl1 != dPlaneClass) && (cl2 != dPlaneClass) && (cl1 != dSphereClass) && (cl2 = dSphereClass))
			{
				continue;
			}

			contact[i].surface.mode = 0;//dContactSlip1 | dContactSlip2;// |
										//		dContactSoftERP | dContactSoftCFM | dContactApprox1;

			if (context->contactSoftCFM > 0)
			{
				contact[i].surface.mode |= dContactSoftCFM;
				contact[i].surface.soft_cfm = context->contactSoftCFM;
			}

			contact[i].surface.mode |= dContactApprox1;
			contact[i].surface.mu = context->frictionCoefficient;//200;
																 //contact[i].surface.slip1 = 0.001f;
																 //contact[i].surface.slip2 = 0.001f;
			contact[i].surface.bounce = 0.0f;
			contact[i].surface.soft_erp = 0.8f;
			dJointID c = dJointCreateContact(context->world, context->contactGroup, &contact[i]);
			dBodyID body1 = dGeomGetBody(contact[i].geom.g1);
			dBodyID body2 = dGeomGetBody(contact[i].geom.g2);
			//store the body velocities at the contact point, used for analyzing damage caused by collisions
			dVector3 body1VelAtContact, body2VelAtContact;
			dSetZero(body1VelAtContact, 3);
			dSetZero(body2VelAtContact, 3);
			if (body1 != NULL)
				dBodyGetPointVel(body1, contact[i].geom.pos[0], contact[i].geom.pos[1], contact[i].geom.pos[2], body1VelAtContact);
			if (body2 != NULL)
				dBodyGetPointVel(body2, contact[i].geom.pos[0], contact[i].geom.pos[1], contact[i].geom.pos[2], body2VelAtContact);
			dAddVectors3(contact[i].vel, body1VelAtContact, body2VelAtContact);
			dJointAttach(c, body1, body2);
			context->contacts[context->contacts.size()] = contact[i];
		}
	}
}


extern "C"
{
	void printTime(std::ostream & out)
	{
		char dateStr[9];
		char timeStr[9];
		_strdate(dateStr);
		_strtime(timeStr);

		out << "[" << dateStr << " " << timeStr << "] ";
	}

	void output(const std::string & msg)
	{
		std::ofstream out;
		out.open("UnityOde.log", std::ios::out | std::ios::app);

		printTime(out);
		out << msg << std::endl;
		out.close();
	}

	void __cdecl abortHandler(int signal)
	{
		output("Received SIGABRT signal");
	}

	void __cdecl sigsegvHandler(int signal)
	{
		output("Received SIGSEGV signal");
		exit(1);
	}

	void terminateHandler()
	{
		output("Received terminate()");
		exit(1);
	}

	void unexpectedHandler()
	{
		output("Received unexpected()");
		exit(1);
	}


	LONG WINAPI UnhandledExceptionFilterOde(PEXCEPTION_POINTERS pExceptionPtrs)
	{
		output("Received unhandled SEH exception");

		stack_trace trace(pExceptionPtrs->ContextRecord, 0);
		output(trace.to_string());

		// Execute default exception handler next
		return EXCEPTION_EXECUTE_HANDLER;
	}


	void setupProcessExceptionHandlers()
	{
		SetUnhandledExceptionFilter(UnhandledExceptionFilterOde);

		std::signal(SIGABRT, &abortHandler);
	}

	void setupThreadExceptionHandlers()
	{
		set_terminate(terminateHandler);
		set_unexpected(unexpectedHandler);

		std::signal(SIGSEGV, &sigsegvHandler);
	}
}

//Sets the number of threads used. All bodies etc. will be duplicated for each thread
bool initOde(int numThreads)
{
	if (s_initialized)
		return true;

	setupProcessExceptionHandlers();

	contexts.resize(numThreads);
	dInitODE2(0);
	allocateODEDataForThread();
	for (int i = 0; i<numThreads; i++)
	{
		contexts[i].world = dWorldCreate();
		contexts[i].space = dHashSpaceCreate(0);
		contexts[i].contactGroup = dJointGroupCreate(0);
		contexts[i].jointGroups[0] = 0; //joint groups not currently used
		contexts[i].bodies[0] = (dBodyID)0;
		contexts[i].joints[0] = (dJointID)0;
		contexts[i].contactSoftCFM = 0;
		dWorldSetGravity(contexts[i].world, 0, -9.81f, 0);
	}
	s_initialized = true;
	return true;
}
bool uninitOde()
{
	if (s_initialized == false)
		return true;
	for (size_t i = 0; i<contexts.size(); i++)
	{
		//delete bodies
		BodyContainer::const_iterator bodyIter;
		for (bodyIter = contexts[i].bodies.begin(); bodyIter != contexts[i].bodies.end(); ++bodyIter) {
			if (!bodyIter->second) continue;
			dBodyDestroy(bodyIter->second);
		}
		contexts[i].bodies.clear();

		//delete geoms
		GeomContainer::const_iterator geomIter;
		for (geomIter = contexts[i].geoms.begin(); geomIter != contexts[i].geoms.end(); ++geomIter) {
			if (!geomIter->second) continue;
			dGeomDestroy(geomIter->second);
		}
		contexts[i].geoms.clear();

		//delete contact joint group, space and world
		dJointGroupDestroy(contexts[i].contactGroup);
		dSpaceDestroy(contexts[i].space);
		dWorldDestroy(contexts[i].world);
	}
	contexts.clear();
	s_initialized = false;
	counter = 0;
	dCloseODE();
	return true;
}

bool initialized()
{
	return s_initialized;
}

void handleOdeError(int errnum, const char *msg, va_list ap)
{
	ITERATE_THREADS(threadIdx)
	{
		contexts[threadIdx].errorOccurred = true;
	}
}

void odeRandSetSeed(unsigned long s)
{
	ITERATE_THREADS(i)
	{
		dRandSetSeed(s);
	}
}

/*int EXPORT_API createOdeBody(int geomId, float mass, bool isKinematic)
{
int id=counter++;
for (size_t i=0; i<contexts.size(); i++)
{
dBodyID body=dBodyCreate(contexts[i].world);
dMass dmass;
//dGeomCapsuleGetParams(contexts[i].geoms[geomIdx],
dMassSetSphereTotal(&dmass,mass,0.1f);
dBodySetMass(body,&dmass);
dGeomID geom=contexts[i].geoms[geomId];
dGeomSetBody(geom,body);
dMatrix3 R;
dRFromEulerAngles(R,M_PI/2.0f,0,0);
dGeomSetOffsetRotation(geom,R);  //assuming a capsule, try to match Unity's rotations.
contexts[i].bodies[id]=body;
}
return id;
}*/

void odeFixUnityRotation(int geomId)
{
	ITERATE_THREADS(i)
	{
		dGeomID geom = contexts[i].geoms[geomId];
		dMatrix3 R;
		dRFromEulerAngles(R, M_PI / 2.0f, 0, 0);
		dGeomSetOffsetRotation(geom, R);  //assuming a capsule, try to match Unity's rotations.
	}
}

void odeSetContactSoftCFM(float cfm)
{
	ITERATE_THREADS(threadIdx)
	{
		OdeThreadContext & c = contexts[threadIdx];
		c.contactSoftCFM = cfm;
	}
}

void odeSetFrictionCoefficient(float mu)
{
	ITERATE_THREADS(threadIdx)
	{
		OdeThreadContext & c = contexts[threadIdx];
		c.frictionCoefficient = mu;
	}
}

// TODO: worldId is not currently used
int odeBodyCreate()
{
	int id = counter++;
	ITERATE_THREADS(i)
	{
		dBodyID body = dBodyCreate(contexts[i].world);
		contexts[i].bodies[id] = body;
		dBodySetAutoDisableFlag(body, 0); //needed for simulation causality
	}
	return id;
}

void odeBodyDestroy(int bodyId)
{
	ITERATE_THREADS(i)
	{
		OdeThreadContext & c = contexts[i];
		dBodyDestroy(c.bodies[bodyId]);
		c.bodies.erase(bodyId);
	}
}

void odeBodySetPosition(int bodyId, float x, float y, float z)
{
	ITERATE_THREADS(threadIdx)
	{
		OdeThreadContext & c = contexts[threadIdx];
		dBodyID body = c.bodies[bodyId];
		dBodySetPosition(body, x, y, z);
	}
}

void odeBodySetRotation(int bodyId, dReal* R)
{
	ITERATE_THREADS(threadIdx)
	{
		OdeThreadContext & c = contexts[threadIdx];
		dBodyID body = c.bodies[bodyId];
		dBodySetRotation(body, R);
	}
}


bool odeBodySetQuaternion(int bodyId, ConstOdeQuaternion q, bool breakOnErrors)
{
	bool isError = false;

	ITERATE_THREADS(threadIdx)
	{
		OdeThreadContext & c = contexts[threadIdx];
		dBodyID body = c.bodies[bodyId];

		if (!breakOnErrors) {
			contexts[threadIdx].errorOccurred = false;
			dSetDebugHandler(handleOdeError);
		}

		dBodySetQuaternion(body, q);

		if (!breakOnErrors) {
			dSetDebugHandler(0);
			isError |= contexts[threadIdx].errorOccurred;
		}
	}

	return !isError;
}

void odeBodySetLinearVel(int bodyId, float x, float y, float z)
{
	ITERATE_THREADS(threadIdx)
	{
		OdeThreadContext & c = contexts[threadIdx];
		dBodyID body = c.bodies[bodyId];
		dBodySetLinearVel(body, x, y, z);
	}
}

void odeBodySetAngularVel(int bodyId, float x, float y, float z)
{
	ITERATE_THREADS(threadIdx)
	{
		OdeThreadContext & c = contexts[threadIdx];
		dBodyID body = c.bodies[bodyId];
		dBodySetAngularVel(body, x, y, z);
	}
}

ConstOdeVector odeBodyGetPosition(int bodyId)
{
	OdeThreadContext & c = contexts[s_getIdx];
	return dBodyGetPosition(c.bodies[bodyId]);
}

// TODO: fix this, conversion from dMatrix3 (which is actually dReal[16] to vmml::mat3f
/*bool odeBodyGetRotation(int bodyId, vmml::mat3f *m)
{
OdeThreadContext & c=contexts[threadIdx];
const dReal *rot = dBodyGetRotation(c.bodies[bodyId]);
memcpy(m, rot, sizeof(dMatrix3));
return true;
}*/

ConstOdeVector odeBodyGetRotation(int bodyId)
{
	OdeThreadContext & c = contexts[s_getIdx];
	return dBodyGetRotation(c.bodies[bodyId]);
}

ConstOdeQuaternion odeBodyGetQuaternion(int bodyId)
{
	OdeThreadContext & c = contexts[s_getIdx];
	return dBodyGetQuaternion(c.bodies[bodyId]);
}

ConstOdeVector odeBodyGetLinearVel(int bodyId)
{
	OdeThreadContext & c = contexts[s_getIdx];
	return dBodyGetLinearVel(c.bodies[bodyId]);
}

ConstOdeVector odeBodyGetAngularVel(int bodyId)
{
	OdeThreadContext & c = contexts[s_getIdx];
	return dBodyGetAngularVel(c.bodies[bodyId]);
}

void odeBodySetMass(int bodyId, float mass)
{
	ITERATE_THREADS(i)
	{
		dMass dmass;
		dBodyID body = contexts[i].bodies[bodyId];
		dBodyGetMass(body, &dmass);
		dmass.mass = mass;
		dBodySetMass(body, &dmass);
	}
}

float odeBodyGetMass(int bodyId)
{
	OdeThreadContext & c = contexts[s_getIdx];
	dMass dmass;
	dBodyGetMass(c.bodies[bodyId], &dmass);
	return dmass.mass;
}

void odeBodyAddForce(int bodyId, ConstOdeVector f)
{
	ITERATE_THREADS(threadIdx)
	{
		OdeThreadContext & c = contexts[threadIdx];
		dBodyID body = c.bodies[bodyId];
		dBodyAddForce(body, f[0], f[1], f[2]);
	}
}
void odeBodyAddTorque(int bodyId, ConstOdeVector f)
{
	ITERATE_THREADS(threadIdx)
	{
		OdeThreadContext & c = contexts[threadIdx];
		dBodyID body = c.bodies[bodyId];
		dBodyAddTorque(body, f[0], f[1], f[2]);
	}
}
void odeBodyAddRelForce(int bodyId, ConstOdeVector f)
{
	ITERATE_THREADS(threadIdx)
	{
		OdeThreadContext & c = contexts[threadIdx];
		dBodyID body = c.bodies[bodyId];
		dBodyAddRelForce(body, f[0], f[1], f[2]);
	}
}
void odeBodyAddRelTorque(int bodyId, ConstOdeVector f)
{
	ITERATE_THREADS(threadIdx)
	{
		OdeThreadContext & c = contexts[threadIdx];
		dBodyID body = c.bodies[bodyId];
		dBodyAddRelTorque(body, f[0], f[1], f[2]);
	}
}

void odeBodyAddForceAtPos(int bodyId, ConstOdeVector f, ConstOdeVector p)
{
	ITERATE_THREADS(threadIdx)
	{
		OdeThreadContext & c = contexts[threadIdx];
		dBodyID body = c.bodies[bodyId];
		dBodyAddForceAtPos(body, f[0], f[1], f[2], p[0], p[1], p[2]);
	}
}
void odeBodyAddForceAtRelPos(int bodyId, ConstOdeVector f, ConstOdeVector p)
{
	ITERATE_THREADS(threadIdx)
	{
		OdeThreadContext & c = contexts[threadIdx];
		dBodyID body = c.bodies[bodyId];
		dBodyAddForceAtRelPos(body, f[0], f[1], f[2], p[0], p[1], p[2]);
	}
}
void odeBodyAddRelForceAtPos(int bodyId, ConstOdeVector f, ConstOdeVector p)
{
	ITERATE_THREADS(threadIdx)
	{
		OdeThreadContext & c = contexts[threadIdx];
		dBodyID body = c.bodies[bodyId];
		dBodyAddRelForceAtPos(body, f[0], f[1], f[2], p[0], p[1], p[2]);
	}
}
void odeBodyAddRelForceAtRelPos(int bodyId, ConstOdeVector f, ConstOdeVector p)
{
	ITERATE_THREADS(threadIdx)
	{
		OdeThreadContext & c = contexts[threadIdx];
		dBodyID body = c.bodies[bodyId];
		dBodyAddRelForceAtRelPos(body, f[0], f[1], f[2], p[0], p[1], p[2]);
	}
}

ConstOdeVector odeBodyGetForce(int bodyId)
{
	OdeThreadContext & c = contexts[s_getIdx];
	return dBodyGetForce(c.bodies[bodyId]);
}
ConstOdeVector odeBodyGetTorque(int bodyId)
{
	OdeThreadContext & c = contexts[s_getIdx];
	return dBodyGetTorque(c.bodies[bodyId]);
}
void odeBodySetForce(int bodyId, ConstOdeVector f)
{
	ITERATE_THREADS(threadIdx)
	{
		OdeThreadContext & c = contexts[threadIdx];
		dBodyID body = c.bodies[bodyId];
		dBodySetForce(body, f[0], f[1], f[2]);
	}
}
void odeBodySetTorque(int bodyId, ConstOdeVector f)
{
	ITERATE_THREADS(threadIdx)
	{
		OdeThreadContext & c = contexts[threadIdx];
		dBodyID body = c.bodies[bodyId];
		dBodySetTorque(body, f[0], f[1], f[2]);
	}
}

bool odeBodySetDynamic(int bodyId)
{
	ITERATE_THREADS(i)
	{
		dBodyID body = contexts[i].bodies[bodyId];
		dBodySetDynamic(body);
	}
	return true;
}

bool odeBodySetKinematic(int bodyId)
{
	ITERATE_THREADS(i)
	{
		dBodyID body = contexts[i].bodies[bodyId];
		dBodySetKinematic(body);
	}
	return true;
}
bool odeBodyIsKinematic(int bodyId)
{
	OdeThreadContext & c = contexts[s_getIdx];
	return dBodyIsKinematic(c.bodies[bodyId]) != 0;
}

/*OdeVec3 odeBodyGetRelPointPos(int bodyId, OdeVec3 *p)
{
OdeThreadContext & c = contexts[s_getIdx];
dVector3 dv;
dBodyGetRelPointPos(c.bodies[bodyId], p->x, p->y, p->z, dv);
return dv;
}

OdeVec3 odeBodyGetRelPointVel(int bodyId, OdeVec3 *p)
{
OdeThreadContext & c = contexts[s_getIdx];
dVector3 dv;
dBodyGetRelPointVel(c.bodies[bodyId], p->x, p->y, p->z, dv);
return dv;
}

OdeVec3 odeBodyGetPointVel(int bodyId, OdeVec3 *p)
{
OdeThreadContext & c = contexts[s_getIdx];
dVector3 dv;
dBodyGetPointVel(c.bodies[bodyId], p->x, p->y, p->z, dv);
return dv;
}
OdeVec3 odeBodyGetPosRelPoint(int bodyId, OdeVec3 *p)
{
OdeThreadContext & c = contexts[s_getIdx];
dVector3 dv;
dBodyGetPosRelPoint(c.bodies[bodyId], p->x, p->y, p->z, dv);
return dv;
}

OdeVec3 odeBodyVectorToWorld(int bodyId, OdeVec3 *p)
{
OdeThreadContext & c = contexts[s_getIdx];
dVector3 dv;
dBodyVectorToWorld(c.bodies[bodyId], p->x, p->y, p->z, dv);
return dv;
}

OdeVec3 odeBodyVectorFromWorld(int bodyId, OdeVec3 *p)
{
OdeThreadContext & c = contexts[s_getIdx];
dVector3 dv;
dBodyVectorFromWorld(c.bodies[bodyId], p->x, p->y, p->z, dv);
return dv;
}*/

// Geometry

void odeGeomDestroy(int geomId)
{
	ITERATE_THREADS(threadIdx)
	{
		OdeThreadContext & c = contexts[threadIdx];
		dGeomDestroy(c.geoms[geomId]);
		c.geoms.erase(geomId);
	}
}

bool odeGeomSetBody(int geomId, int bodyId)
{
	ITERATE_THREADS(i)
	{
		dGeomID geom = contexts[i].geoms[geomId];
		dBodyID body = contexts[i].bodies[bodyId];
		dGeomSetBody(geom, body);
	}
	return true;
}

int odeGeomGetBody(int geomId)
{
	OdeThreadContext & c = contexts[s_getIdx];
	dBodyID body = dGeomGetBody(c.geoms[geomId]);

	// TODO: slow linear search, should find a more efficient way
	BodyContainer::const_iterator it;
	for (it = c.bodies.begin(); it != c.bodies.end(); ++it)
	{
		if (it->second == body)
		{
			return it->first;
		}
	}
	return -1;
}

void odeGeomSetPosition(int geomId, float x, float y, float z)
{
	ITERATE_THREADS(threadIdx)
	{
		OdeThreadContext & c = contexts[threadIdx];
		dGeomID geom = c.geoms[geomId];
		dGeomSetPosition(geom, x, y, z);
	}
}

void odeGeomSetRotation(int geomId, dReal* R)
{
	ITERATE_THREADS(threadIdx)
	{
		OdeThreadContext & c = contexts[threadIdx];
		dGeomID geom = c.geoms[geomId];
		dGeomSetRotation(geom, R);
	}
}

void odeGeomSetQuaternion(int geomId, ConstOdeQuaternion q)
{
	ITERATE_THREADS(threadIdx)
	{
		OdeThreadContext & c = contexts[s_getIdx];
		dGeomID geom = c.geoms[geomId];
		dGeomSetQuaternion(geom, q);
	}
}

ConstOdeVector odeGeomGetPosition(int geomId)
{
	OdeThreadContext & c = contexts[s_getIdx];
	return dGeomGetPosition(c.geoms[geomId]);
}


const dReal * odeGeomGetRotation(int geomId)
{
	OdeThreadContext & c = contexts[s_getIdx];
	return dGeomGetRotation(c.geoms[geomId]);
}


void odeGeomGetQuaternion(int geomId, OdeQuaternion result)
{
	OdeThreadContext & c = contexts[s_getIdx];
	dGeomGetQuaternion(c.geoms[geomId], result);
}

void odeGeomSetOffsetWorldPosition(int geomId, float x, float y, float z)
{
	OdeThreadContext & c = contexts[s_getIdx];
	dGeomSetOffsetWorldPosition(c.geoms[geomId], x, y, z);
}

// GEOMETRY


// Geom: Box class

int odeCreateBox(float lx, float ly, float lz)
{
	int id = counter++;
	ITERATE_THREADS(threadIdx)
	{
		dGeomID geom = dCreateBox(contexts[threadIdx].space, lx, ly, lz);
		contexts[threadIdx].geoms[id] = geom;
	}
	return id;
}

void odeGeomBoxSetLengths(int geomId, float lx, float ly, float lz)
{
	ITERATE_THREADS(threadIdx)
	{
		OdeThreadContext & c = contexts[threadIdx];
		dGeomBoxSetLengths(c.geoms[geomId], lx, ly, lz);
	}
}

void odeGeomBoxGetLengths(int geomId, float &lx, float &ly, float &lz)
{
	OdeThreadContext & c = contexts[s_getIdx];
	dVector3 result;
	dGeomBoxGetLengths(c.geoms[geomId], result);
	lx = result[0];
	ly = result[1];
	lz = result[2];
}

float dGeomBoxPointDepth(int geomId, float x, float y, float z)
{
	OdeThreadContext & c = contexts[s_getIdx];
	return dGeomBoxPointDepth(c.geoms[geomId], x, y, z);
}

void odeGeomSetCategoryBits(int geomId, unsigned long bits)
{
	ITERATE_THREADS(threadIdx)
	{
		OdeThreadContext & c = contexts[threadIdx];
		dGeomSetCategoryBits(c.geoms[geomId], bits);
	}
}

void odeGeomSetCollideBits(int geomId, unsigned long bits)
{
	ITERATE_THREADS(threadIdx)
	{
		OdeThreadContext & c = contexts[threadIdx];
		dGeomSetCollideBits(c.geoms[geomId], bits);
	}
}

unsigned long odeGeomGetCategoryBits(int geomId)
{
	OdeThreadContext & c = contexts[s_getIdx];
	return dGeomGetCategoryBits(c.geoms[geomId]);
}

unsigned long odeGeomGetCollideBits(int geomId)
{
	OdeThreadContext & c = contexts[s_getIdx];
	return dGeomGetCollideBits(c.geoms[geomId]);
}

// Geom: Sphere class
int odeCreateSphere(float radius)
{
	int id = counter++;
	ITERATE_THREADS(threadIdx)
	{
		dGeomID geom = dCreateSphere(contexts[threadIdx].space, radius);
		contexts[threadIdx].geoms[id] = geom;
	}
	return id;
}

void odeGeomSphereSetRadius(int geomId, float radius)
{
	ITERATE_THREADS(threadIdx)
	{
		OdeThreadContext & c = contexts[threadIdx];
		dGeomSphereSetRadius(c.geoms[geomId], radius);
	}
}

float odeGeomSphereGetRadius(int geomId)
{
	OdeThreadContext & c = contexts[s_getIdx];
	return dGeomSphereGetRadius(c.geoms[geomId]);
}

float dGeomSpherePointDepth(int geomId, float x, float y, float z)
{
	OdeThreadContext & c = contexts[s_getIdx];
	return dGeomSpherePointDepth(c.geoms[geomId], x, y, z);
}

// Geom: Plane class

// TODO: spaceId currently not used
int EXPORT_API odeCreatePlane(int spaceId, float a, float b, float c, float d)
{
	int id = counter++;
	ITERATE_THREADS(i)
	{
		dGeomID geom = dCreatePlane(contexts[i].space, a, b, c, d);
		contexts[i].geoms[id] = geom;
	}
	return id;
}
void odeGeomPlaneSetParams(int geomId, float fa, float fb, float fc, float fd)
{
	ITERATE_THREADS(threadIdx)
	{
		OdeThreadContext & c = contexts[threadIdx];
		dGeomID plane = c.geoms[geomId];
		dGeomPlaneSetParams(plane, fa, fb, fc, fd);
	}
}
/*OdeVec4 odeGeomPlaneGetParams(int geomId)
{
OdeThreadContext & c = contexts[s_getIdx];
dVector4 dResult;
dGeomPlaneGetParams(c.geoms[geomId], dResult);
return dResult;
}*/
float odeGeomPlanePointDepth(int geomId, float x, float y, float z)
{
	OdeThreadContext & c = contexts[s_getIdx];
	dReal d = dGeomPlanePointDepth(c.geoms[geomId], x, y, z);
	return static_cast<float>(d);
}

// Geom: Heightfield class

int EXPORT_API odeCreateHeightfield(const float *heightData, float width, float depth, int widthSamples, int depthSamples, float scale, float offset, float thickness, int wrap)
{
	int id = counter++;
	ITERATE_THREADS(i)
	{
		dHeightfieldDataID data = dGeomHeightfieldDataCreate();
		dGeomHeightfieldDataBuildSingle(data, heightData, 1, width, depth, widthSamples, depthSamples, scale, offset, thickness, wrap);
		dGeomID geom = dCreateHeightfield(contexts[i].space, data, 0);
		contexts[i].geoms[id] = geom;
	}
	return id;
}

// Geom: Capsule class

// TODO: spaceId currently not used
int EXPORT_API odeCreateCapsule(int spaceId, float radius, float length)
{
	int id = counter++;
	ITERATE_THREADS(i)
	{
		// TODO: why this?
		float correctedLength = _max(0, length - radius*2.0f);
		dGeomID geom = dCreateCapsule(contexts[i].space, radius, correctedLength);
		contexts[i].geoms[id] = geom;
	}
	return id;
}
void odeGeomCapsuleSetParams(int geomId, float radius, float length)
{
	ITERATE_THREADS(threadIdx)
	{
		OdeThreadContext & c = contexts[threadIdx];
		dGeomID capsule = c.geoms[geomId];
		dGeomCapsuleSetParams(capsule, radius, length);
	}
}
void odeGeomCapsuleGetParams(int geomId, float &radius, float &length)
{
	OdeThreadContext & c = contexts[s_getIdx];
	dReal r, l;
	dGeomCapsuleGetParams(c.geoms[geomId], &r, &l);
	radius = static_cast<float>(r);
	length = static_cast<float>(l);
}
float odeGeomCapsulePointDepth(int geomId, float x, float y, float z)
{
	OdeThreadContext & c = contexts[s_getIdx];
	dReal d = dGeomCapsulePointDepth(c.geoms[geomId], x, y, z);
	return static_cast<float>(d);
}

///////////////////////////////////////////////////////////////////////////////
// Joints
///////////////////////////////////////////////////////////////////////////////

// Joint: Creating and Destroying Joints

int odeJointCreateBall()
{
	int id = counter++;
	ITERATE_THREADS(i)
	{
		dJointGroupID jointGroup = contexts[i].jointGroups[0];
		dJointID joint = dJointCreateBall(contexts[i].world, jointGroup);
		contexts[i].joints[id] = joint;
		dJointSetFeedback(joint, &contexts[i].jointFeedbacks[id]);
	}
	return id;
}
int odeJointCreateHinge()
{
	int id = counter++;
	ITERATE_THREADS(i)
	{
		dJointGroupID jointGroup = contexts[i].jointGroups[0];
		dJointID joint = dJointCreateHinge(contexts[i].world, jointGroup);
		contexts[i].joints[id] = joint;

		dJointSetFeedback(joint, &contexts[i].jointFeedbacks[id]);
	}
	return id;
}
int odeJointCreateSlider()
{
	int id = counter++;
	ITERATE_THREADS(i)
	{
		dJointGroupID jointGroup = contexts[i].jointGroups[0];
		dJointID joint = dJointCreateSlider(contexts[i].world, jointGroup);
		contexts[i].joints[id] = joint;
	}
	return id;
}
int odeJointCreateUniversal()
{
	int id = counter++;
	ITERATE_THREADS(i)
	{
		dJointGroupID jointGroup = contexts[i].jointGroups[0];
		dJointID joint = dJointCreateUniversal(contexts[i].world, jointGroup);
		contexts[i].joints[id] = joint;
	}
	return id;
}
int odeJointCreateHinge2()
{
	int id = counter++;
	ITERATE_THREADS(i)
	{
		dJointGroupID jointGroup = contexts[i].jointGroups[0];
		dJointID joint = dJointCreateHinge2(contexts[i].world, jointGroup);
		contexts[i].joints[id] = joint;
	}
	return id;
}
int odeJointCreatePR()
{
	int id = counter++;
	ITERATE_THREADS(i)
	{
		dJointGroupID jointGroup = contexts[i].jointGroups[0];
		dJointID joint = dJointCreatePR(contexts[i].world, jointGroup);
		contexts[i].joints[id] = joint;
	}
	return id;
}
int odeJointCreatePU()
{
	int id = counter++;
	ITERATE_THREADS(i)
	{
		dJointGroupID jointGroup = contexts[i].jointGroups[0];
		dJointID joint = dJointCreatePU(contexts[i].world, jointGroup);
		contexts[i].joints[id] = joint;
	}
	return id;
}
int odeJointCreatePiston()
{
	int id = counter++;
	ITERATE_THREADS(i)
	{
		dJointGroupID jointGroup = contexts[i].jointGroups[0];
		dJointID joint = dJointCreatePiston(contexts[i].world, jointGroup);
		contexts[i].joints[id] = joint;
	}
	return id;
}
int odeJointCreateFixed()
{
	int id = counter++;
	ITERATE_THREADS(i)
	{
		dJointGroupID jointGroup = contexts[i].jointGroups[0];
		dJointID joint = dJointCreateFixed(contexts[i].world, jointGroup);
		contexts[i].joints[id] = joint;
		dJointSetFeedback(joint, &contexts[i].jointFeedbacks[id]);
	}
	return id;
}
int odeJointCreateAMotor()
{
	int id = counter++;
	ITERATE_THREADS(i)
	{
		dJointGroupID jointGroup = contexts[i].jointGroups[0];
		dJointID joint = dJointCreateAMotor(contexts[i].world, jointGroup);
		contexts[i].joints[id] = joint;
		contexts[i].amotors[id] = joint; //saving and restoring needs some amotor specific data and code
		dJointSetFeedback(joint, &contexts[i].jointFeedbacks[id]);
	}
	return id;
}
int odeJointCreateLMotor()
{
	int id = counter++;
	ITERATE_THREADS(i)
	{
		dJointGroupID jointGroup = contexts[i].jointGroups[0];
		dJointID joint = dJointCreateLMotor(contexts[i].world, jointGroup);
		contexts[i].joints[id] = joint;
	}
	return id;
}
int odeJointCreatePlane2D()
{
	int id = counter++;
	ITERATE_THREADS(i)
	{
		dJointGroupID jointGroup = contexts[i].jointGroups[0];
		dJointID joint = dJointCreatePlane2D(contexts[i].world, jointGroup);
		contexts[i].joints[id] = joint;
	}
	return id;
}

// Joint: Miscellaneous Joint Functions

void odeJointAttach(int jointId, int bodyId1, int bodyId2)
{
	ITERATE_THREADS(i)
	{
		dJointID joint = contexts[i].joints[jointId];
		dBodyID body1 = 0, body2 = 0;
		if (bodyId1 != 0)
			body1 = contexts[i].bodies[bodyId1];
		if (bodyId2 != 0)
			body2 = contexts[i].bodies[bodyId2];
		dJointAttach(joint, body1, body2);
	}
}

int odeJointGetBody(int jointId, int index)
{
	OdeThreadContext & c = contexts[s_getIdx];
	dBodyID body = dJointGetBody(c.joints[jointId], index);
	// TODO: slow linear search, should find a more efficient way
	BodyContainer::const_iterator it;
	for (it = c.bodies.begin(); it != c.bodies.end(); ++it)
	{
		if (it->second == body)
			return it->first;
	}
	return -1;
}

int odeJointGetType(int jointId)
{
	OdeThreadContext & c = contexts[s_getIdx];
	return dJointGetType(c.joints[jointId]);
}

// Joint: Joint parameter setting functions

// Joint: Ball and Socket parameters

void odeJointSetBallAnchor(int jointId, float x, float y, float z)
{
	ITERATE_THREADS(threadIdx)
	{
		OdeThreadContext & c = contexts[threadIdx];
		dJointID joint = c.joints[jointId];
		dJointSetBallAnchor(joint, x, y, z);
	}
}

void odeJointGetBallAnchor(int jointId, float &x, float &y, float &z)
{
	OdeThreadContext & c = contexts[s_getIdx];
	dVector3 dResult;
	dJointGetBallAnchor(c.joints[jointId], dResult);
	x = dResult[0];
	y = dResult[1];
	z = dResult[2];
}

void odeJointGetBallAnchor2(int jointId, float &x, float &y, float &z)
{
	OdeThreadContext & c = contexts[s_getIdx];
	dVector3 dResult;
	dJointGetBallAnchor2(c.joints[jointId], dResult);
	x = dResult[0];
	y = dResult[1];
	z = dResult[2];
}

// Joint: Hinge parameters

void odeJointSetHingeAnchor(int jointId, float x, float y, float z)
{
	ITERATE_THREADS(i)
	{
		dJointID joint = contexts[i].joints[jointId];
		dJointSetHingeAnchor(joint, x, y, z);
	}
}

void odeJointSetFixed(int jointId)
{
	ITERATE_THREADS(i)
	{
		dJointID joint = contexts[i].joints[jointId];
		dJointSetFixed(joint);
	}
}

void odeJointSetHingeAxis(int jointId, float x, float y, float z)
{
	ITERATE_THREADS(i)
	{
		dJointID joint = contexts[i].joints[jointId];
		dJointSetHingeAxis(joint, x, y, z);
	}
}

void odeJointSetHinge2Anchor(int jointId, float x, float y, float z)
{
	ITERATE_THREADS(i)
	{
		dJointID joint = contexts[i].joints[jointId];
		dJointSetHinge2Anchor(joint, x, y, z);
	}
}

void odeJointSetHinge2Axis(int jointId, int axis, float x, float y, float z)
{
	ITERATE_THREADS(i)
	{
		dJointID joint = contexts[i].joints[jointId];
		if (axis == 1)
			dJointSetHinge2Axis1(joint, x, y, z);
		else
			dJointSetHinge2Axis2(joint, x, y, z);
	}
}

void odeJointGetHingeAnchor(int jointId, float &x, float &y, float &z)
{
	OdeThreadContext & c = contexts[s_getIdx];
	dVector3 dResult;
	dJointGetHingeAnchor(c.joints[jointId], dResult);
	x = dResult[0];
	y = dResult[1];
	z = dResult[2];
}
void odeJointGetHingeAnchor2(int jointId, float &x, float &y, float &z)
{
	OdeThreadContext & c = contexts[s_getIdx];
	dVector3 dResult;
	dJointGetHingeAnchor2(c.joints[jointId], dResult);
	x = dResult[0];
	y = dResult[1];
	z = dResult[2];
}
void odeJointGetHingeAxis(int jointId, float &x, float &y, float &z)
{
	OdeThreadContext & c = contexts[s_getIdx];
	dVector3 dResult;
	dJointGetHingeAxis(c.joints[jointId], dResult);
	x = dResult[0];
	y = dResult[1];
	z = dResult[2];
}
float odeJointGetHingeAngle(int jointId)
{
	OdeThreadContext & c = contexts[s_getIdx];
	dReal a = dJointGetHingeAngle(c.joints[jointId]);
	return static_cast<float>(a);
}

float EXPORT_API odeJointGetHingeAngleFromBodyRotations(int jointId, ConstOdeQuaternion q1, ConstOdeQuaternion q2)
{
	OdeThreadContext & c = contexts[s_getIdx];
	dReal a = dJointGetHingeAngleFromBodyRotations(c.joints[jointId], q1, q2);
	return static_cast<float>(a);

}
float odeJointGetHingeAngleRate(int jointId)
{
	OdeThreadContext & c = contexts[s_getIdx];
	dReal a = dJointGetHingeAngleRate(c.joints[jointId]);
	return static_cast<float>(a);
}

// Joint: AMotor parameters

void odeJointSetAMotorMode(int jointId, int mode)
{
	ITERATE_THREADS(i)
	{
		dJointID joint = contexts[i].joints[jointId];
		dJointSetAMotorMode(joint, mode);
	}
}
int odeJointGetAMotorMode(int jointId)
{
	OdeThreadContext & c = contexts[s_getIdx];
	int dMode = dJointGetAMotorMode(c.joints[jointId]);
	return dMode;
}
void odeJointSetAMotorNumAxes(int jointId, int num)
{
	ITERATE_THREADS(i)
	{
		dJointID joint = contexts[i].joints[jointId];
		dJointSetAMotorNumAxes(joint, num);
	}
}
int odeJointGetAMotorNumAxes(int jointId)
{
	OdeThreadContext & c = contexts[s_getIdx];
	int dNum = dJointGetAMotorNumAxes(c.joints[jointId]);
	return dNum;
}
void odeJointSetAMotorAxis(int jointId, int anum, int rel, float x, float y, float z)
{
	ITERATE_THREADS(i)
	{
		dJointID joint = contexts[i].joints[jointId];
		dJointSetAMotorAxis(joint, anum, rel, x, y, z);
	}
}
void odeJointGetAMotorAxis(int jointId, int anum, float &x, float &y, float &z)
{
	OdeThreadContext & c = contexts[s_getIdx];
	dVector3 dResult;
	dJointGetAMotorAxis(c.joints[jointId], anum, dResult);
	x = dResult[0];
	y = dResult[1];
	z = dResult[2];
}
int odeJointGetAMotorAxisRel(int jointId, int anum)
{
	OdeThreadContext & c = contexts[s_getIdx];
	int dRel = dJointGetAMotorAxisRel(c.joints[jointId], anum);
	return dRel;
}
void odeJointSetAMotorAngle(int jointId, int anum, float angle)
{
	ITERATE_THREADS(i)
	{
		dJointID joint = contexts[i].joints[jointId];
		dJointSetAMotorAngle(joint, anum, angle);
	}
}
float odeJointGetAMotorAngle(int jointId, int anum)
{
	OdeThreadContext & c = contexts[s_getIdx];
	dReal dAngle = dJointGetAMotorAngle(c.joints[jointId], anum);
	return static_cast<float>(dAngle);
}
float odeJointGetAMotorAngleRate(int jointId, int anum)
{
	OdeThreadContext & c = contexts[s_getIdx];
	dReal dAngleRate = dJointGetAMotorAngleRate(c.joints[jointId], anum);
	return static_cast<float>(dAngleRate);
}
void odeJointGetAMotorAnglesFromBodyRotations(int jointId, ConstOdeQuaternion q1, ConstOdeQuaternion q2, OdeVector result)
{
	OdeThreadContext & c = contexts[s_getIdx];
	dJointID joint = c.joints[jointId];

	dJointGetAMotorAnglesFromBodyRotations(joint, q1, q2, result);
}

// Joint: Parameter functions

void odeJointSetBallParam(int jointId, int parameter, float value)
{
	ITERATE_THREADS(threadIdx)
	{
		OdeThreadContext & c = contexts[threadIdx];
		dJointID joint = c.joints[jointId];
		dJointSetBallParam(joint, parameter, value);
	}
}
void odeJointSetHingeParam(int jointId, int parameter, float value)
{
	ITERATE_THREADS(threadIdx)
	{
		OdeThreadContext & c = contexts[threadIdx];
		dJointID joint = c.joints[jointId];
		dJointSetHingeParam(joint, parameter, value);
	}
}
void odeJointSetSliderParam(int jointId, int parameter, float value)
{
	ITERATE_THREADS(threadIdx)
	{
		OdeThreadContext & c = contexts[threadIdx];
		dJointID joint = c.joints[jointId];
		dJointSetSliderParam(joint, parameter, value);
	}
}
void odeJointSetHinge2Param(int jointId, int parameter, float value)
{
	ITERATE_THREADS(threadIdx)
	{
		OdeThreadContext & c = contexts[threadIdx];
		dJointID joint = c.joints[jointId];
		dJointSetHinge2Param(joint, parameter, value);
	}
}
void odeJointSetUniversalParam(int jointId, int parameter, float value)
{
	ITERATE_THREADS(threadIdx)
	{
		OdeThreadContext & c = contexts[threadIdx];
		dJointID joint = c.joints[jointId];
		dJointSetUniversalParam(joint, parameter, value);
	}
}
void odeJointSetAMotorParam(int jointId, int parameter, float value)
{
	ITERATE_THREADS(threadIdx)
	{
		OdeThreadContext & c = contexts[threadIdx];
		dJointID joint = c.joints[jointId];
		dJointSetAMotorParam(joint, parameter, value);
	}
}
void odeJointSetLMotorParam(int jointId, int parameter, float value)
{
	ITERATE_THREADS(threadIdx)
	{
		OdeThreadContext & c = contexts[threadIdx];
		dJointID joint = c.joints[jointId];
		dJointSetLMotorParam(joint, parameter, value);
	}
}
void odeJointSetPRParam(int jointId, int parameter, float value)
{
	ITERATE_THREADS(threadIdx)
	{
		OdeThreadContext & c = contexts[threadIdx];
		dJointID joint = c.joints[jointId];
		dJointSetPRParam(joint, parameter, value);
	}
}
void odeJointSetPUParam(int jointId, int parameter, float value)
{
	ITERATE_THREADS(threadIdx)
	{
		OdeThreadContext & c = contexts[threadIdx];
		dJointID joint = c.joints[jointId];
		dJointSetPUParam(joint, parameter, value);
	}
}
void odeJointSetPistonParam(int jointId, int parameter, float value)
{
	ITERATE_THREADS(threadIdx)
	{
		OdeThreadContext & c = contexts[threadIdx];
		dJointID joint = c.joints[jointId];
		dJointSetPistonParam(joint, parameter, value);
	}
}
float odeJointGetBallParam(int jointId, int parameter)
{
	OdeThreadContext & c = contexts[s_getIdx];
	dReal dValue = dJointGetBallParam(c.joints[jointId], parameter);
	return static_cast<float>(dValue);
}
float odeJointGetHingeParam(int jointId, int parameter)
{
	OdeThreadContext & c = contexts[s_getIdx];
	dReal dValue = dJointGetHingeParam(c.joints[jointId], parameter);
	return static_cast<float>(dValue);
}
float odeJointGetSliderParam(int jointId, int parameter)
{
	OdeThreadContext & c = contexts[s_getIdx];
	dReal dValue = dJointGetSliderParam(c.joints[jointId], parameter);
	return static_cast<float>(dValue);
}
float odeJointGetHinge2Param(int jointId, int parameter)
{
	OdeThreadContext & c = contexts[s_getIdx];
	dReal dValue = dJointGetHinge2Param(c.joints[jointId], parameter);
	return static_cast<float>(dValue);
}
float odeJointGetUniversalParam(int jointId, int parameter)
{
	OdeThreadContext & c = contexts[s_getIdx];
	dReal dValue = dJointGetUniversalParam(c.joints[jointId], parameter);
	return static_cast<float>(dValue);
}
float odeJointGetAMotorParam(int jointId, int parameter)
{
	OdeThreadContext & c = contexts[s_getIdx];
	dReal dValue = dJointGetAMotorParam(c.joints[jointId], parameter);
	return static_cast<float>(dValue);
}
float odeJointGetLMotorParam(int jointId, int parameter)
{
	OdeThreadContext & c = contexts[s_getIdx];
	dReal dValue = dJointGetLMotorParam(c.joints[jointId], parameter);
	return static_cast<float>(dValue);
}
float odeJointGetPRParam(int jointId, int parameter)
{
	OdeThreadContext & c = contexts[s_getIdx];
	dReal dValue = dJointGetPRParam(c.joints[jointId], parameter);
	return static_cast<float>(dValue);
}
float odeJointGetPUParam(int jointId, int parameter)
{
	OdeThreadContext & c = contexts[s_getIdx];
	dReal dValue = dJointGetPUParam(c.joints[jointId], parameter);
	return static_cast<float>(dValue);
}
float odeJointGetPistonParam(int jointId, int parameter)
{
	OdeThreadContext & c = contexts[s_getIdx];
	dReal dValue = dJointGetPistonParam(c.joints[jointId], parameter);
	return static_cast<float>(dValue);
}

// Joint: Setting Joint Torques/Forces Directly

void odeJointAddHingeTorque(int jointId, float torque)
{
	ITERATE_THREADS(threadIdx)
	{
		OdeThreadContext & c = contexts[threadIdx];
		dJointID joint = c.joints[jointId];
		dJointAddHingeTorque(joint, torque);
	}
}

void odeJointAddUniversalTorques(int jointId, float torque1, float torque2)
{
	ITERATE_THREADS(threadIdx)
	{
		OdeThreadContext & c = contexts[threadIdx];
		dJointID joint = c.joints[jointId];
		dJointAddUniversalTorques(joint, torque1, torque2);
	}
}

void odeJointAddSliderForce(int jointId, float force)
{
	ITERATE_THREADS(threadIdx)
	{
		OdeThreadContext & c = contexts[threadIdx];
		dJointID joint = c.joints[jointId];
		dJointAddSliderForce(joint, force);
	}
}

void odeJointAddHinge2Torques(int jointId, float torque1, float torque2)
{
	ITERATE_THREADS(threadIdx)
	{
		OdeThreadContext & c = contexts[threadIdx];
		dJointID joint = c.joints[jointId];
		dJointAddHinge2Torques(joint, torque1, torque2);
	}
}

void odeJointAddAMotorTorques(int jointId, float torque1, float torque2, float torque3)
{
	ITERATE_THREADS(threadIdx)
	{
		OdeThreadContext & c = contexts[threadIdx];
		dJointID joint = c.joints[jointId];
		dJointAddAMotorTorques(joint, torque1, torque2, torque3);
	}
}

bool EXPORT_API stepOde(float stepSize, bool breakOnErrors)
{
	bool isError = false;

	ITERATE_THREADS(threadIdx)
	{
		contexts[threadIdx].contacts.clear();

		if (!breakOnErrors) {
			contexts[threadIdx].errorOccurred = false;
			dSetDebugHandler(handleOdeError);
		}

		//generate contact joints
		dSpaceCollide(contexts[threadIdx].space, &contexts[threadIdx], &nearCallback);
		//step

		if (breakOnErrors || !contexts[threadIdx].errorOccurred)
		{
			dWorldStep(contexts[threadIdx].world, stepSize);
			//dWorldQuickStep(contexts[threadIdx].world,stepSize);
		}

		if (!breakOnErrors) {
			dSetDebugHandler(0);
			isError |= contexts[threadIdx].errorOccurred;
		}

		// remove all contact joints
		dJointGroupEmpty(contexts[threadIdx].contactGroup);
	}

	return !isError;
}

bool EXPORT_API stepOdeFast(float stepSize, bool breakOnErrors)
{
	bool isError = false;

	ITERATE_THREADS(threadIdx)
	{
		contexts[threadIdx].contacts.clear();

		if (!breakOnErrors) {
			contexts[threadIdx].errorOccurred = false;
			dSetDebugHandler(handleOdeError);
		}

		//generate contact joints
		dSpaceCollide(contexts[threadIdx].space, &contexts[threadIdx], &nearCallback);
		//step

		if (breakOnErrors || !contexts[threadIdx].errorOccurred)
		{
			//dWorldStep(contexts[threadIdx].world,stepSize);
			dWorldQuickStep(contexts[threadIdx].world, stepSize);
		}

		if (!breakOnErrors) {
			dSetDebugHandler(0);
			isError |= contexts[threadIdx].errorOccurred;
		}

		// remove all contact joints
		dJointGroupEmpty(contexts[threadIdx].contactGroup);
	}

	return !isError;

}

static int getExternalGeomId(dGeomID bodyId)
{
	OdeThreadContext & c = contexts[s_getIdx];

	GeomContainer::const_iterator it;
	for (it = c.geoms.begin(); it != c.geoms.end(); ++it)
	{
		dGeomID geomIdCompare = it->second;
		if (geomIdCompare == bodyId)
			return it->first;
	}
	return 0;
}

static int getExternalBodyId(dBodyID bodyId)
{
	if (!bodyId)
		return 0;

	OdeThreadContext & c = contexts[s_getIdx];

	BodyContainer::const_iterator it;
	for (it = c.bodies.begin(); it != c.bodies.end(); ++it)
	{
		dBodyID bodyIdCompare = it->second;
		if (bodyIdCompare == bodyId)
			return it->first;
	}
	return 0;
}

int odeGetContactCount()
{
	OdeThreadContext &context = contexts[s_getIdx];
	return context.contacts.size();
}

void odeGetContactInfo(int index, int &body1Id, int &body2Id, OdeVector pos, OdeVector normal, OdeVector contactVel)
{
	OdeThreadContext &context = contexts[s_getIdx];
	assert(context.contacts.find(index) != context.contacts.end());
	dContact &contact = context.contacts[index];

	body1Id = contact.geom.g1 ? getExternalBodyId(dGeomGetBody(contact.geom.g1)) : 0;
	body2Id = contact.geom.g2 ? getExternalBodyId(dGeomGetBody(contact.geom.g2)) : 0;
	dCopyVector3(pos, contact.geom.pos);
	dCopyVector3(normal, contact.geom.normal);
	dCopyVector3(contactVel, contact.vel);
}

float odeGetMaxContactSpeed(int bodyId)
{
	OdeThreadContext &context = contexts[s_getIdx];
	ContactContainer::const_iterator ci;
	dBodyID b = context.bodies[bodyId];
	float result = -1;
	for (ci = context.contacts.begin(); ci != context.contacts.end(); ++ci)
	{
		dBodyID b1 = dGeomGetBody(ci->second.geom.g1);
		dBodyID b2 = dGeomGetBody(ci->second.geom.g1);
		if (b == b1 || b == b2)
		{
			float sqSpeed = ci->second.vel[0] * ci->second.vel[0] + ci->second.vel[1] * ci->second.vel[1] + ci->second.vel[2] * ci->second.vel[2];
			result = _max(result, sqSpeed);
		}
	}
	if (result>0)
		result = sqrtf(result);
	return result;
}

bool AreBodiesEqual(dBodyID b1, dBodyID b2)
{
	if ((b1 != NULL && b2 == NULL) || (b1 == NULL && b2 != NULL))
		return false;
	if (b1 == NULL && b2 == NULL)
	{
		printf("URGENT! POSSIBLE BUG DETECTED AT UNITY ODE CODE.\n");
		return true;
	}
	if (abs(b1->mass.mass - b2->mass.mass) > 0.01)
		return false;
	if (abs(b1->posr.pos[0] - b2->posr.pos[0]) > 0.01)
		return false;
	if (abs(b1->posr.pos[1] - b2->posr.pos[1]) > 0.01)
		return false;
	if (abs(b1->posr.pos[2] - b2->posr.pos[2]) > 0.01)
		return false;
	return true;
}

bool EXPORT_API odeGetContact(int body1Id, int body2Id, OdeVector out_pos, OdeVector out_normal, OdeVector out_vel)
{
	OdeThreadContext &context = contexts[s_getIdx];
	ContactContainer::const_iterator ci;
	dBodyID ba = context.bodies[body1Id];
	dBodyID bb = context.bodies[body2Id];
	for (ci = context.contacts.begin(); ci != context.contacts.end(); ++ci)
	{
		dBodyID b1 = dGeomGetBody(ci->second.geom.g1);
		dBodyID b2 = dGeomGetBody(ci->second.geom.g2);
		//if ((ba==b1 && bb==b2) || (ba==b2 || bb==b1))
		if ((AreBodiesEqual(ba, b1) && AreBodiesEqual(bb, b2)) || (AreBodiesEqual(ba, b2) && AreBodiesEqual(bb, b1)))
		{
			const dContact &contact = ci->second;
			dCopyVector3(out_pos, contact.geom.pos);
			dCopyVector3(out_normal, contact.geom.normal);
			dCopyVector3(out_vel, contact.vel);
			return true;
		}
	}
	return false;
}

bool EXPORT_API odeGetGeomContact(int geom1Id, int geom2Id, OdeVector out_pos, OdeVector out_normal, OdeVector out_vel)
{
	OdeThreadContext &context = contexts[s_getIdx];
	ContactContainer::const_iterator ci;
	dGeomID ga = context.geoms[geom1Id];
	dGeomID gb = context.geoms[geom2Id];
	for (ci = context.contacts.begin(); ci != context.contacts.end(); ++ci)
	{
		dGeomID g1 = ci->second.geom.g1;
		dGeomID g2 = ci->second.geom.g2;
		if ((ga == g1 && gb == g2) || (ga == g2 && gb == g1))
		{
			const dContact &contact = ci->second;
			dCopyVector3(out_pos, contact.geom.pos);
			dCopyVector3(out_normal, contact.geom.normal);
			dCopyVector3(out_vel, contact.vel);
			return true;
		}
	}
	return false;
}



/// Saves current ODE states from master context
// TODO: save joint groups
bool EXPORT_API saveOdeState(int slot, int sourceContext)
{
	SaveState & save = ::saves[slot];

	save.seed = dRandGetSeed();
	save.bodies.resize(contexts[sourceContext].bodies.size());
	BodyContainer::const_iterator i;
	int saveIdx = 0;
	//save bodies
	for (i = contexts[sourceContext].bodies.begin(); i != contexts[sourceContext].bodies.end(); ++i) {
		dBodyID odeBody = i->second;
		if (!odeBody) continue;

		memcpy(save.bodies[saveIdx].location, dBodyGetPosition(odeBody), sizeof(dReal) * 3);
		memcpy(save.bodies[saveIdx].velocity, dBodyGetLinearVel(odeBody), sizeof(dReal) * 3);
		memcpy(save.bodies[saveIdx].angularVelocity, dBodyGetAngularVel(odeBody), sizeof(dReal) * 3);
		memcpy(save.bodies[saveIdx].quaternion, dBodyGetQuaternion(odeBody), sizeof(dReal) * 4);
		saveIdx++;
	}

	//save motor angles: Currently, retrieving aMotor angles only returns correct values after a simulation step has been run. 
	//This can throw off simulations that set joint parameters based on these angles, since they will get different values after being restored to a given state
	//The ode wiki suggests that the angle computing code from amotorGetInfo1() in joint.cpp can be inserted into dJointGetAMotorAngle() to update the value every time.
	//However, there's no amotorGetInfo1() nor joint.cpp anymore -> we simply save and restore the angles here.
	save.amotors.resize(contexts[sourceContext].amotors.size());
	JointContainer::const_iterator ji;
	saveIdx = 0;
	for (ji = contexts[sourceContext].amotors.begin(); ji != contexts[sourceContext].amotors.end(); ++ji) {
		dJointID odeJoint = ji->second;
		if (!odeJoint) continue;
		dxJointAMotor* joint = (dxJointAMotor*)odeJoint;
		memcpy(save.amotors[saveIdx].axis, joint->axis, sizeof(dVector3) * 3);
		memcpy(save.amotors[saveIdx].limot, joint->limot, sizeof(dxJointLimitMotor) * 3);
		memcpy(save.amotors[saveIdx].angle, joint->angle, sizeof(dReal) * 3);
		saveIdx++;
	}

	save.joints.resize(contexts[sourceContext].joints.size());
	saveIdx = 0;
	for (ji = contexts[sourceContext].joints.begin(); ji != contexts[sourceContext].joints.end(); ++ji) {
		dJointID odeJoint = ji->second;
		if (!odeJoint) continue;
		dJointType type = dJointGetType(odeJoint);
		switch (type)
		{
		case dJointTypeHinge:
			save.joints[saveIdx].fmax[0] = dJointGetHingeParam(odeJoint, dParamFMax);
			break;
		case dJointTypeAMotor:
			save.joints[saveIdx].fmax[0] = dJointGetAMotorParam(odeJoint, dParamFMax);
			save.joints[saveIdx].fmax[1] = dJointGetAMotorParam(odeJoint, dParamFMax2);
			save.joints[saveIdx].fmax[2] = dJointGetAMotorParam(odeJoint, dParamFMax3);
			break;
		}
		saveIdx++;
	}

	save.contacts.resize(contexts[sourceContext].contacts.size());
	ContactContainer::const_iterator ci;
	saveIdx = 0;
	for (ci = contexts[sourceContext].contacts.begin(); ci != contexts[sourceContext].contacts.end(); ++ci)
	{
		save.contacts[saveIdx].geomId1 = getExternalGeomId(ci->second.geom.g1);
		save.contacts[saveIdx].geomId2 = getExternalGeomId(ci->second.geom.g2);
		memcpy(save.contacts[saveIdx].position, ci->second.geom.pos, sizeof(dVector3));
		memcpy(save.contacts[saveIdx].normal, ci->second.geom.normal, sizeof(dVector3));
		memcpy(save.contacts[saveIdx].vel, ci->second.vel, sizeof(dVector3));
		saveIdx++;
	}

	save.contactSoftCFM = contexts[sourceContext].contactSoftCFM;

	return true;
}

/// Sets ODE states for calling context from saved master context
bool EXPORT_API restoreOdeState(int slot, bool breakOnErrors)
{
	if (::saves.find(slot) == ::saves.end()) {
		return false;
	}

	bool isError = false;

	ITERATE_THREADS(threadIdx)
	{
		if (!breakOnErrors) {
			contexts[threadIdx].errorOccurred = false;
			dSetDebugHandler(handleOdeError);
		}

		OdeThreadContext & c = contexts[threadIdx];

		SaveState & save = ::saves[slot];

		dRandSetSeed(save.seed);
		BodyContainer::const_iterator i;
		int saveIdx = 0;

		if (save.bodies.size() < c.bodies.size())
		{
			return false;
		}

		for (i = c.bodies.begin(); i != c.bodies.end(); ++i) {
			dBodyID odeBody = i->second;
			if (!odeBody) continue;

			dBodySetPosition(odeBody, save.bodies[saveIdx].location[0], save.bodies[saveIdx].location[1], save.bodies[saveIdx].location[2]);
			dBodySetQuaternionWithoutNormalization(odeBody, save.bodies[saveIdx].quaternion);
			dBodySetLinearVel(odeBody, save.bodies[saveIdx].velocity[0], save.bodies[saveIdx].velocity[1], save.bodies[saveIdx].velocity[2]);
			dBodySetAngularVel(odeBody, save.bodies[saveIdx].angularVelocity[0], save.bodies[saveIdx].angularVelocity[1], save.bodies[saveIdx].angularVelocity[2]);
			dBodyEnable(odeBody);
			saveIdx++;
		}

		if (save.amotors.size() < c.amotors.size())
		{
			return false;
		}

		JointContainer::const_iterator ji;
		saveIdx = 0;
		for (ji = c.amotors.begin(); ji != c.amotors.end(); ++ji) {
			dJointID odeJoint = ji->second;
			if (!odeJoint) continue;
			dxJointAMotor* joint = (dxJointAMotor*)odeJoint;
			memcpy(joint->axis, save.amotors[saveIdx].axis, sizeof(dVector3) * 3);
			memcpy(joint->limot, save.amotors[saveIdx].limot, sizeof(dxJointLimitMotor) * 3);
			memcpy(joint->angle, save.amotors[saveIdx].angle, sizeof(dReal) * 3);

			saveIdx++;
		}

		saveIdx = 0;
		for (ji = c.joints.begin(); ji != c.joints.end(); ++ji)
		{
			dJointID odeJoint = ji->second;
			if (!odeJoint) continue;
			dJointType type = dJointGetType(odeJoint);
			switch (type)
			{
			case dJointTypeHinge:
				dJointSetHingeParam(odeJoint, dParamFMax, save.joints[saveIdx].fmax[0]);
				break;
			case dJointTypeAMotor:
				dJointSetAMotorParam(odeJoint, dParamFMax, save.joints[saveIdx].fmax[0]);
				dJointSetAMotorParam(odeJoint, dParamFMax2, save.joints[saveIdx].fmax[1]);
				dJointSetAMotorParam(odeJoint, dParamFMax3, save.joints[saveIdx].fmax[2]);
				break;
			}
			saveIdx++;
		}

		c.contacts.clear();
		c.contactSoftCFM = save.contactSoftCFM;
		for (saveIdx = 0; saveIdx < (int)save.contacts.size(); saveIdx++)
		{
			dContact & contact = c.contacts[saveIdx];
			int geomId1 = save.contacts[saveIdx].geomId1;
			int geomId2 = save.contacts[saveIdx].geomId2;

			if (c.geoms.find(geomId1) == c.geoms.end() || c.geoms.find(geomId2) == c.geoms.end())
			{
				continue;
			}

			contact.geom.g1 = c.geoms[geomId1];
			contact.geom.g2 = c.geoms[geomId2];
			memcpy(contact.geom.pos, &save.contacts[saveIdx].position, sizeof(dVector3));
			memcpy(contact.geom.normal, &save.contacts[saveIdx].normal, sizeof(dVector3));
			memcpy(contact.vel, &save.contacts[saveIdx].vel, sizeof(dVector3));
		}

		if (!breakOnErrors) {
			dSetDebugHandler(0);
			isError |= contexts[threadIdx].errorOccurred;
		}
	}

	return !isError;
}

template <class T>
void writePOD(std::ostream &s, T i)
{
	s.write((const char*)(&i), sizeof(i));
}

bool EXPORT_API saveOdeStateToFile(const char *filename, int slot, int extraFloatsAmount, float *extraFloats)
{
	SaveState & save = ::saves[slot];

	std::ofstream file(filename, std::ios::binary | std::ios::out);
	if (!file)
		return false;

	writePOD(file, save.bodies.size());
	for (size_t i = 0; i != save.bodies.size(); ++i)
		writePOD(file, save.bodies[i]);

	writePOD(file, save.amotors.size());
	for (size_t i = 0; i != save.amotors.size(); ++i)
		writePOD(file, save.amotors[i]);

	writePOD(file, save.joints.size());
	for (size_t i = 0; i != save.joints.size(); ++i)
		writePOD(file, save.joints[i]);

	writePOD(file, save.contacts.size());
	for (size_t i = 0; i != save.contacts.size(); ++i)
		writePOD(file, save.contacts[i]);

	writePOD(file, save.contactSoftCFM);
	writePOD(file, save.frictionCoefficient);
	writePOD(file, save.seed);

	for (size_t i = 0; i != extraFloatsAmount; ++i)
		writePOD(file, extraFloats[i]);

	return true;
}

template <class T>
T readPOD(std::istream &s)
{
	T ret;
	s.read((char*)&ret, sizeof(T));
	return ret;
}

bool EXPORT_API loadOdeStateFromFile(const char *filename, int slot, int extraFloatsAmount, float *extraFloats)
{
	if (::saves.find(slot) == ::saves.end()) {
		return false;
	}

	std::ifstream file(filename, std::ios::binary | std::ios::out);
	if (!file)
		return false;

	SaveState & save = ::saves[slot];

	save.bodies.resize(readPOD<size_t>(file));
	for (size_t i = 0; i != save.bodies.size(); ++i)
		save.bodies[i] = readPOD<BodyState>(file);

	save.amotors.resize(readPOD<size_t>(file));
	for (size_t i = 0; i != save.amotors.size(); ++i)
		save.amotors[i] = readPOD<AMotorState>(file);

	save.joints.resize(readPOD<size_t>(file));
	for (size_t i = 0; i != save.joints.size(); ++i)
		save.joints[i] = readPOD<JointState>(file);

	save.contacts.resize(readPOD<size_t>(file));
	for (size_t i = 0; i != save.contacts.size(); ++i)
		save.contacts[i] = readPOD<ContactState>(file);

	save.contactSoftCFM = readPOD<float>(file);
	save.frictionCoefficient = readPOD<float>(file);
	save.seed = readPOD<unsigned long>(file);

	for (size_t i = 0; i != extraFloatsAmount; ++i)
		extraFloats[i] = readPOD<float>(file);

	return true;
}

///////////////////////////////////////////////////////////////////////////////
// Support Functions
///////////////////////////////////////////////////////////////////////////////

// Support: Mass functions

void odeMassSetSphere(int bodyId, float density, float radius)
{
	dMass dmass;
	dMassSetSphere(&dmass, density, radius);

	ITERATE_THREADS(threadIdx)
	{
		dBodyID body = contexts[threadIdx].bodies[bodyId];
		dBodySetMass(body, &dmass);
	}
}
void odeMassSetSphereTotal(int bodyId, float total_mass, float radius)
{
	dMass dmass;
	dMassSetSphereTotal(&dmass, total_mass, radius);

	ITERATE_THREADS(threadIdx)
	{
		dBodyID body = contexts[threadIdx].bodies[bodyId];
		dBodySetMass(body, &dmass);
	}
}
void odeMassSetCapsule(int bodyId, float density, float radius, float length)
{
	dMass dmass;
	dMassSetCapsule(&dmass, density, 3, radius, length);

	ITERATE_THREADS(threadIdx)
	{
		dBodyID body = contexts[threadIdx].bodies[bodyId];
		dBodySetMass(body, &dmass);
	}
}
void odeMassSetCapsuleTotal(int bodyId, float total_mass, float radius, float length)
{
	float correctedLength = _max(0, length - radius*2.0f);  //see createcapsule
	dMass dmass;
	dMassSetCapsuleTotal(&dmass, total_mass, 3, radius, correctedLength);

	ITERATE_THREADS(threadIdx)
	{
		dBodyID body = contexts[threadIdx].bodies[bodyId];
		dBodySetMass(body, &dmass);
	}
}
void odeMassSetCylinder(int bodyId, float density, float radius, float length)
{
	dMass dmass;
	dMassSetCylinder(&dmass, density, 3, radius, length);

	ITERATE_THREADS(threadIdx)
	{
		dBodyID body = contexts[threadIdx].bodies[bodyId];
		dBodySetMass(body, &dmass);
	}
}
void odeMassSetCylinderTotal(int bodyId, float total_mass, float radius, float length)
{
	dMass dmass;
	dMassSetCylinderTotal(&dmass, total_mass, 3, radius, length);

	ITERATE_THREADS(threadIdx)
	{
		dBodyID body = contexts[threadIdx].bodies[bodyId];
		dBodySetMass(body, &dmass);
	}
}
void odeMassSetBox(int bodyId, float density, float lx, float ly, float lz)
{
	dMass dmass;
	dMassSetBox(&dmass, density, lx, ly, lz);

	ITERATE_THREADS(threadIdx)
	{
		dBodyID body = contexts[threadIdx].bodies[bodyId];
		dBodySetMass(body, &dmass);
	}
}
void odeMassSetBoxTotal(int bodyId, float total_mass, float lx, float ly, float lz)
{
	dMass dmass;
	dMassSetBoxTotal(&dmass, total_mass, lx, ly, lz);

	ITERATE_THREADS(threadIdx)
	{
		dBodyID body = contexts[threadIdx].bodies[bodyId];
		dBodySetMass(body, &dmass);
	}
}

///////////////////////////////////////////////////////////////////////////////
// World
///////////////////////////////////////////////////////////////////////////////

void EXPORT_API odeWorldSetGravity(float x, float y, float z)
{
	ITERATE_THREADS(i)
	{
		dWorldSetGravity(contexts[i].world, x, y, z);
	}
}
void EXPORT_API odeWorldSetCFM(float cfm)
{
	ITERATE_THREADS(i)
	{
		dWorldSetCFM(contexts[i].world, cfm);
	}
}

void EXPORT_API odeWorldSetERP(float erp)
{
	ITERATE_THREADS(i)
	{
		dWorldSetCFM(contexts[i].world, erp);
	}
}



void EXPORT_API allocateODEDataForThread()
{
	// This is here because some exception handlers must be set for each thread
	setupThreadExceptionHandlers();

	dAllocateODEDataForThread(dAllocateMaskAll);
}

void enableJointFeedback(bool enable)
{
	ITERATE_THREADS(threadIdx)
	{
		OdeThreadContext & c = contexts[threadIdx];
		for (auto & j : c.joints)
		{
			int id = j.first;
			dJointID joint = j.second;

			if (enable)
			{
				dJointSetFeedback(joint, &c.jointFeedbacks[id]);
			}
			else
			{
				dJointSetFeedback(joint, 0);
				c.jointFeedbacks.erase(id);
			}
		}
	}
}

void odeBodyGetAccumulatedForce(int bodyId, int jointType, OdeVector result)
{
	dSetZero(result, 3);

	OdeThreadContext & c = contexts[s_getIdx];
	dBodyID body = c.bodies[bodyId];

	int numJoints = dBodyGetNumJoints(body);
	for (int i = 0; i < numJoints; i++)
	{
		dJointID joint = dBodyGetJoint(body, i);
		dJointType type = dJointGetType(joint);
		if (jointType<0 || type == jointType)
		{
			dJointFeedback * feedback = dJointGetFeedback(joint);

			if (!feedback)
				continue;

			// Check if we should accumulate for body1 or body2
			if (dJointGetBody(joint, 0) == body)
				dAddVectors3(result, result, feedback->f1);
			else if (dJointGetBody(joint, 1) == body)
				dAddVectors3(result, result, feedback->f2);
		}
	}
}

void odeBodyGetAccumulatedTorque(int bodyId, int jointType, OdeVector result)
{
	dSetZero(result, 3);

	OdeThreadContext & c = contexts[s_getIdx];
	dBodyID body = c.bodies[bodyId];

	int numJoints = dBodyGetNumJoints(body);
	for (int i = 0; i < numJoints; i++)
	{
		dJointID joint = dBodyGetJoint(body, i);
		dJointType type = dJointGetType(joint);
		if (jointType<0 || type == jointType)
		{
			dJointFeedback * feedback = dJointGetFeedback(joint);

			if (!feedback)
				continue;

			// Check if we should accumulate for body1 or body2
			if (dJointGetBody(joint, 0) == body)
				dAddVectors3(result, result, feedback->t1);
			else if (dJointGetBody(joint, 1) == body)
				dAddVectors3(result, result, feedback->t2);
		}
	}
}

ConstOdeVector odeJointGetAccumulatedTorque(int jointId, int bodyIdx)
{
	OdeThreadContext & c = contexts[s_getIdx];
	dJointFeedback * feedback = dJointGetFeedback(c.joints[jointId]);
	return bodyIdx == 0 ? feedback->t1 : feedback->t2;
}

ConstOdeVector odeJointGetAccumulatedForce(int jointId, int bodyIdx)
{
	OdeThreadContext & c = contexts[s_getIdx];
	dJointFeedback * feedback = dJointGetFeedback(c.joints[jointId]);
	return bodyIdx == 0 ? feedback->f1 : feedback->f2;
}


void EXPORT_API setCurrentOdeContext(int threadIdx)
{
	assert(threadIdx<(int)contexts.size());
	s_setIdx = threadIdx;
	if (threadIdx<0) //<0 means we iterate over all contexts. However the getters only return a single value - in this case from thread 0
		s_getIdx = 0;
	else
		s_getIdx = threadIdx;
}


int EXPORT_API getCurrentOdeContext()
{
	return s_setIdx;
}

/*OdeVec3 vecGet()
{
OdeVec3 result;
return result;
}

void vecSet( OdeVec3 r )
{
}

void vecRef(OdeVec3 &r)
{
}

void vecPtr(OdeVec3 *r)
{
}
void vecConstRef(const OdeVec3 &r)
{}*/

void EXPORT_API odeWorldSetContactMaxCorrectingVel(float vel)
{
	ITERATE_THREADS(i)
	{
		dWorldSetContactMaxCorrectingVel(contexts[i].world, vel);
	}
}

void EXPORT_API odeWorldSetContactSurfaceLayer(float depth)
{
	ITERATE_THREADS(i)
	{
		dWorldSetContactSurfaceLayer(contexts[i].world, depth);
	}

}

int odeRaycast(float px, float py, float pz, float dx, float dy, float dz, float length, OdeVector out_pos, float & out_depth, unsigned long collideBits, unsigned long categoryBits)
{
	OdeThreadContext & c = contexts[s_getIdx];

	dGeomID ray = dCreateRay(NULL, length);
	dGeomSetCollideBits(ray, collideBits);
	dGeomSetCategoryBits(ray, categoryBits);
	dGeomRaySet(ray, px, py, pz, dx, dy, dz);

	dContactGeom contacts[NUM_CONTACTS];

	int flags = 0 | NUM_CONTACTS;
	int nContacts = dCollide(ray, (dGeomID)c.space, flags, contacts, sizeof(dContactGeom));

	if (nContacts > 0)
	{
		const dContactGeom * closest = std::min_element(contacts, contacts + nContacts,
			[](const dContactGeom & a, const dContactGeom & b)
		{
			return a.depth < b.depth;
		});
		memcpy(out_pos, closest->pos, sizeof(dVector3));
		out_depth = closest->depth;
		dBodyID hit = dGeomGetBody(closest->g2);
		for (BodyContainer::const_iterator it = c.bodies.begin(); it != c.bodies.end(); ++it)
		{
			if (it->second == hit)
			{
				memcpy(out_pos, closest->pos, sizeof(dVector3));
				out_depth = closest->depth;
				dGeomDestroy(ray);
				return it->first;
			}
		}
	}

	dGeomDestroy(ray);

	return -1;
}

bool odeRaycastGeom(float px, float py, float pz, float dx, float dy, float dz, float length, OdeVector out_pos, float & out_depth, unsigned long collideBits, unsigned long categoryBits)
{
	OdeThreadContext & c = contexts[s_getIdx];

	dGeomID ray = dCreateRay(NULL, length);
	dGeomSetCollideBits(ray, collideBits);
	dGeomSetCategoryBits(ray, categoryBits);
	dGeomRaySet(ray, px, py, pz, dx, dy, dz);

	dContactGeom contacts[NUM_CONTACTS];

	int flags = 0 | NUM_CONTACTS;
	int nContacts = dCollide(ray, (dGeomID)c.space, flags, contacts, sizeof(dContactGeom));

	if (nContacts > 0)
	{
		const dContactGeom * closest = std::min_element(contacts, contacts + nContacts,
			[](const dContactGeom & a, const dContactGeom & b)
		{
			return a.depth < b.depth;
		});
		memcpy(out_pos, closest->pos, sizeof(dVector3));
		out_depth = closest->depth;
	}
	dGeomDestroy(ray);
	return nContacts >0;
}

EXPORT_API void odeJointSetFmax(int jointId, float fmax1, float fmax2, float fmax3)
{
	ITERATE_THREADS(threadIdx)
	{
		dJointID odeJoint = contexts[threadIdx].joints[jointId];
		dJointType type = dJointGetType(odeJoint);
		switch (type)
		{
		case dJointTypeHinge:
			dJointSetHingeParam(odeJoint, dParamFMax, fmax1);
			break;
		case dJointTypeAMotor:
			dJointSetAMotorParam(odeJoint, dParamFMax, fmax1);
			dJointSetAMotorParam(odeJoint, dParamFMax2, fmax2);
			dJointSetAMotorParam(odeJoint, dParamFMax3, fmax3);
			break;
		default:
			std::exception("odeJointSetFMax: unsupported joint type");
		}
	}

}
static const float rad2deg = 360.0f / (2.0f*PI);
EXPORT_API void odeJointGetMotorAnglesDegrees(int jointId, OdeVector result)
{
	dJointID odeJoint = contexts[s_getIdx].joints[jointId];
	dJointType type = dJointGetType(odeJoint);
	switch (type)
	{
	case dJointTypeHinge:
		result[0] = dJointGetHingeAngle(odeJoint)*rad2deg;
		result[1] = 0;
		result[2] = 0;
		break;
	case dJointTypeAMotor:
		result[0] = dJointGetAMotorAngle(odeJoint, 0)*rad2deg;
		result[1] = dJointGetAMotorAngle(odeJoint, 1)*rad2deg;
		result[2] = dJointGetAMotorAngle(odeJoint, 2)*rad2deg;
		break;
	default:
		std::exception("odeJointGetMotorAngles: unsupported joint type");
	}
}

EXPORT_API void odeJointSetAMotorVelocitiesDegreesPerSecond(int jointId, float vel1, float vel2, float vel3)
{
	ITERATE_THREADS(threadIdx)
	{
		dJointID odeJoint = contexts[threadIdx].joints[jointId];
		dJointSetAMotorParam(odeJoint, dParamVel1, vel1*AaltoGames::deg2rad);
		dJointSetAMotorParam(odeJoint, dParamVel2, vel2*AaltoGames::deg2rad);
		dJointSetAMotorParam(odeJoint, dParamVel3, vel3*AaltoGames::deg2rad);
	}
}

EXPORT_API void odeJointSetAMotorVelocitiesRadiansPerSecond(int jointId, float vel1, float vel2, float vel3)
{
	ITERATE_THREADS(threadIdx)
	{
		dJointID odeJoint = contexts[threadIdx].joints[jointId];
		dJointSetAMotorParam(odeJoint, dParamVel1, vel1);
		dJointSetAMotorParam(odeJoint, dParamVel2, vel2);
		dJointSetAMotorParam(odeJoint, dParamVel3, vel3);
	}
}

EXPORT_API void clampControlVelocitiesAtStops(int nJoints, int *jointIds, float *velocities)
{
	int paramIdx = 0;
	for (int i = 0; i<nJoints; i++)
	{
		dJointID odeJoint = contexts[s_getIdx].joints[jointIds[i]];
		dJointType type = dJointGetType(odeJoint);
		switch (type)
		{
		case dJointTypeHinge:
		{
			float angle = dJointGetHingeAngle(odeJoint);
			float targetSpeed = velocities[paramIdx];
			if ((targetSpeed < 0 && angle <= dJointGetHingeParam(odeJoint, dParamLoStop1))
				|| (targetSpeed > 0 && angle >= dJointGetHingeParam(odeJoint, dParamHiStop1)))
			{
				velocities[paramIdx] = 0;
			}
			paramIdx++;
			break;
		}
		case dJointTypeAMotor:
		{
			float angle = dJointGetAMotorAngle(odeJoint, 0);
			float targetSpeed = velocities[paramIdx];
			if ((targetSpeed < 0 && angle <= dJointGetAMotorParam(odeJoint, dParamLoStop1))
				|| (targetSpeed > 0 && angle >= dJointGetAMotorParam(odeJoint, dParamHiStop1)))
			{
				velocities[paramIdx] = 0;
			}
			paramIdx++;
			angle = dJointGetAMotorAngle(odeJoint, 1);
			targetSpeed = velocities[paramIdx];
			if ((targetSpeed < 0 && angle <= dJointGetAMotorParam(odeJoint, dParamLoStop2))
				|| (targetSpeed > 0 && angle >= dJointGetAMotorParam(odeJoint, dParamHiStop2)))
			{
				velocities[paramIdx] = 0;
			}
			paramIdx++;
			angle = dJointGetAMotorAngle(odeJoint, 2);
			targetSpeed = velocities[paramIdx];
			if ((targetSpeed < 0 && angle <= dJointGetAMotorParam(odeJoint, dParamLoStop3))
				|| (targetSpeed > 0 && angle >= dJointGetAMotorParam(odeJoint, dParamHiStop3)))
			{
				velocities[paramIdx] = 0;
			}
			paramIdx++;
			break;
		}
		default:
			std::exception("odeJointGetMotorAngles: unsupported joint type");
		}
	}
}

EXPORT_API void setMotorVelocities(int nJoints, int *jointIds, const float *velocities)
{
	int paramIdx = 0;
	for (int i = 0; i<nJoints; i++)
	{
		dJointID odeJoint = contexts[s_getIdx].joints[jointIds[i]];
		dJointType type = dJointGetType(odeJoint);
		switch (type)
		{
		case dJointTypeHinge:
		{
			dJointSetHingeParam(odeJoint, dParamVel, velocities[paramIdx++] * AaltoGames::deg2rad);
			break;
		}
		case dJointTypeAMotor:
		{
			dJointSetAMotorParam(odeJoint, dParamVel1, velocities[paramIdx++] * AaltoGames::deg2rad);
			dJointSetAMotorParam(odeJoint, dParamVel2, velocities[paramIdx++] * AaltoGames::deg2rad);
			dJointSetAMotorParam(odeJoint, dParamVel3, velocities[paramIdx++] * AaltoGames::deg2rad);
			break;
		}
		default:
			std::exception("setMotorVelocities: unsupported joint type");
		}
	}
}

void * odeGetSpace()
{
	return contexts[s_getIdx].space;
}
