#pragma once

static inline Quaternionf ode2eigenq(ConstOdeQuaternion odeq)
{
	return Quaternionf(odeq[0], odeq[1], odeq[2], odeq[3]);
}
static inline void eigen2odeq(const Quaternionf &q, OdeQuaternion out_odeq)
{
	out_odeq[0] = q.w();
	out_odeq[1] = q.x();
	out_odeq[2] = q.y();
	out_odeq[3] = q.z();
}
static const float boneDensity = 1.0f;
static const float springKp = 1000.0f;
static const float springStopKp = 1000.0f;
static const float springDamping = 1.0f;
static const float timeStep = 1.0f / 30.0f;
static const float defaultFmax = 80.0f;
static const float maxControlTorque = 80.0f;
static const float torqueControlStabilizingFmax = 5.0f;
static const float maxControlSpeed = 720.0f;
static const float controlFmaxScale = 100.0f;  //if fmax optimization used, the control vector has scaled fmax values to have all controls of approximately similar magnitude
//static const bool useTorque=false; 
static const float angleLimitBuffer = deg2rad*2.5f;




class OdeRig
{
public:
	enum BodySides
	{
		left = 0, right = 1
	};
	class Joint;
	class Bone {
	public:
		EIGEN_MAKE_ALIGNED_OPERATOR_NEW;  //needed as we have fixed size eigen matrices as members
		static const int maxBoneGeoms = 4;
		int geoms[maxBoneGeoms];
		int body;
		float length, radius;
		//BoneData *parent; //null if no parent
		Quaternionf initialRotation;
		float mass;
		Vector3f initialDirection;
		Vector3f initialEndPos;
		Joint *joint;
	};
	class Joint
	{
	public:
		EIGEN_MAKE_ALIGNED_OPERATOR_NEW;  //needed as we have fixed size eigen matrices as members
		Bone *bone1, *bone2;
		int joint;
		int motor; //0 if no motor
		int nMotorDof; //1 for hinge, 3 for ball, 0 for fixed
		Vector3f angleMin, angleMax; //for the motor/joint
	};
	std::vector<Bone *> bones;
	std::vector<Joint *> joints;
	int nTotalDofs;
	int nMotors;
	int stateDim;
	int controlDim;
	bool enableSelfCollisions;

	OdeRig()
	{
		enableSelfCollisions = false;
	}
	//call this after character-specific init that creates bones. Counts the number of dofs etc.
	void genericInit(float totalMass)
	{
		//link joints and bones
		for (Joint *j : joints)
		{
			int body1 = odeJointGetBody(j->joint, 0);
			int body2 = odeJointGetBody(j->joint, 1);
			for (Bone *b : bones)
			{
				if (b->body == body2)
				{
					j->bone2 = b;
					b->joint = j;
				}
				if (b->body == body1)
				{
					j->bone1 = b;
				}
			}
		}
		nTotalDofs = 0;
		nMotors = 0;
		for (size_t i = 0; i < joints.size(); i++)
		{
			nTotalDofs += joints[i]->nMotorDof;
			if (joints[i]->nMotorDof > 0)
				nMotors++;
		}

		controlDim = nTotalDofs;
		scaleMass(totalMass);
		float temp[1024];
		stateDim = computeStateVector(temp, Quaternionf::Identity());
		saveInitialPose();
	}
	void saveInitialPose()
	{
		Vector3f com;
		computeCOM(com);
		for (size_t i = 0; i < bones.size(); i++)
		{
			Bone *b = bones[i];
			b->initialEndPos = getBoneEndPos(b) - com;
			b->initialRotation = ode2eigenq(odeBodyGetQuaternion(b->body));
		}
	}

	void scaleMass(float newMass)
	{
		//scale total mass to desired value
		float mass = 0;
		for (size_t i = 0; i < bones.size(); i++)
		{
			mass += odeBodyGetMass(bones[i]->body);
		}
		float scaleFactor = newMass / mass;
		printf("Scaling mass by %f\n", scaleFactor);
		for (size_t i = 0; i < bones.size(); i++)
		{
			odeMassSetCapsuleTotal(bones[i]->body, odeBodyGetMass(bones[i]->body)*scaleFactor, bones[i]->radius, bones[i]->length);
			bones[i]->mass = odeBodyGetMass(bones[i]->body);
		}
	}
	float getAppliedSqJointTorques()
	{
		float result = 0;
		for (size_t i = 0; i < joints.size(); i++)
		{
			if (joints[i]->nMotorDof == 3)
			{
				result += Vector3f(odeJointGetAccumulatedTorque(joints[i]->motor, 0)).squaredNorm();
				result += Vector3f(odeJointGetAccumulatedTorque(joints[i]->motor, 1)).squaredNorm();
				//result+=Vector3f(odeJointGetAccumulatedForce(joints[i]->joint,0)).squaredNorm();
				//result+=Vector3f(odeJointGetAccumulatedForce(joints[i]->joint,1)).squaredNorm();
			}
			else if (joints[i]->nMotorDof == 1)
			{
				result += Vector3f(odeJointGetAccumulatedTorque(joints[i]->joint, 0)).squaredNorm();
				result += Vector3f(odeJointGetAccumulatedTorque(joints[i]->joint, 1)).squaredNorm();
				//result+=Vector3f(odeJointGetAccumulatedForce(joints[i]->joint,0)).squaredNorm();
				//result+=Vector3f(odeJointGetAccumulatedForce(joints[i]->joint,1)).squaredNorm();
			}
		}
		return result;
	}
	void computeCOM(Vector3f &result) const
	{
		result = Vector3f::Zero();
		float totalMass = 0;
		for (size_t i = 0; i < bones.size(); i++)
		{
			Vector3f pos(odeBodyGetPosition(bones[i]->body));
			result += pos*bones[i]->mass;
			totalMass += bones[i]->mass;
		}
		result /= totalMass;
	}
	void computeMeanVel(Vector3f &result) const
	{
		result = Vector3f::Zero();
		float totalMass = 0;
		for (size_t i = 0; i < bones.size(); i++)
		{
			Vector3f vel(odeBodyGetLinearVel(bones[i]->body));
			result += vel*bones[i]->mass;
			totalMass += bones[i]->mass;
		}
		result /= totalMass;
	}
	void pushStateVector3f(int &idx, float *state, const Vector3f &v)
	{
		state[idx++] = v[0];
		state[idx++] = v[1];
		state[idx++] = v[2];
	}
	void pushStateQuaternionf(int &idx, float *state, const Quaternionf &q)
	{
		state[idx++] = q.x();
		state[idx++] = q.y();
		state[idx++] = q.z();
		state[idx++] = q.w();
	}

	virtual int computeStateVector(float *out_state, const Quaternionf& stateRotation)
	{
		int idx = 0;
		Vector3f com;//,meanVel;
		computeCOM(com);
		com.z() = 0;



		for (size_t i = 0; i < bones.size(); i++)
		{
			Vector3f pos(odeBodyGetPosition(bones[i]->body));
			Vector3f vel_tmp(odeBodyGetLinearVel(bones[i]->body));
			Vector3f vel = stateRotation*vel_tmp;

			Vector3f avel_tmp(odeBodyGetAngularVel(bones[i]->body));
			Vector3f avel = stateRotation*avel_tmp;

			Quaternionf q_tmp = ode2eigenq(odeBodyGetQuaternion(bones[i]->body));
			Quaternionf q = stateRotation*q_tmp;

			pushStateVector3f(idx, out_state, pos - com);
			pushStateVector3f(idx, out_state, vel);
			float avelWeight = 1.0f;
			pushStateVector3f(idx, out_state, avelWeight*avel);
			pushStateQuaternionf(idx, out_state, q);
		}


		return idx;
	}
	//radians per second
	void getCurrentAngleRates(float *out_rates)
	{
		int varIdx = 0;
		for (size_t motorIdx = 0; motorIdx < joints.size(); motorIdx++)
		{
			Joint *j = joints[motorIdx];
			if (j->nMotorDof == 3)
			{
				out_rates[varIdx++] = odeJointGetAMotorAngleRate(j->motor, 0);
				out_rates[varIdx++] = odeJointGetAMotorAngleRate(j->motor, 1);
				out_rates[varIdx++] = odeJointGetAMotorAngleRate(j->motor, 2);
			}
			else if (j->nMotorDof == 1)
			{
				out_rates[varIdx++] = odeJointGetHingeAngleRate(j->joint);
			}
		}
	}
	virtual void applyControl(const float *control)
	{
		int varIdx = 0;
		for (size_t motorIdx = 0; motorIdx < joints.size(); motorIdx++)
		{
			if (joints[motorIdx]->nMotorDof == 3)
			{
				Joint *j = joints[motorIdx];
				int motor = joints[motorIdx]->motor;
				float angle[3], c[3], fmax[3];

				for (int i = 0; i < 3; i++)
				{
					angle[i] = odeJointGetAMotorAngle(motor, i);
					c[i] = control[varIdx++];
					if (fabs(j->angleMin[i] - j->angleMax[i]) < deg2rad*1.0f)
					{
						c[i] = 0;
						fmax[i] = maxControlTorque;
					}
					else
					{
						fmax[i] = torqueControlStabilizingFmax;
						if ((angle[i] <= j->angleMin[i] + angleLimitBuffer) && c[i] < 0)
						{
							fmax[i] = fabs(c[i]);
							c[i] = 0;
						}
						else if ((angle[i] >= j->angleMax[i] - angleLimitBuffer) && c[i] > 0)
						{
							fmax[i] = fabs(c[i]);
							c[i] = 0;
						}
					}
				}
				//odeJointSetAMotorParam(motor,dParamFMax1,fabs(c[0]));
				//odeJointSetAMotorParam(motor,dParamFMax2,fabs(c[1]));
				//odeJointSetAMotorParam(motor,dParamFMax3,fabs(c[2]));
				//float targetSpeed=720.0f;
				//odeJointSetAMotorVelocitiesDegreesPerSecond(motor,targetSpeed*sign(c[0]),targetSpeed*sign(c[1]),targetSpeed*sign(c[2]));

				odeJointSetAMotorVelocitiesRadiansPerSecond(motor, c[0], c[1], c[2]);

			}
			else if (joints[motorIdx]->nMotorDof == 1)
			{
				Joint *j = joints[motorIdx];
				int motor = joints[motorIdx]->joint;
				float angle[3], c[3], fmax[3];
				float buffer = 0.1f;
				for (int i = 0; i < 1; i++)
				{
					angle[i] = odeJointGetHingeAngle(motor);
					c[i] = control[varIdx++];
					fmax[i] = torqueControlStabilizingFmax;
					if ((angle[i] <= j->angleMin[i] + angleLimitBuffer) && c[i] < 0)
					{
						fmax[i] = fabs(c[i]);
						c[i] = 0;
					}
					else if ((angle[i] >= j->angleMax[i] - angleLimitBuffer) && c[i] > 0)
					{
						fmax[i] = fabs(c[i]);
						c[i] = 0;
					}
				}
				//odeJointSetHingeParam(motor,dParamFMax,fabs(c[0]));
				//float targetSpeed=720.0f;
				//odeJointSetHingeParam(motor,dParamVel,targetSpeed*sign(c[0]));

				odeJointSetHingeParam(motor, dParamVel, c[0]);

			}
		}
	}

	int getCurrentMotorAngles(float *out_angles) const
	{
		int varIdx = 0;
		int varIdxFmax = nTotalDofs;
		for (size_t motorIdx = 0; motorIdx < joints.size(); motorIdx++)
		{
			if (joints[motorIdx]->nMotorDof == 3)
			{
				for (int i = 0; i < 3; i++)
				{
					out_angles[varIdx++] = odeJointGetAMotorAngle(joints[motorIdx]->motor, i);
				}
				if (varIdxFmax < controlDim)
					out_angles[varIdxFmax++] = odeJointGetAMotorParam(joints[motorIdx]->motor, dParamFMax1);
			}
			else if (joints[motorIdx]->nMotorDof == 1)
			{
				out_angles[varIdx++] = odeJointGetHingeAngle(joints[motorIdx]->joint);
				if (varIdxFmax < controlDim)
					out_angles[varIdxFmax++] = odeJointGetHingeParam(joints[motorIdx]->joint, dParamFMax);
			}
		}

		return varIdx;
	}
	Joint *createHinge(int body1, int body2, const Vector3f &anchor, const Vector3f &axis, float minAngle, float maxAngle)
	{
		Joint *j = new Joint();
		j->nMotorDof = 1;
		int joint = odeJointCreateHinge();
		j->joint = joint;
		j->angleMin[0] = minAngle;
		j->angleMax[0] = maxAngle;
		odeJointAttach(joint, body1, body2);
		odeJointSetHingeAnchor(joint, anchor.x(), anchor.y(), anchor.z());
		odeJointSetHingeAxis(joint, axis.x(), axis.y(), axis.z());
		odeJointSetHingeParam(joint, dParamLoStop1, minAngle);
		odeJointSetHingeParam(joint, dParamHiStop1, maxAngle);

		odeJointSetHingeParam(joint, dParamFMax, defaultFmax);
		odeJointSetHingeParam(joint, dParamBounce, 0);
		odeJointSetHingeParam(joint, dParamFudgeFactor, -1);
		float cfm, erp;
		calculateCfmErp(timeStep, springKp, springDamping, cfm, erp);
		odeJointSetHingeParam(joint, dParamCFM1, cfm);
		odeJointSetHingeParam(joint, dParamERP1, erp);

		calculateCfmErp(timeStep, springStopKp, springDamping, cfm, erp);
		odeJointSetHingeParam(joint, dParamStopCFM1, cfm);
		odeJointSetHingeParam(joint, dParamStopERP1, erp);
		return j;
	}
	void calculateCfmErp(float timeStep, float kp, float kd, float &cfm, float &erp)
	{
		erp = timeStep * kp / (timeStep * kp + kd);
		cfm = 1.0f / (timeStep * kp + kd);
	}

	Joint *createMotoredBallJoint(int body1, int body2, const Vector3f &anchor, const Vector3f &axis0, const Vector3f &axis2, const Vector3f &angleMin, const Vector3f &angleMax)
	{
		Joint *j = new Joint();
		j->nMotorDof = 3;
		int joint = odeJointCreateBall();
		j->joint = joint;
		j->angleMin = angleMin;
		j->angleMax = angleMax;
		odeJointAttach(joint, body1, body2);
		odeJointSetBallAnchor(joint, anchor.x(), anchor.y(), anchor.z());
		joint = odeJointCreateAMotor();
		j->motor = joint;
		odeJointAttach(joint, body1, body2);
		float fudge = -1.0f;
		odeJointSetAMotorParam(joint, dParamFudgeFactor, fudge);
		odeJointSetAMotorParam(joint, dParamFudgeFactor2, fudge);
		odeJointSetAMotorParam(joint, dParamFudgeFactor3, fudge);
		odeJointSetAMotorMode(joint, 1);//Euler
		odeJointSetAMotorNumAxes(joint, 3);
		//ode manual says that for Euler motors, axis1 does not have to be set (we set it for visualization), axis0 and axis2 have to be perpendicular,
		//axis 0 must be relative to body 1, and axis 2 must be relative to body 2
		odeJointSetAMotorAxis(joint, 0, 1, axis0.x(), axis0.y(), axis0.z());
		odeJointSetAMotorAxis(joint, 2, 2, axis2.x(), axis2.y(), axis2.z());
		Vector3f axis1 = axis0.cross(axis2);
		odeJointSetAMotorAxis(joint, 1, 2, axis1.x(), axis1.y(), axis1.z());
		odeJointSetAMotorParam(joint, dParamLoStop1, angleMin[0]);
		odeJointSetAMotorParam(joint, dParamHiStop1, angleMax[0]);
		odeJointSetAMotorParam(joint, dParamLoStop2, angleMin[1]);
		odeJointSetAMotorParam(joint, dParamHiStop2, angleMax[1]);
		odeJointSetAMotorParam(joint, dParamLoStop3, angleMin[2]);
		odeJointSetAMotorParam(joint, dParamHiStop3, angleMax[2]);

		odeJointSetAMotorParam(joint, dParamFMax1, defaultFmax);
		odeJointSetAMotorParam(joint, dParamFMax2, defaultFmax);
		odeJointSetAMotorParam(joint, dParamFMax3, defaultFmax);

		odeJointSetAMotorParam(joint, dParamBounce1, 0);
		odeJointSetAMotorParam(joint, dParamBounce2, 0);
		odeJointSetAMotorParam(joint, dParamBounce3, 0);


		float cfm, erp;
		calculateCfmErp(timeStep, springKp, springDamping, cfm, erp);
		odeJointSetAMotorParam(joint, dParamCFM1, cfm);
		odeJointSetAMotorParam(joint, dParamCFM2, cfm);
		odeJointSetAMotorParam(joint, dParamCFM3, cfm);
		odeJointSetAMotorParam(joint, dParamERP1, erp);
		odeJointSetAMotorParam(joint, dParamERP2, erp);
		odeJointSetAMotorParam(joint, dParamERP3, erp);

		calculateCfmErp(timeStep, springStopKp, springDamping, cfm, erp);
		odeJointSetAMotorParam(joint, dParamStopCFM1, cfm);
		odeJointSetAMotorParam(joint, dParamStopCFM2, cfm);
		odeJointSetAMotorParam(joint, dParamStopCFM3, cfm);
		odeJointSetAMotorParam(joint, dParamStopERP1, erp);
		odeJointSetAMotorParam(joint, dParamStopERP2, erp);
		odeJointSetAMotorParam(joint, dParamStopERP3, erp);
		return j;

	}
	Bone *createBone(const Vector3f &pos, float l, float radius, const Vector3f &dir)
	{
		return createBone(pos.x(), pos.y(), pos.z(), l, radius, dir);
	}
	Bone *createBone(float x, float y, float z, float l, float radius, const Vector3f &dir)
	{
		Bone *bone = new Bone();
		int geom = odeCreateCapsule(0, radius, l);
		odeGeomSetCategoryBits(geom, 0x1);
		odeGeomSetCollideBits(geom, enableSelfCollisions ? 0xffffffff : 0xfffffff0);
		bone->geoms[0] = geom;
		int body = odeBodyCreate();
		float odeq[4];
		Quaternionf q = Quaternionf::FromTwoVectors(Vector3f(0, 0, 1), dir);
		eigen2odeq(q, odeq);
		odeBodySetQuaternion(body, odeq, false);
		odeBodySetPosition(body, x + 0.5f*l*dir.x(), y + 0.5f*l*dir.y(), z + 0.5f*l*dir.z());
		odeGeomSetBody(geom, body);
		odeMassSetCapsule(body, boneDensity, radius, l);
		bone->body = body;
		bone->initialRotation = ode2eigenq(odeBodyGetQuaternion(body));
		bone->length = l;
		bone->radius = radius;
		bone->initialDirection = Vector3f(0, 0, 1);//the default direction that amounts to dir when rotated
		return bone;
	}

	Vector3f getBoneStartPos(Bone *b) const
	{
		Vector3f pos(odeBodyGetPosition(b->body));
		Quaternionf q = ode2eigenq(odeBodyGetQuaternion(b->body));
		return pos - 0.5f*b->length*(q*b->initialDirection);
	}

	Vector3f getBoneEndPos(Bone *b) const
	{
		Vector3f pos(odeBodyGetPosition(b->body));
		Quaternionf q = ode2eigenq(odeBodyGetQuaternion(b->body));
		return pos + 0.5f*b->length*(q*b->initialDirection);
	}
	void drawCrosshair(const Vector3f pt)
	{
		float d = 0.1f;
		rcDrawLine(pt.x() - d, pt.y(), pt.z(), pt.x() + d, pt.y(), pt.z());
		rcDrawLine(pt.x(), pt.y() - d, pt.z(), pt.x(), pt.y() + d, pt.z());
		rcDrawLine(pt.x(), pt.y(), pt.z() - d, pt.x(), pt.y(), pt.z() + d);

	}
	//many of our rigs are legged, and cost functions often use a term for COM being on top of feet => all rigs should tell how many legs they have, if any
	virtual int numberOfLegs() const
	{
		return 0;
	}
	virtual Vector3f getFootPos(int footIndex) const
	{
		return Vector3f::Zero();
	}
	virtual void debugVisualize()
	{
		//drawCrosshair(Vector3f(odeBodyGetPosition(bones[bPelvis]->body)));
		//drawCrosshair(getBoneEndPos(bones[bLeftThigh]));
		//drawCrosshair(getBoneEndPos(bones[bRightThigh]));

		float axisDrawLength = 0.15f;
		for (size_t i = 0; i < joints.size(); i++)
		{
			Joint *j = joints[i];
			if (j->nMotorDof == 3)
			{
				Vector3f anchor;
				odeJointGetBallAnchor(j->joint, anchor.x(), anchor.y(), anchor.z());
				Vector3f axis;
				odeJointGetAMotorAxis(j->motor, 0, axis.x(), axis.y(), axis.z());
				axis *= axisDrawLength;
				axis += anchor;
				rcSetColor(1, 0, 0, 1);
				rcDrawLine(&anchor[0], &axis[0]);
				odeJointGetAMotorAxis(j->motor, 1, axis.x(), axis.y(), axis.z());
				axis *= axisDrawLength;
				axis += anchor;
				rcSetColor(0, 1, 0, 1);
				rcDrawLine(&anchor[0], &axis[0]);
				odeJointGetAMotorAxis(j->motor, 2, axis.x(), axis.y(), axis.z());
				axis *= axisDrawLength;
				axis += anchor;
				rcSetColor(0, 0, 1, 1);
				rcDrawLine(&anchor[0], &axis[0]);
				//drawCrosshair(anchor);
			}
			else if (j->nMotorDof == 1)
			{
				Vector3f anchor;
				odeJointGetHingeAnchor(j->joint, anchor.x(), anchor.y(), anchor.z());
				Vector3f axis;
				odeJointGetHingeAxis(j->joint, axis.x(), axis.y(), axis.z());
				axis *= axisDrawLength;
				axis += anchor;
				rcSetColor(1, 0, 0, 1);
				rcDrawLine(&anchor[0], &axis[0]);
			}
		}

	}

	void setFmaxForAllMotors(float f)
	{
		for (size_t i = 0; i < joints.size(); i++)
		{
			Joint *j = joints[i];
			if (j->nMotorDof == 3)
			{
				odeJointSetAMotorParam(j->motor, dParamFMax1, f);
				odeJointSetAMotorParam(j->motor, dParamFMax2, f);
				odeJointSetAMotorParam(j->motor, dParamFMax3, f);
			}
			else if (j->nMotorDof == 1)
			{
				odeJointSetHingeParam(j->joint, dParamFMax1, f);
			}
		}

	}
	void setMotorSpringConstants(float springKp, float springDamping)
	{
		float cfm, erp;
		calculateCfmErp(timeStep, springKp, springDamping, cfm, erp);
		for (size_t i = 0; i < joints.size(); i++)
		{
			Joint *j = joints[i];
			if (j->nMotorDof == 3)
			{
				odeJointSetAMotorParam(j->motor, dParamERP1, erp);
				odeJointSetAMotorParam(j->motor, dParamERP2, erp);
				odeJointSetAMotorParam(j->motor, dParamERP3, erp);
				odeJointSetAMotorParam(j->motor, dParamCFM1, cfm);
				odeJointSetAMotorParam(j->motor, dParamCFM2, cfm);
				odeJointSetAMotorParam(j->motor, dParamCFM3, cfm);
			}
			else if (j->nMotorDof == 1)
			{
				odeJointSetHingeParam(j->joint, dParamERP1, erp);
				odeJointSetHingeParam(j->joint, dParamCFM1, cfm);
			}
		}

	}
};
