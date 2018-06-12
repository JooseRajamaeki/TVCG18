#pragma once
#include "OdeRig.h"



class Biped : public OdeRig
{
public:

	enum Bones
	{
		bPelvis=0,bLeftThigh,bLeftShin,bLeftFoot,bRightThigh,bRightShin,bRightFoot,bSpine,bHead,bLeftUpperArm,bLeftForeArm,bRightUpperArm,bRightForeArm,bCount
	}; //
	void init(bool constrainTo2d=false, bool fixInAir=false)
	{
		enableSelfCollisions=true;
		const float legRadius=0.05f;
		const float armRadius=0.04f;
		const float torsoRadius=0.09f;
		const float spineLen=0.25f;
		const float pelvisLen=spineLen;
		const float headRadius=0.065f;
		const float headLen=0.25f;
		float halfw=0.11f;
		float neckToShoulder=halfw*1.5f;
		float shinLen=0.45f;
		float thighLen=shinLen*0.9f;
		float foreArmLen=shinLen*0.9f;
		float upperArmLen=foreArmLen*0.8f;
		Vector3f fwd(-1,0,0);
		Vector3f up(0,0,1);
		float startHeight=0.7f;
		float pelvisShift=-0.05f;
		bones.resize(bCount,NULL);
		Vector3f bonePos(0,0,startHeight+shinLen+thighLen+pelvisLen+pelvisShift);
		//pelvis and legs
		bones[bPelvis]=createBone(bonePos,pelvisLen,torsoRadius, -up);  //thigh
		bones[bLeftThigh]=createBone(0,halfw,startHeight+shinLen+thighLen,thighLen,legRadius,-up);  //thigh
		bones[bLeftShin]=createBone(0,halfw,startHeight+shinLen,shinLen,legRadius,-up);		//shin
		bones[bRightThigh]=createBone(0,-halfw,startHeight+shinLen+thighLen,thighLen,legRadius,-up);	//thigh
		bones[bRightShin]=createBone(0,-halfw,startHeight+shinLen,shinLen,legRadius,-up);		//shin
		bones[bLeftFoot]=createBone(0.07f,halfw,startHeight,0.25f,legRadius,fwd); //foot
		bones[bRightFoot]=createBone(0.07f,-halfw,startHeight,0.25f,legRadius,fwd); //foot

		//spine and head
		Vector3f neckPos=bonePos + Vector3f(0,0,spineLen);
		bones[bSpine]=createBone(neckPos,spineLen,torsoRadius, -up);  //thigh
		bones[bHead]=createBone(neckPos+Vector3f(0,0,headLen),headLen,headRadius, -up);  //thigh

		//left arm
		float shouldersDown=0.025f;
		bonePos=neckPos+Vector3f(0,-neckToShoulder,-shouldersDown);
		bones[bLeftUpperArm]=createBone(bonePos,upperArmLen,armRadius, -up);  //thigh
		bones[bLeftForeArm]=createBone(getBoneEndPos(bones[bLeftUpperArm]),foreArmLen,armRadius, -up);  //thigh

		//right arm
		bonePos=neckPos+Vector3f(0,neckToShoulder,-shouldersDown);
		bones[bRightUpperArm]=createBone(bonePos,upperArmLen,armRadius, -up);  //thigh
		bones[bRightForeArm]=createBone(getBoneEndPos(bones[bRightUpperArm]),foreArmLen,armRadius, -up);  //thigh


		//Vector3f hipAnchor(0,0,startHeight+boneLen*2.0f);
		Vector3f axis0(0,1,0); //sideways
		Vector3f axis2=fwd; //fwd
		float hipSwingInwards=deg2rad*15.0f;
		float hipSwingOutwards=deg2rad*15.0f;
		float hipTwist=deg2rad*0.1f;
		float hipSwingBack=deg2rad*30.0f;
		float hipSwingFwd=deg2rad*45.0f;

		float spineSwingBack=deg2rad*15.0f;
		float spineSwingFwd=deg2rad*30.0f;
		float spineSwingSideways=deg2rad*15.0f;
		float spineTwist=deg2rad*15.0f;

		float headAngleLimitsScaleRelativeToSpine=0.5f;

		float shoulderSwingBack=deg2rad*60.0f;
		float shoulderSwingFwd=deg2rad*120.0f;
		float shoulderTwist=deg2rad*15.0f;
		float shoulderSwingInwards=deg2rad*15.0f;
		float shoulderSwingOutwards=deg2rad*120.0f;
		float kneeRangeMax=deg2rad*90.0f;
		float kneeRangeMin=deg2rad*5.0f;
		float elbowRangeMax=deg2rad*90.0f;
		float elbowRangeMin=deg2rad*1.0f;
		const bool relaxedLimits=true;
		if (relaxedLimits)
		{
			 hipSwingInwards=deg2rad*15.0f;
			 hipSwingOutwards=deg2rad*70.0f;
			 hipTwist=deg2rad*30.0f;
			 hipSwingBack=deg2rad*30.0f;
			 hipSwingFwd=deg2rad*130.0f;

			 spineSwingBack=deg2rad*15.0f;
			 spineSwingFwd=deg2rad*45.0f;
			 spineSwingSideways=deg2rad*30.0f;
			 spineTwist=deg2rad*30.0f;

			 headAngleLimitsScaleRelativeToSpine=0.5f;

			 shoulderSwingBack=deg2rad*60.0f;
			 shoulderSwingFwd=deg2rad*150.0f;
			 shoulderTwist=deg2rad*15.0f;
			 shoulderSwingInwards=deg2rad*15.0f;
			 shoulderSwingOutwards=deg2rad*150.0f;
			 kneeRangeMax=deg2rad*160.0f;
			 kneeRangeMin=deg2rad*5.0f;
			 elbowRangeMax=deg2rad*120.0f;
			 elbowRangeMin=deg2rad*1.0f;
		}

		Vector3f angleMin(-hipSwingFwd,-hipTwist,-hipSwingInwards); 
		Vector3f angleMax(hipSwingBack,hipTwist,hipSwingOutwards);

		//leg joints
		joints.push_back(createMotoredBallJoint(bones[bPelvis]->body,bones[bLeftThigh]->body,getBoneStartPos(bones[bLeftThigh]),axis0,axis2,angleMin,angleMax));
		angleMin.z()=-hipSwingOutwards;
		angleMax.z()=hipSwingInwards;
		joints.push_back(createMotoredBallJoint(bones[bPelvis]->body,bones[bRightThigh]->body,getBoneStartPos(bones[bRightThigh]),axis0,axis2,angleMin,angleMax));
		joints.push_back(createHinge(bones[bLeftThigh]->body,bones[bLeftShin]->body,getBoneStartPos(bones[bLeftShin]),axis0,kneeRangeMin,kneeRangeMax));
		joints.push_back(createHinge(bones[bRightThigh]->body,bones[bRightShin]->body,getBoneStartPos(bones[bRightShin]),axis0,kneeRangeMin,kneeRangeMax));
		joints.push_back(createHinge(bones[bLeftShin]->body,bones[bLeftFoot]->body,Vector3f(0,halfw,startHeight),axis0,-deg2rad*15.0f,deg2rad*45.0f));
		joints.push_back(createHinge(bones[bRightShin]->body,bones[bRightFoot]->body,Vector3f(0,-halfw,startHeight),axis0,-deg2rad*15.0f,deg2rad*45.0f));

		//spinal joints
		angleMin=Vector3f(-spineSwingFwd,-spineTwist,-spineSwingSideways); 
		angleMax=Vector3f(spineSwingBack,spineTwist,spineSwingSideways);
		joints.push_back(createMotoredBallJoint(bones[bSpine]->body,bones[bPelvis]->body,getBoneStartPos(bones[bPelvis]),axis0,axis2,angleMin,angleMax));
		joints.push_back(createMotoredBallJoint(bones[bHead]->body,bones[bSpine]->body,getBoneStartPos(bones[bSpine]),axis0,axis2,headAngleLimitsScaleRelativeToSpine*angleMin,headAngleLimitsScaleRelativeToSpine*angleMax));

		//left arm joints
		angleMin=Vector3f(-shoulderSwingFwd,-shoulderTwist,-shoulderSwingOutwards); 
		angleMax=Vector3f(shoulderSwingBack,shoulderTwist,shoulderSwingInwards);
		Joint *leftShoulderJoint=createMotoredBallJoint(bones[bSpine]->body,bones[bLeftUpperArm]->body,getBoneStartPos(bones[bLeftUpperArm]),axis0,axis2,angleMin,angleMax);
		joints.push_back(leftShoulderJoint);
		Joint *leftElbowJoint=createHinge(bones[bLeftUpperArm]->body,bones[bLeftForeArm]->body,getBoneStartPos(bones[bLeftForeArm]),axis0,-elbowRangeMax,-elbowRangeMin);
		joints.push_back(leftElbowJoint);

		//right arm joints
		float temp=angleMin.z();
		angleMin.z()=-fabs(angleMax.z());
		angleMax.z()=fabs(temp);
		Joint *rightShoulderJoint=createMotoredBallJoint(bones[bSpine]->body,bones[bRightUpperArm]->body,getBoneStartPos(bones[bRightUpperArm]),axis0,axis2,angleMin,angleMax);
		joints.push_back(rightShoulderJoint);
		Joint *rightElbowJoint=createHinge(bones[bRightUpperArm]->body,bones[bRightForeArm]->body,getBoneStartPos(bones[bRightForeArm]),axis0,-elbowRangeMax,-elbowRangeMin);
		joints.push_back(rightElbowJoint);

		//relax shoulders
		//float shoulderFMax=1.0f;
		//odeJointSetAMotorParam(leftShoulderJoint->motor,dParamFMax1,shoulderFMax);
		//odeJointSetAMotorParam(leftShoulderJoint->motor,dParamFMax2,shoulderFMax);
		//odeJointSetAMotorParam(leftShoulderJoint->motor,dParamFMax3,shoulderFMax);
		//odeJointSetAMotorParam(rightShoulderJoint->motor,dParamFMax1,shoulderFMax);
		//odeJointSetAMotorParam(rightShoulderJoint->motor,dParamFMax2,shoulderFMax);
		//odeJointSetAMotorParam(rightShoulderJoint->motor,dParamFMax3,shoulderFMax);
		//odeJointSetHingeParam(leftElbowJoint->joint,dParamFMax1,shoulderFMax);
		//odeJointSetHingeParam(rightElbowJoint->joint,dParamFMax1,shoulderFMax);
		if (fixInAir)
		{
			int joint=odeJointCreateFixed();
			odeJointAttach(joint,0,bones[0]->body);
			odeJointSetFixed(joint);
		}
		genericInit(30.0f);
	}

	virtual int computeStateVector(float *out_state,const Quaternionf& stateRotation)
	{
		int idx=0;
		//since ground contacts are highly significant, add foot bone tip y pos and vel with greater weight
		for (int i=0; i<=1; i++)
		{
			Vector3f footPos=getFootPos((BodySides)i);
			Vector3f footVel(odeBodyGetLinearVel(getFootBone((BodySides)i)->body));
			float weight=10.0f;
			out_state[idx++]=weight*footPos.z();
			out_state[idx++]=weight*footVel.z();
		}
		//the base class computes default state vector (positions,velocities,rotations,angular velocities)
		idx+=OdeRig::computeStateVector(&out_state[idx],stateRotation);

		return idx;
	}
	virtual int numberOfLegs() const
	{
		return 2;
	}
	Bone *getFootBone(BodySides side) const
	{
		Bone *b=side==right ? bones[bLeftFoot] : bones[bRightFoot];
		return b;
	}
	virtual Vector3f getFootPos(int side) const
	{
		Bone *b=getFootBone((BodySides)side);
		Vector3f pos(odeBodyGetPosition(b->body));
		return pos;
	}
	virtual void debugVisualize()
	{
		OdeRig::debugVisualize();
		//drawCrosshair(getFootPos(left));
		//drawCrosshair(getFootPos(right));
	}
	virtual void applyControl(const float *control)
	{
		OdeRig::applyControl(control);

		const bool use_soft_tissue = false;
		if (use_soft_tissue) {
			//soft tissue force that pushes arms when near torso
			Bones armBones[2] = { bLeftUpperArm,bRightUpperArm };
			Quaternionf torsoQ = ode2eigenq(odeBodyGetQuaternion(bones[bSpine]->body));
			for (int i = 0; i < 2; i++)
			{
				Bone *b = bones[armBones[i]];
				Quaternionf armQ = ode2eigenq(odeBodyGetQuaternion(b->body));
				float angle = torsoQ.angularDistance(armQ);
				float pushTorque = 25.0f*expf(-squared(angle / (deg2rad*20.0f)));
				//fwd axis is axis 2 (zero based index)
				if (i == 0)
					odeJointAddAMotorTorques(b->joint->motor, 0, 0, -pushTorque);
				else
					odeJointAddAMotorTorques(b->joint->motor, 0, 0, pushTorque);
			}
		}
	}

};
