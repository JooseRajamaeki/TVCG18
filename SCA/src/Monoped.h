#pragma once
#include "OdeRig.h"


class Monoped : public OdeRig
{
public:
	static const int nBones=3;
	void init(bool constrainTo2d=false, bool fixInAir=false)
	{
		const float boneRadius=0.05f;
		const float boneLen=0.5f;
		Vector3f fwd(-1,0,0);
		Vector3f up(0,0,1);
		float startHeight=0.25f+boneLen*(float)nBones;
		bones.resize(nBones);
		Vector3f startPos1(0,0,startHeight);
		Vector3f startPos2(0,0,startHeight-boneLen);
		Vector3f startPos3(0,0,startHeight-boneLen*2.0f);
		bones[1]=createBone(0,0,startHeight,boneLen,boneRadius,-up); 
		bones[0]=createBone(0,0,startHeight-boneLen,boneLen,boneRadius,-up);  //0 because bone 0 used to check falling
		bones[2]=createBone(0,0,startHeight-boneLen*2.0f,boneLen,boneRadius,-up);  

		//Vector3f hipAnchor(0,0,startHeight+boneLen*2.0f);
		Vector3f axis0(0,1,0); //sideways
		Vector3f axis2=fwd; //fwd
		float maxAngle=deg2rad*90.0f;
		Vector3f angleMin(-maxAngle,-deg2rad*15.0f,-maxAngle); 
		Vector3f angleMax(maxAngle,deg2rad*15.0f,maxAngle); 
		joints.push_back(createMotoredBallJoint(bones[1]->body,bones[0]->body,startPos2,axis0,axis2,angleMin,angleMax));
		joints.push_back(createMotoredBallJoint(bones[0]->body,bones[2]->body,startPos3,axis0,axis2,angleMin,angleMax));
		if (fixInAir)
		{
			int joint=odeJointCreateFixed();
			//int joint=odeJointCreatePlane2D();
			odeJointAttach(joint,0,bones[0]->body);
			odeJointSetFixed(joint);
		}
		genericInit(10.0f);
	}

	virtual int computeStateVector(float *out_state, const Quaternionf& stateRotation)
	{
		//the base class computes default state vector (positions,velocities,rotations,angular velocities)
		int idx=OdeRig::computeStateVector(out_state, stateRotation);
		//since ground contacts are highly significant, add foot bone tip y pos and vel with greater weight
		Vector3f footPos=getBoneEndPos(bones[nBones-1]);
		Vector3f footVel(odeBodyGetLinearVel(bones[nBones-1]->body));
		float weight=10.0f;
		out_state[idx++]=weight*footPos.z();
		out_state[idx++]=weight*footVel.z();
		return idx;
	}
	virtual int numberOfLegs() const
	{
		return 1;
	}
	virtual Vector3f getFootPos(int idx) const
	{
		assert(idx==0);
		return getBoneEndPos(bones[nBones-1]);
	}
	//virtual void debugVisualize()
	//{
	//	OdeRig::debugVisualize();
	//	rcSetColor(1,1,1,1);
	//	drawCrosshair(getFootPos(0));
	//}
};
