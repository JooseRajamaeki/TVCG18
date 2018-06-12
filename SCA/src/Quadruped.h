#pragma once
#include "OdeRig.h"

class Quadruped : public OdeRig
{
public:
	enum Bones
	{
		bPelvis=0,bLeftThigh,bLeftShin,bRightThigh,bRightShin,bLeftUpperArm,bLeftForeArm,bRightUpperArm,bRightForeArm,bCount
	};

	void init(bool constrainTo2d=false, bool fixInAir=false)
	{
		enableSelfCollisions=true;
		const float torsoRadius=0.1f;
		float halfw=0.12f;
		float torsoLen=0.75f;
		Vector3f fwd(-1,0,0);
		Vector3f up(0,0,1);
		Vector3f right(0,1,0);
		float startHeight=1.5f;
		bones.resize(bCount);
		Vector3f pelvisPos(0,0,startHeight);

		bones[bPelvis]=createBone(pelvisPos.x(),pelvisPos.y(),pelvisPos.z(),torsoLen,torsoRadius, fwd); 
		createLeg(bones[bPelvis],getBoneStartPos(bones[bPelvis])+halfw*right,bRightThigh,bRightShin,false);
		createLeg(bones[bPelvis],getBoneStartPos(bones[bPelvis])-halfw*right,bLeftThigh,bLeftShin,false);
		createLeg(bones[bPelvis],getBoneEndPos(bones[bPelvis])+halfw*right,bRightUpperArm,bRightForeArm,false);
		createLeg(bones[bPelvis],getBoneEndPos(bones[bPelvis])-halfw*right,bLeftUpperArm,bLeftForeArm,false);

		if (fixInAir)
		{
			int joint=odeJointCreateFixed();
			//int joint=odeJointCreatePlane2D();
			odeJointAttach(joint,0,bones[bPelvis]->body);
			odeJointSetFixed(joint);
		}
		genericInit(30.0f);
	}
	void createLeg(const Bone *parent, const Vector3f &startPos, Bones upperBone,Bones lowerBone, bool flipElbow)
	{
		const float legRadius=0.05f;
		float shinLen=0.45f;
		float thighLen=shinLen*0.9f;
//		float shinLen=0.3f;
//		float thighLen=shinLen;
		Vector3f down(0,0,-1);
		bones[upperBone]=createBone(startPos.x(),startPos.y(),startPos.z(),thighLen,legRadius,down);  //thigh
		Vector3f startPos2=getBoneEndPos(bones[upperBone]);
		bones[lowerBone]=createBone(startPos2.x(),startPos2.y(),startPos2.z(),shinLen,legRadius,down);
		float hipSwingInwards=deg2rad*30.0f;
		float hipSwingOutwards=deg2rad*30.0f;
		float hipTwist=deg2rad*0.1f;  //TODO hinge2 joints
		float hipSwingFwd=deg2rad*30.0f;
		float hipSwingBwd=deg2rad*30.0f;
		Vector3f angleMin(-hipSwingBwd,-hipTwist,-hipSwingInwards); 
		Vector3f angleMax(hipSwingFwd,hipTwist,hipSwingOutwards);
		Vector3f fwd(-1,0,0);
		Vector3f up(0,0,1);
		Vector3f right(0,1,0);
		Vector3f axis0(0,1,0); //sideways
		Vector3f axis2=fwd; //fwd
		joints.push_back(createMotoredBallJoint(parent->body,bones[upperBone]->body,startPos,axis0,axis2,angleMin,angleMax));
		if (flipElbow)
			joints.push_back(createHinge(bones[upperBone]->body,bones[lowerBone]->body,startPos2,axis0,deg2rad*1.0f,deg2rad*90.0f));
		else
			joints.push_back(createHinge(bones[upperBone]->body,bones[lowerBone]->body,startPos2,axis0,-deg2rad*90.0f,-deg2rad*1.0f));
	}
	virtual int numberOfLegs() const
	{
		return 4;
	}
	virtual Vector3f getFootPos(int footIndex) const
	{
		Bones footBones[4]={bLeftForeArm,bRightForeArm,bLeftShin,bRightShin};	
		return getBoneEndPos(bones[footBones[footIndex]]);
	}
	virtual int computeStateVector(float *out_state, const Quaternionf& stateRotation)
	{
		//the base class computes default state vector (positions,velocities,rotations,angular velocities)
		int idx=OdeRig::computeStateVector(out_state, stateRotation);

		//since ground contacts are highly significant, add foot bone tip y pos and vel with greater weight
		Bones footBones[4]={bLeftForeArm,bRightForeArm,bLeftShin,bRightShin};
		for (int i=0; i<4; i++)
		{
			Vector3f footPos=getBoneEndPos(bones[footBones[i]]);
			Vector3f footVel(odeBodyGetLinearVel(bones[footBones[i]]->body));
			float weight=10.0f;
			out_state[idx++]=weight*footPos.z();
			out_state[idx++]=weight*footVel.z();
		}

		//for symmetric gaits, it's also very relevant which side is higher
		//{
		//	float weight=10.0f;
		//	Vector3f shoulderDiff=getBoneStartPos(bones[bLeftUpperArm])-getBoneStartPos(bones[bRightUpperArm]);
		//	out_state[idx++]=weight*shoulderDiff.y();
		//	shoulderDiff=getBoneStartPos(bones[bLeftThigh])-getBoneStartPos(bones[bRightThigh]);
		//	out_state[idx++]=weight*shoulderDiff.y();
		//}
		return idx;
	}

};
