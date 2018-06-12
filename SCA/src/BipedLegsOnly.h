#pragma once
#include "OdeRig.h"

#define USEFEET

class BipedLegsOnly : public OdeRig
{
public:
	enum Bones
	{
#ifdef USEFEET
		bLeftThigh=0,bLeftShin,bLeftFoot,bRightThigh,bRightShin,bRightFoot,bCount
#else
		bPelvis=0,bLeftThigh,bLeftShin,bRightThigh,bRightShin,bCount
#endif
	};
	void init(bool constrainTo2d=false, bool fixInAir=false)
	{
		enableSelfCollisions=true;
		const float legRadius=0.05f;
		const float torsoRadius=0.1f;
		float halfw=0.12f;
		float shinLen=0.45f;
		float thighLen=shinLen*0.9f;
		Vector3f fwd(-1,0,0);
		Vector3f up(0,0,1);
		float startHeight=0.5f;
		bones.resize(bCount);
		Vector3f pelvisPos(0,0,startHeight+shinLen+thighLen);
		//bones[bPelvis]=createBone(pelvisPos.x(),pelvisPos.y(),pelvisPos.z(),torsoRadius,torsoRadius, -up);  //thigh
		bones[bLeftThigh]=createBone(0,halfw,startHeight+shinLen+thighLen,thighLen,legRadius,-up);  //thigh
		bones[bLeftShin]=createBone(0,halfw,startHeight+shinLen,shinLen,legRadius,-up);		//shin
		bones[bRightThigh]=createBone(0,-halfw,startHeight+shinLen+thighLen,thighLen,legRadius,-up);	//thigh
		bones[bRightShin]=createBone(0,-halfw,startHeight+shinLen,shinLen,legRadius,-up);		//shin
#ifdef USEFEET
			bones[bLeftFoot]=createBone(0.07f,halfw,startHeight,0.25f,legRadius,fwd); //foot
			bones[bRightFoot]=createBone(0.07f,-halfw,startHeight,0.25f,legRadius,fwd); //foot
#endif
		//Vector3f hipAnchor(0,0,startHeight+boneLen*2.0f);
		Vector3f axis0(0,1,0); //sideways
		Vector3f axis2=fwd; //fwd
		float hipSwingInwards=deg2rad*15.0f;
		float hipSwingOutwards=deg2rad*15.0f;
		float hipTwist=deg2rad*0.1f;
		float hipSwingBack=deg2rad*30.0f;
		float hipSwingFwd=deg2rad*45.0f;
		Vector3f angleMin(-hipSwingFwd,-hipTwist,-hipSwingInwards); 
		Vector3f angleMax(hipSwingBack,hipTwist,hipSwingOutwards);

		//joints.push_back(createMotoredBallJoint(bones[bPelvis]->body,bones[bLeftThigh]->body,getBoneStartPos(bones[bLeftThigh]),axis0,axis2,angleMin,angleMax));
		//angleMin.z()=-hipSwingOutwards;
		//angleMax.z()=hipSwingInwards;
		//joints.push_back(createMotoredBallJoint(bones[bPelvis]->body,bones[bRightThigh]->body,getBoneStartPos(bones[bRightThigh]),axis0,axis2,angleMin,angleMax));

		joints.push_back(createMotoredBallJoint(bones[bRightThigh]->body,bones[bLeftThigh]->body,pelvisPos,axis0,axis2,angleMin,angleMax));


		joints.push_back(createHinge(bones[bLeftThigh]->body,bones[bLeftShin]->body,getBoneStartPos(bones[bLeftShin]),axis0,deg2rad*5.0f,deg2rad*90.0f));

		joints.push_back(createHinge(bones[bRightThigh]->body,bones[bRightShin]->body,getBoneStartPos(bones[bRightShin]),axis0,deg2rad*5.0f,deg2rad*90.0f));
#ifdef USEFEET
		joints.push_back(createHinge(bones[bLeftShin]->body,bones[bLeftFoot]->body,Vector3f(0,halfw,startHeight),axis0,-deg2rad*15.0f,deg2rad*30.0f));
		joints.push_back(createHinge(bones[bRightShin]->body,bones[bRightFoot]->body,Vector3f(0,-halfw,startHeight),axis0,-deg2rad*15.0f,deg2rad*30.0f));
#endif
		if (fixInAir)
		{
			int joint=odeJointCreateFixed();
			//int joint=odeJointCreatePlane2D();
			odeJointAttach(joint,0,bones[0]->body);
			odeJointSetFixed(joint);
		}
		genericInit(30.0f);
	}

	virtual int computeStateVector(float *out_state, const Quaternionf& stateRotation)
	{
		//the base class computes default state vector (positions,velocities,rotations,angular velocities)
		int idx=OdeRig::computeStateVector(out_state, stateRotation);
		//since ground contacts are highly significant, add foot bone tip y pos and vel with greater weight
		for (int i=0; i<=1; i++)
		{
			Vector3f footPos=getFootPos((BodySides)i);
			Vector3f footVel(odeBodyGetLinearVel(getFootBone((BodySides)i)->body));
			float weight=10.0f;
			out_state[idx++]=weight*footPos.y();
			out_state[idx++]=weight*footVel.y();
		}
		return idx;
	}
	virtual int numberOfLegs() const
	{
		return 2;
	}
	Bone *getFootBone(BodySides side) const
	{
#ifdef USEFEET
		Bone *b=side==right ? bones[bLeftFoot] : bones[bRightFoot];
#else
		Bone *b=side==right ? bones[bLeftShin] : bones[bRightShin];
#endif
		return b;
	}
	virtual Vector3f getFootPos(int side) const
	{
		Bone *b=getFootBone((BodySides)side);
#ifdef USEFEET
		Vector3f pos(odeBodyGetPosition(b->body));
		return pos;
#else
		return getBoneEndPos(b);
#endif
	}
	//virtual void debugVisualize()
	//{
	//	OdeRig::debugVisualize();
	//	drawCrosshair(getFootPos(left));
	//	drawCrosshair(getFootPos(right));
	//}
};
