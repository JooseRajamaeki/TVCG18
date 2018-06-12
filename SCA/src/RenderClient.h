#pragma once
#include "RenderCommands.h"

#if _MSC_VER // this is defined when compiling with Visual Studio
#define EXPORT_API __declspec(dllexport) // Visual Studio needs annotating exported functions with this
#else
#define EXPORT_API // Xode does not need annotating exported functions, so define is empty
#endif // _MSC_VER

//// Define our own types for SWIG. This is needed because arrays (such as dQuaternion) can't be returned from functions
//typedef dReal * OdeQuaternion;
//typedef dReal * OdeVector;
//typedef const dReal * ConstOdeQuaternion;
//typedef const dReal * ConstOdeVector;
//typedef const int * BodyIDList;

class RenderClientData
{
public:
	float physicsTimeStep;
	bool defaultMouseControlsEnabled;
	float maxAllowedTimeStep;
	RenderClientData()
		:physicsTimeStep(1.0f/60.0f),
		defaultMouseControlsEnabled(true),
		maxAllowedTimeStep(1.0f)
	{
	}
};


//These are the functions that the renderer will call on the client.
//Implement these in your app. 
void EXPORT_API rcInit();
void EXPORT_API rcUninit();
void EXPORT_API rcUpdate();
void EXPORT_API rcOnKeyDown(int key);
void EXPORT_API rcOnKeyUp(int key);
void EXPORT_API rcGetClientData(RenderClientData &data);
void EXPORT_API rcOnMouse(float rayStartX, float rayStartY, float rayStartZ, float rayDirX, float rayDirY, float rayDirZ, int button, int x, int y);



