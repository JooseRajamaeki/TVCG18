#pragma once

#if _MSC_VER // this is defined when compiling with Visual Studio
#define EXPORT_API __declspec(dllexport) // Visual Studio needs annotating exported functions with this
#else
#define EXPORT_API // Xode does not need annotating exported functions, so define is empty
#endif // _MSC_VER

#ifndef SWIG
#ifndef dSINGLE
#error ("This code only works with single-precision ODE");
#endif
#endif

enum RenderCommandTypes
{
	rcCapsule=0,rcSphere,rcBox,rcLine,rcMaterial,rcColor,rcViewPoint,rcPrint,rctScreenShot,rctLightPosition,rctMarkerCircle
};

enum RenderMaterials
{
    RM_NONE = 0,       /* uses the current color instead of a texture */
    RM_WOOD,
    RM_CHECKERED,
    RM_GROUND,
    RM_SKY
};


struct RCVector3
{
public:
	float x,y,z;
	RCVector3(const float *data=0)
	{
		if (data)
		{
			x=data[0];
			y=data[1];
			z=data[2];
		}
	}
	RCVector3(float x,float y,float z)
		:x(x),y(y),z(z)
	{
	}

#ifndef SWIG
	operator float *()
	{
		return (float *)this;
	}
#endif
};

struct RCQuaternion
{
	float w,x,y,z;
	RCQuaternion(const float *data=0)
	{
		if (data)
		{
			w=data[0];
			x=data[1];
			y=data[2];
			z=data[3];
		}
	}
#ifndef SWIG	
	operator float *()
	{
		return (float *)this;
	}
#endif
};

struct GeomRenderSettings
{
public:
	RenderMaterials material;
	float a,r,g,b;
};

struct RenderCommand
{
public: 
	RenderCommandTypes type;
	RenderMaterials material;
	float a,r,g,b;
	float radius,length;
	RCVector3 pos,pos2,sides;	
	RCQuaternion q;
};

int EXPORT_API rcGetNumCommands();
void EXPORT_API rcGetCommand(int idx,RenderCommand &out_rc);
void EXPORT_API rcClear();
EXPORT_API const char *  rcGetPrintedString(int idx);
int EXPORT_API rcGetNumPrintedStrings();

//The following are typically called by the client, not the renderer
#ifndef SWIG

void rcDebug (const char *msg, ...);
void rcSetMaterial (RenderMaterials material);
void rcSetColor (float red, float green, float blue, float alpha=1);
void rcAddCommand(RenderCommand &c);

void rcDrawBox (const float pos[3], const float R[12], const float sides[3]);
void rcDrawSphere (const float pos[3], const float R[12], float radius);
void rcDrawCapsule (const float pos[3], const float R[12], float length, float radius);
void rcDrawLine (const float pos1[3], const float pos2[3]);
void rcDrawLine (float x0,float y0,float z0,float x1,float y1,float z1);
void rcDrawMarkerCircle(const float pos[3], float radius);

//only supported by Unity renderer. Will be saved to Screenshots folder (in current folder), with sequential naming to allow conversion to video.
void rcTakeScreenShot(); 
//lights points from this position to origin (drawstuff does not currently support anything else...)
void rcSetLightPosition(float x, float y, float z);

struct dxSpace; //forward declare
void rcDrawAllObjects(dxSpace *space, bool geomUserDataIsRenderSettings=true);
void rcSetViewPoint(float x, float y, float z, float lookAtX, float lookAtY, float lookAtZ);
void rcPrintString(const char *str, ...);
#endif