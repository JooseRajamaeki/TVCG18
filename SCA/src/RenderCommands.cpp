#include <windows.h>
#include "RenderCommands.h"
#include <stdarg.h>
#include <varargs.h>
#include <stdio.h>

#include <vector>
#include <ode/ode.h>
#include <ode/collision_space.h>

static std::vector<RenderCommand> commands;
static std::vector<std::string> printedStrings;
static RenderMaterials currentMaterial=RM_NONE;

#ifndef dSINGLE
#error ("This code only works with single-precision ODE");
#endif


void EXPORT_API rcClear()
{
	commands.clear();
	printedStrings.clear();
	currentMaterial=RM_NONE;
}

int EXPORT_API rcGetNumCommands()
{
	return (int)commands.size();
}

void EXPORT_API rcGetCommand(int idx,RenderCommand &out_rc)
{
	memcpy(&out_rc,&commands[idx],sizeof(RenderCommand));
	//out_rc=commands[idx];
}


void rcAddCommand(RenderCommand &c)
{
	if (commands.capacity()==0)
	{
		commands.reserve(32000);
	}
	commands.push_back(c);
}


void rcDebug (const char *str, ...)
{
	char c[256];

	va_list params;
	va_start( params, str );     // params to point to the parameter list

	vsprintf_s(c, 256, str, params);
	va_end(params);
	 OutputDebugString(c);
  // *((char *)0) = 0;	 ... commit SEGVicide ?
}


void rcSetMaterial (RenderMaterials material)
{
	RenderCommand c;
	c.material=material;
	c.type=rcMaterial;
	rcAddCommand(c);
}


void rcSetColor (float red, float green, float blue, float alpha)
{
	RenderCommand c;
	c.type=rcColor;
	c.r=red;
	c.g=green;
	c.b=blue;
	c.a=alpha;
	rcAddCommand(c);
}

void rcDrawBox (const float pos[3], const float R[12], const float sides[3])
{
	RenderCommand c;
	c.type=rcBox;
	c.pos=RCVector3(pos);
	dQfromR((dReal *)&c.q,R);
	c.sides=RCVector3(sides);
	rcAddCommand(c);
}

void rcDrawSphere (const float pos[3], const float R[12], float radius)
{
	RenderCommand c;
	c.type=rcSphere;
	c.radius=radius;
	c.pos=RCVector3(pos);
	dQfromR((dReal *)&c.q,R);
	rcAddCommand(c);	
}

void rcDrawMarkerCircle(const float pos[3], float radius)
{
	RenderCommand c;
	c.type=rctMarkerCircle;
	c.radius=radius;
	c.pos=RCVector3(pos);
	rcAddCommand(c);	
}
void rcDrawCapsule (const float pos[3], const float R[12], float length, float radius)
{
	RenderCommand c;
	c.type=rcCapsule;
	c.radius=radius;
	c.length=length;
	c.pos=RCVector3(pos);
	dQfromR((dReal *)&c.q,R);
	rcAddCommand(c);	
}

void rcDrawLine (const float pos1[3], const float pos2[3])
{
	RenderCommand c;
	c.type=rcLine;
	c.pos=RCVector3(pos1);
	c.pos2=RCVector3(pos2);
	rcAddCommand(c);	
}

void rcDrawLine (float x0,float y0,float z0,float x1,float y1,float z1)
{
	RenderCommand c;
	c.type=rcLine;
	c.pos=RCVector3(x0,y0,z0);
	c.pos2=RCVector3(x1,y1,z1);
	rcAddCommand(c);	
}

void rcSetViewPoint(float x, float y, float z, float lookAtX, float lookAtY, float lookAtZ)
{
	RenderCommand c;
	c.type=rcViewPoint;
	c.pos=RCVector3(x,y,z);
	c.pos2=RCVector3(lookAtX,lookAtY,lookAtZ);
	rcAddCommand(c);	
}

void rcTakeScreenShot()
{
	RenderCommand c;
	c.type=rctScreenShot;
	rcAddCommand(c);	
}

void rcSetLightPosition(float x, float y, float z)
{
	RenderCommand c;
	c.type=rctLightPosition;
	c.pos=RCVector3(x,y,z);
	rcAddCommand(c);

}

void rcDrawAllObjects(dxSpace *space, bool geomUserDataIsRenderSettings)
{

	int nGeoms=dSpaceGetNumGeoms(space);
	for (int i=0; i<nGeoms; i++)
	{
		dGeomID geom=dSpaceGetGeom (space,i);
		int c=dGeomGetClass(geom);
		if (geomUserDataIsRenderSettings)
		{
			GeomRenderSettings *settings=(GeomRenderSettings *)dGeomGetData(geom);
			if (settings!=NULL)
			{
				rcSetColor(settings->r,settings->g,settings->b,settings->a);
				rcSetMaterial(settings->material);
			}
		}
		if (c==dSphereClass)
		{
			rcDrawSphere(dGeomGetPosition(geom),dGeomGetRotation(geom),dGeomSphereGetRadius(geom));
		}
		else if (c==dCapsuleClass)
		{
			float radius,length;
			dGeomCapsuleGetParams(geom,&radius,&length);
			rcDrawCapsule(dGeomGetPosition(geom),dGeomGetRotation(geom),length,radius);
		}
		else if (c==dBoxClass)
		{
			dVector3 lengths;
			dGeomBoxGetLengths(geom,lengths);
			rcDrawBox(dGeomGetPosition(geom),dGeomGetRotation(geom),lengths);
		}
	}
}


void rcPrintString( const char *str, ... )
{
	char c[256];

	va_list params;
	va_start( params, str );     // params to point to the parameter list

	vsprintf_s(c, 256, str, params);
	va_end(params);
	printedStrings.push_back(std::string(c));
}

EXPORT_API const char * rcGetPrintedString( int idx )
{
	return printedStrings[idx].c_str();
}

EXPORT_API int rcGetNumPrintedStrings()
{
	return printedStrings.size();
}

