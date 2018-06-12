#include <stdlib.h>
#include <Windows.h>
#include "Keyboard.h"
#include <assert.h>

namespace AaltoGames{
	namespace Keyboard{
		static bool state[256];
		static bool oldState[256];
		static bool initMe=true;
		bool keyDown(int vkey){
				SHORT state=GetAsyncKeyState(vkey);
				if (state<0)
					return true;
				return false;
		}
		void update()
		{
			for (int i=0; i<256; i++){
				oldState[i]=state[i];
				state[i]=keyDown(i);
			}
			if (initMe){
				memcpy(oldState,state,sizeof(state));
				initMe=false;
			}
		}

		bool keyHit( int virtualKey )
		{
			assert(virtualKey>=0 && virtualKey<256);
			if(state[virtualKey] && !oldState[virtualKey])
				return true;
			return false;
		}
	}
}