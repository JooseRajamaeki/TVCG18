#ifndef WIN_KEYBOARD_H
#define WIN_KEYBOARD_H

namespace AaltoGames{

	namespace Keyboard{
		void update();
		//is the key currently down (pressed)?
		bool keyDown(int virtualKey);
		//was the key pressed down between the last two update() calls?
		bool keyHit(int virtualKey);
	}
}

#endif