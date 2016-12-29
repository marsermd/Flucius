#ifndef TIMER_H
#define TIMER_H
#include <ctime>

class Timer {
public:
	Timer();
	~Timer()	{}

	void step();
	float getDelta();
	void pause();
	void unpause();

protected:
	bool isPaused;
	float pauseTime;
	float startTime;
};

#endif