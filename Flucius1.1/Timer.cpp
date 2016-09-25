#include "Timer.h"

void Timer::step()
{
	startTime = clock() / 1000.0f;
}

float Timer::getDelta()
{
	if(isPaused)
		return pauseTime - startTime;
	else
		return clock() / 1000.0f - startTime;
}

void Timer::pause()
{
	if(isPaused)
		return;	

	isPaused = true;
	pauseTime = clock() / 1000.0f;
}

void Timer::unpause()
{
	if(!isPaused)
		return;	

	isPaused = false;
	startTime += clock() / 1000.0f - pauseTime;
}