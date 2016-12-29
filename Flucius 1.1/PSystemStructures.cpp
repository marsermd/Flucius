#include "PSystemStructures.h"

void SPH::Settings::setRestDencity(float value)
{
	oneOverRestDencity = 1 / value;
}

void SPH::Settings::setGravity(glm::vec3 value)
{
	gravity[0] = value[0];
	gravity[1] = value[1];
	gravity[2] = value[2];
}