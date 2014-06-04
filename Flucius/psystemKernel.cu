#include "PSystem.h"

#include <glm\gtx\transform.hpp>
#include <glm\gtx\norm.hpp>
#include <cuda_runtime.h>
#include "device_launch_parameters.h" 
#include <thrust\device_ptr.h>
#include <thrust\detail\raw_pointer_cast.h>

#include "cudaHelper.h"


__global__ void boxCollisionResponse( Particle* particles)
{

}

void PSystem::update() {

}