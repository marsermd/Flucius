#include "grid.h"

#include <math.h>
#include <glm\glm.hpp>
#include <glm\gtx\norm.hpp>
#include <stdio.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h" 
#include <thrust\device_ptr.h>
#include <thrust\detail\raw_pointer_cast.h>

#include "tables.h"
#include "Partition3D.h"
#include "cudaHelper.h"
#include "PSystem.h"

#define uint unsigned int
#define PI 3.1415f

//________________________________INLINE HELPERS________________________________________________________
inline __device__ __host__ float lerp(float a, float b, float t)
{
	return a + t*(b-a);
}

inline __device__ __host__ glm::vec3 lerp(glm::vec3 &a, glm::vec3 &b, float t)
{
	return a + t*(b-a);
}

//_______________________________DEVICE VARIABLES________________________________________________________________________________________________________

int *triTable_dev;
int *vertsCountTable_dev;

int *verticesOccupied_dev = 0;
int *verticesOccupiedScan_dev = 0;
int *verticesCompact_dev = 0;

int *verticesToCubes_dev = 0;
int *cubesToVertices_dev = 0;

int *cubesOccupied_dev = 0;
int *cubesOccupiedScan_dev = 0;
int *cubesCompact_dev = 0;
int *cubeVerticesCnt_dev = 0;
int *cubeVerticesScan_dev = 0;

Vertex *triangleVertices_dev = 0; 

struct cudaGraphicsResource *cuda_vbo_resource;

//_______________________________CUDA PART___________________________________________________________________________________________________________

int ThrustExScanWrapper(int *output, int *input, unsigned int numElements)
{
	thrust::exclusive_scan(
		thrust::device_ptr<int>(input),
		thrust::device_ptr<int>(input + numElements), // * sizeof(int)?
		thrust::device_ptr<int>(output)
		);
	checkCudaErrorsWithLine("thrust scan failed");
	int lastElement, lastScanElement;
	cudaMemcpy((void *) &lastElement, (void *)(input + numElements - 1), sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy((void *) &lastScanElement, (void *)(output + numElements - 1), sizeof(int), cudaMemcpyDeviceToHost);
	checkCudaErrorsWithLine("copy from thrust scan to device failed");
	return lastElement + lastScanElement;
}

__global__ void resetValuesKernel(GridVertex * vertices, int vCount)
{
	int fullID = min(blockIdx.y * 65535 * THREADS_CNT + blockIdx.x * THREADS_CNT + threadIdx.x, vCount - 1);
	vertices[fullID].value = 0.0f;

	/*int z = vertices[fullID].id % 31;
	int y = (vertices[fullID].id / 31) % 31;
	int x = (vertices[fullID].id / (31 * 31)) % 31;*/

	vertices[fullID].normal = glm::vec3(0);
}

#define GRID_R2 0.0004
__constant__ float R4 = GRID_R2 * GRID_R2;
__global__ void calcGridKernel(glm::vec3 * particles, GridVertex * vertices, int pCount, int vCount, int cnt, int * partitions, int * partitionIdx,
							   int maxItems, float r, int size, int ttlCount)
{
	int fullID = threadIdx.x + blockIdx.x * THREADS_CNT + blockIdx.y * 65535 * THREADS_CNT;

	int particleID = fullID / cnt;
	int neighbourID = (fullID % cnt) / maxItems;
	int elementID = (fullID % cnt) % maxItems;

	if (particleID >= pCount || neighbourID >= NEIGHBOURS_3D) return;
	glm::vec3 p = particles[particleID];
	int x,y,z;
	x = GET_CLOSEST_POS(p.x, r);
	y = GET_CLOSEST_POS(p.y, r);
	z = GET_CLOSEST_POS(p.z, r);

	z += neighbourID % 3 - 1;
	neighbourID /= 3;
	y += neighbourID % 3 - 1;
	neighbourID /= 3;
	x += neighbourID % 3 - 1;

	int partitionID = __mul24(size, __mul24(size, x)) + __mul24(size, y) + z;
	if (partitionID >= ttlCount || partitionID < 0) return;
	int baseVertexID = partitions[partitionID];
	int vertexID = baseVertexID + elementID;
	if (vertexID >= vCount || partitionIdx[vertexID] != partitionIdx[baseVertexID]) return;

	glm::vec3 diff = vertices[vertexID].pos - p;
	float dist2 = glm::dot(diff, diff);
	float backDist4 = __fmul_rz(R4, __frcp_rz((max(__fmul_rz(dist2, dist2), 0.0000001f))));
	diff *= backDist4;
	vertices[vertexID].normal += diff;
	vertices[vertexID].value += backDist4;

	/*atomicAdd(&vertices[vertexID].normal.x, diff.x);
	atomicAdd(&vertices[vertexID].normal.y, diff.y);
	atomicAdd(&vertices[vertexID].normal.z, diff.z);
	atomicAdd(&vertices[vertexID].value,    backDist4);*/
}

__global__ void resetCubes(GridCube * cubes, int cCount) 
{
	int id = min(threadIdx.x + blockIdx.x * THREADS_CNT + blockIdx.y * 65535 * THREADS_CNT, cCount - 1);
	cubes[id].cubeID = 0;
}

__global__ void classifyVertices(GridVertex * vertices, int * verticesOccupied, int vCount, float threshold) {
	int id = min(threadIdx.x + blockIdx.x * THREADS_CNT + blockIdx.y * 65535 * THREADS_CNT, vCount - 1);
	verticesOccupied[id] = (int) (vertices[id].value > threshold);
}

__global__ void compactVertices(int * verticesCompact, int * verticesOccupied, int * verticesOccupiedScan, int vCount) {
	int id = threadIdx.x + blockIdx.x * THREADS_CNT + blockIdx.y * 65535 * THREADS_CNT;
	if (id < vCount && verticesOccupied[id]) {
		verticesCompact[verticesOccupiedScan[id]] = id;
	}
}

__global__ void calcCubeIndices(GridVertex * vertices, int * verticesToCubes, int vCount, int * verticesCompact, GridCube * cubes, int vCompactCount) {
	int id = threadIdx.x + blockIdx.x * THREADS_CNT + blockIdx.y * 65535 * THREADS_CNT;
	int listId = id / vCompactCount;
	id = verticesCompact[id % vCompactCount];
	int cube = verticesToCubes[vertices[id].id + vCount * listId];
	if (listId < 8 && cube >= 0) {
		int val = 1 << listId;
		cubes[cube].cubeID |= val;
	}
}

__global__ void classifyCubes(GridCube * cubes, int cCount, int * cubesOccupied, int *cubeVerticesCnt, int *vertsCountTable)
{
	int id = min(threadIdx.x + blockIdx.x * THREADS_CNT + blockIdx.y * 65535 * THREADS_CNT, cCount - 1);
	int cubeIndex = cubes[id].cubeID;
	cubesOccupied[id] = cubeIndex != 0 && cubeIndex != 255;
	cubeVerticesCnt[id] = vertsCountTable[cubeIndex];
}

__global__ void compactCubes(int *cubesCompact, int *cubesOccupied, int *cubesOccupiedScan, GridCube *cubes, int cCount)
{
	int id = threadIdx.x + blockIdx.x * THREADS_CNT + blockIdx.y * 65535 * THREADS_CNT;
	if (id < cCount && cubesOccupied[id]) {
		cubesCompact[cubesOccupiedScan[id]] = id;
	}
}


__device__ void vertexInterp2(GridVertex &v0, GridVertex &v1, float threshold, glm::vec3 &p, glm::vec3 &n)
{
	float t;
	if (v0.value - threshold < 0.0001f)
	{
		t = 0;
	} 
	else if (v1.value - threshold < 0.0001f)
	{
		t = 1;
	}
	else
	{
		t = (threshold - v0.value) / (v1.value - v0.value);
	}
	p = lerp(v0.pos, v1.pos, t);
	n = lerp(v0.normal, v1.normal, t);
	n = glm::normalize(n);
}

__global__ void generateTriangles(
	Vertex *triangleVertices, 
	int *cubesCompact, GridCube *cubes, int * cubesToVertices, 
	int *cubeVerticesScan, GridVertex *vertices, 
	int *trianglesTable, int *vertsCountTable,
	int activeCubes, int maxVerts, float threshold)
{
	int id = min(threadIdx.x + blockIdx.x * THREADS_CNT + blockIdx.y * 65535 * THREADS_CNT, activeCubes - 1);
	id = cubesCompact[id];
	glm::vec3 vertlist[12];
	glm::vec3 normlist[12];

	GridCube currentCube = cubes[id];

	GridVertex curVertices[8] = {
		vertices[cubesToVertices[id * 8 + 0]],
		vertices[cubesToVertices[id * 8 + 1]],	
		vertices[cubesToVertices[id * 8 + 2]],	
		vertices[cubesToVertices[id * 8 + 3]],	
		vertices[cubesToVertices[id * 8 + 4]],	
		vertices[cubesToVertices[id * 8 + 5]],	
		vertices[cubesToVertices[id * 8 + 6]],	
		vertices[cubesToVertices[id * 8 + 7]]	
	};

	vertexInterp2(curVertices[0], curVertices[1], threshold, vertlist[0], normlist[0]);
	vertexInterp2(curVertices[1], curVertices[2], threshold, vertlist[1], normlist[1]);
	vertexInterp2(curVertices[2], curVertices[3], threshold, vertlist[2], normlist[2]);
	vertexInterp2(curVertices[3], curVertices[0], threshold, vertlist[3], normlist[3]);

	vertexInterp2(curVertices[4], curVertices[5], threshold, vertlist[4], normlist[4]);
	vertexInterp2(curVertices[5], curVertices[6], threshold, vertlist[5], normlist[5]);
	vertexInterp2(curVertices[6], curVertices[7], threshold, vertlist[6], normlist[6]);
	vertexInterp2(curVertices[7], curVertices[4], threshold, vertlist[7], normlist[7]);

	vertexInterp2(curVertices[0], curVertices[4], threshold, vertlist[8], normlist[8]);
	vertexInterp2(curVertices[1], curVertices[5], threshold, vertlist[9], normlist[9]);
	vertexInterp2(curVertices[2], curVertices[6], threshold, vertlist[10], normlist[10]);
	vertexInterp2(curVertices[3], curVertices[7], threshold, vertlist[11], normlist[11]);


	int numVerts = vertsCountTable[currentCube.cubeID];
	for (int i = 0; i < numVerts; i++)
	{
		GLuint edge = trianglesTable[currentCube.cubeID * 16 + i];
		int index = cubeVerticesScan[id] + i;

		if (index < maxVerts)
		{
			for (int j = 0; j < 3; j++) {
				triangleVertices[index].position[j] = vertlist[edge][j];
				triangleVertices[index].normal[j] = normlist[edge][j];
			}
		}
	}
}

//_______________________________LAUNCHERS___________________________________________________________________________________________________

int Grid::cudaCalcGrid(glm::vec3 * particles_dev, int pCount) {	
	resetValuesKernel<<<getBlocks(numVertices), getThreads(numVertices)>>>(vertices_dev, numVertices);
	cudaDeviceSynchronize();
	checkCudaErrorsWithLine("failed reseting vertices values");

	int cnt = NEIGHBOURS_3D * partition3D->maxItemsPerPartition;
	cnt = ((cnt - 1) / THREADS_CNT + 1) * THREADS_CNT;
	calcGridKernel<<<getBlocks(cnt * pCount), getThreads(cnt * pCount)>>>(particles_dev, vertices_dev, pCount, numVertices, cnt, 
		partition3D->partitions_dev, partition3D->partitionIdx_dev, partition3D->maxItemsPerPartition, partition3D->r,
		partition3D->countx, partition3D->ttlCount);
	cudaDeviceSynchronize();
	checkCudaErrorsWithLine("failed calculating grid");
	return 0;
}

int activeCubes, totalVertices;
int Grid::cudaAnalyzeCubes(float threshold) {
	classifyVertices<<<getBlocks(numVertices), getThreads(numVertices)>>>(vertices_dev, verticesOccupied_dev, numVertices, threshold);
	cudaDeviceSynchronize();
	checkCudaErrorsWithLine("failed classify vertices");
	int activeVertices = ThrustExScanWrapper(verticesOccupiedScan_dev, verticesOccupied_dev, numVertices);
	compactVertices<<<getBlocks(numVertices), getThreads(numVertices)>>>(verticesCompact_dev, verticesOccupied_dev, verticesOccupiedScan_dev, numVertices);
	checkCudaErrorsWithLine("failed compact vertices");
	

	resetCubes<<<getBlocks(numCubes), getThreads(numCubes)>>>(cubes_dev, numCubes);
	calcCubeIndices<<<getBlocks(activeVertices * 8), getThreads(activeVertices * 8)>>>(vertices_dev, verticesToCubes_dev, numVertices, verticesCompact_dev, cubes_dev, activeVertices);
	classifyCubes<<<getBlocks(numCubes), getThreads(numCubes)>>>(cubes_dev, numCubes, cubesOccupied_dev, cubeVerticesCnt_dev, vertsCountTable_dev);
	checkCudaErrorsWithLine("failed classify cubes");

	activeCubes = ThrustExScanWrapper(cubesOccupiedScan_dev, cubesOccupied_dev, numCubes);
	compactCubes<<<getBlocks(numCubes), getThreads(numCubes)>>>(cubesCompact_dev, cubesOccupied_dev, cubesOccupiedScan_dev, cubes_dev, numCubes);
	checkCudaErrorsWithLine("failed compacting cubes");

	totalVertices = ThrustExScanWrapper(cubeVerticesScan_dev, cubeVerticesCnt_dev, numCubes);
	checkCudaErrorsWithLine("thrust failed");
	printf("%d, %d\n", totalVertices, activeCubes);
	return totalVertices;
}

void Grid::cudaComputeSurface(int maxVerts, float threshold) {
	size_t num_bytes;
	cudaGraphicsMapResources(1, &cuda_vbo_resource, 0);
	cudaGraphicsResourceGetMappedPointer((void **)&triangleVertices_dev, &num_bytes, cuda_vbo_resource);
	checkCudaErrorsWithLine("failed setting up vbo");

	generateTriangles<<<getBlocks(activeCubes), getThreads(activeCubes)>>>(
		triangleVertices_dev, cubesCompact_dev, cubes_dev, cubesToVertices_dev,
		cubeVerticesScan_dev, vertices_dev,
		triTable_dev, vertsCountTable_dev,
		activeCubes, maxVerts, threshold);
	checkCudaErrorsWithLine("generate triangles failed");

	cudaGraphicsUnmapResources(1, &cuda_vbo_resource, 0);
	cudaDeviceSynchronize();
	checkCudaErrorsWithLine("failed unsetting vbo");
}

//_______________________________INIT&CLEANUP_________________________________________________________________________________

//_______________________________INIT&RESTORE KERNELS________________________________________________________________________________________________
__global__ void initVerticesKernel(GridVertex * vertices, int vCount, int * verticesToCubes, float delta, int count) {
	int fullID = min(blockIdx.y * 65535 * THREADS_CNT + blockIdx.x * THREADS_CNT + threadIdx.x, vCount - 1);
	int x, y, z;
	int tmpID = fullID;
	z = tmpID % (count + 1);
	tmpID /= count + 1;
	y = tmpID % (count + 1);
	tmpID /= count + 1;
	x = tmpID % (count + 1);
	vertices[fullID].pos = glm::vec3(x * delta, y * delta, z * delta);
	vertices[fullID].id = fullID;
#pragma unroll
	for (int i = 0; i < 8; i++) {
		verticesToCubes[fullID + vCount * i] = -1;
	}
}

__global__ void initCubesKernel(GridVertex * vertices, int * verticesToCubes, int vCount, GridCube * cubes, int cCount, int count) {
	int fullID = min(blockIdx.y * 65535 * THREADS_CNT + blockIdx.x * THREADS_CNT + threadIdx.x, cCount - 1);
	int x, y, z;
	int tmpID = fullID;
	z = tmpID % count;
	tmpID /= count;
	y = tmpID % count;
	tmpID /= count;
	x = tmpID % count;

	int vID[8];
	vID[0] = ( x   *(count+1) +  y)    * (count+1) + z;
	vID[1] = ( x   *(count+1) +  y)    * (count+1) + z+1;
	vID[2] = ( x   *(count+1) + (y+1)) * (count+1) + z+1;
	vID[3] = ( x   *(count+1) + (y+1)) * (count+1) + z;
	vID[4] = ((x+1)*(count+1) +  y)    * (count+1) + z;
	vID[5] = ((x+1)*(count+1) +  y)    * (count+1) + z+1;
	vID[6] = ((x+1)*(count+1) + (y+1)) * (count+1) + z+1;
	vID[7] = ((x+1)*(count+1) + (y+1)) * (count+1) + z;
#pragma unroll
	for (int i = 0; i < 8; i++) {
		verticesToCubes[vID[i] + vCount * i] = fullID;
	}
}

//after vertices are sorted, cube-vertices connections are corupted. So they have to be restored
__global__ void restoreCVconnectionsKernel(GridVertex * vertices, int * verticesToCubes, int vCount, int * cubesToVertices) {
	int id = min(blockIdx.y * 65535 * THREADS_CNT + blockIdx.x * THREADS_CNT + threadIdx.x, vCount - 1);
#pragma unroll
	for (int i = 0; i < 8; i++) {
		int fullId = vertices[id].id + vCount * i;
		if (verticesToCubes[fullId] >= 0) {
			cubesToVertices[verticesToCubes[fullId] * 8 + i] = id;
		}
	}
}

//_______________________________INIT&CLEANUP FUNCTIONS______________________________________________________________
void Grid::cudaRestoreCVConnections() {
	restoreCVconnectionsKernel<<<getBlocks(numVertices), getThreads(numVertices)>>>(vertices_dev, verticesToCubes_dev, numVertices, cubesToVertices_dev);
	checkCudaErrorsWithLine("failed restoring connections");
}

void Grid::initGrid() {
	initVerticesKernel<<<getBlocks(numVertices), getThreads(numVertices)>>>(vertices_dev, numVertices, verticesToCubes_dev, box.size.x / count, count);
	checkCudaErrorsWithLine("failed init vertices");

	initCubesKernel<<<getBlocks(numCubes), getThreads(numCubes)>>>(vertices_dev, verticesToCubes_dev, numVertices, cubes_dev, numCubes, count);
	checkCudaErrorsWithLine("failed init cubes");
}

void Grid::cudaInit(int pCount) {
	checkCudaErrors(cudaSetDevice(0));

	createCudaMemory(pCount);
	initGrid();

	allocateTextures();

	checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cuda_vbo_resource, vbo, cudaGraphicsRegisterFlagsNone));
}

void Grid::cudaClear() {
	cudaGraphicsUnregisterResource(cuda_vbo_resource);
	deleteCudaMemory();
}

void Grid::createCudaMemory(int pCount) {
	// the "numVertices + 1" is a hack, needed to be able to access to an imaginary vertex, to store garbage
	checkCudaErrors(cudaMalloc((void**)&vertices_dev, (numVertices + 1) * sizeof(GridVertex)));
	checkCudaErrors(cudaMalloc((void**)&cubes_dev, numCubes * sizeof(GridCube)));

	checkCudaErrors(cudaMalloc((void**)&verticesOccupied_dev, numVertices * sizeof(int)));
	checkCudaErrors(cudaMalloc((void**)&verticesOccupiedScan_dev, numVertices * sizeof(int)));
	checkCudaErrors(cudaMalloc((void**)&verticesCompact_dev, numVertices * sizeof(int)));

	checkCudaErrors(cudaMalloc((void**)&verticesToCubes_dev, numVertices * 8 * sizeof(int)));
	checkCudaErrors(cudaMalloc((void**)&cubesToVertices_dev, numCubes * 8 * sizeof(int)));

	checkCudaErrors(cudaMalloc((void**)&cubesOccupied_dev, numCubes * sizeof(int)));
	checkCudaErrors(cudaMalloc((void**)&cubesOccupiedScan_dev, numCubes * sizeof(int)));
	checkCudaErrors(cudaMalloc((void**)&cubesCompact_dev, numCubes * sizeof(int)));
	checkCudaErrors(cudaMalloc((void**)&cubeVerticesCnt_dev, numCubes * sizeof(int)));
	checkCudaErrors(cudaMalloc((void**)&cubeVerticesScan_dev, numCubes * sizeof(int)));

	checkCudaErrorsWithLine("failed mallocing");
}

void Grid::deleteCudaMemory() {
	cudaFree(vertices_dev);
	cudaFree(cubes_dev);

	cudaFree(verticesOccupied_dev);
	cudaFree(verticesOccupiedScan_dev);
	cudaFree(verticesCompact_dev);

	cudaFree(verticesToCubes_dev);
	cudaFree(cubesToVertices_dev);

	cudaFree(cubesOccupied_dev);
	cudaFree(cubesOccupiedScan_dev);
	cudaFree(cubesCompact_dev);
	cudaFree(cubeVerticesCnt_dev);
	cudaFree(cubeVerticesScan_dev);

	cudaFree(triTable_dev);
	cudaFree(vertsCountTable_dev);

	checkCudaErrorsWithLine("failed deleting memory");
}


void Grid::allocateTextures()
{
	checkCudaErrors(cudaMalloc((void **) &triTable_dev, 256*16*sizeof(GLuint)));
	checkCudaErrors(cudaMemcpy((void *)triTable_dev, (void *)trianglesTable, 256*16*sizeof(int), cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMalloc((void **) &vertsCountTable_dev, 256*sizeof(GLuint)));
	checkCudaErrors(cudaMemcpy((void *)vertsCountTable_dev, (void *)vertexCountTable, 256*sizeof(int), cudaMemcpyHostToDevice));

	checkCudaErrorsWithLine("failed mallocing textures");
}