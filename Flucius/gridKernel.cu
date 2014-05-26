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

inline __device__ __host__ glm::vec3 lerp(glm::vec3 a, glm::vec3 b, float t)
{
	return a + t*(b-a);
}

//_______________________________DEVICE VARIABLES________________________________________________________________________________________________________

int *edgeTable_dev;
int *triTable_dev;
int *vertsCountTable_dev;

texture<GLuint, 1, cudaReadModeElementType> edgeTex;
texture<GLuint, 1, cudaReadModeElementType> triTex;
texture<GLuint, 1, cudaReadModeElementType> vertsCountTex;

int *verticesOccupied_dev = 0;
int *verticesOccupiedScan_dev = 0;
int *verticesCompact_dev = 0;

int *cubesOccupied_dev = 0;
int *cubesOccupiedScan_dev = 0;
int *cubesCompact_dev = 0;
int *cubeVerticesCnt_dev = 0;
int *cubeVerticesScan_dev = 0;

Vertex *triangleVertices_dev = 0; 

struct cudaGraphicsResource *cuda_vbo_resource;

cudaError_t cudaStatus;

//_______________________________CUDA PART___________________________________________________________________________________________________________

int ThrustExScanWrapper(int *output, int *input, unsigned int numElements)
{
	thrust::exclusive_scan(
		thrust::device_ptr<int>(input),
		thrust::device_ptr<int>(input + numElements),
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
	//if(fullID > 5508000)
	//	vertices[fullID].value = 1.2f;

	vertices[fullID].normal = glm::vec3(0);
}

__constant__ float R4 = PARTICLE_R2 * PARTICLE_R2;
__global__ void calcGridKernel(Particle * particles, GridVertex * vertices, int pCount, int vCount, int cnt, int * partitions, int * partitionIdx,
							   int maxItems, float r, int size, int ttlCount)
{
	int fullID = threadIdx.x + blockIdx.x * THREADS_CNT + blockIdx.y * 65535 * THREADS_CNT;

	int particleID = fullID / cnt;
	int neighbourID = (fullID % cnt) / maxItems;
	int elementID = (fullID % cnt) % maxItems;

	if (particleID >= pCount || neighbourID >= NEIGHBOURS_3D) return;
	int x,y,z;
	glm::vec3 p = particles[particleID].pos;
	x = GET_CLOSEST_POS(p.x, r);
	y = GET_CLOSEST_POS(p.y, r);
	z = GET_CLOSEST_POS(p.z, r);

	z += neighbourID % 3 - 1;
	neighbourID /= 3;
	y += neighbourID % 3 - 1;
	neighbourID /= 3;
	x += neighbourID % 3 - 1;

	int partitionID = __mul24(size, __mul24(size, x)) + __mul24(size, y) + z;
	int baseVertexID = partitions[partitionID];
	int vertexID = baseVertexID + elementID;
	if (partitionID >= ttlCount || vertexID >= vCount || partitionIdx[vertexID] != partitionIdx[baseVertexID]) return;

	glm::vec3 diff = vertices[vertexID].pos - particles[particleID].pos;
	float dist2 = glm::dot(diff, diff);
	//float backDist4 = dist2 > PARTICLE_R2 * 2 ? 0 : -__log2f((dist2 / 2) * (dist2 / 2) / PARTICLE_R2) / 1.3f;
	//float backDist4 = dist2 > PARTICLE_R2 * 2 ? 0 : __fdiv_rd(__fadd_rz(__cosf(__fmul_rz(__fdiv_rz(dist2, __fmul_rz(PARTICLE_R2, 4)), PI)), 1), 1.7f);
	float backDist4 = __fmul_rz(R4, __frcp_rz((max(__fmul_rz(dist2, dist2), 0.0000001f))));
	if (backDist4 > 0.01) {
		vertices[vertexID].normal += diff * backDist4;
		vertices[vertexID].value += backDist4;
	}
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

__global__ void calcCubeIndices(GridVertex * vertices, int * verticesCompact, GridCube * cubes, int vCompactCount) {
	int id = threadIdx.x + blockIdx.x * THREADS_CNT + blockIdx.y * 65535 * THREADS_CNT;
	int listId = id / vCompactCount;
	id = verticesCompact[id % vCompactCount];
	int cube = vertices[id].cubeID[listId];
	if (listId < 8 && cube >= 0) {
		int val = 1 << listId;
		cubes[cube].cubeID |= val;
	}
}

__global__ void classifyCubes(GridCube * cubes, int cCount, int * cubesOccupied, int *cubeVerticesCnt)
{
	int id = min(threadIdx.x + blockIdx.x * THREADS_CNT + blockIdx.y * 65535 * THREADS_CNT, cCount - 1);
	int cubeIndex = cubes[id].cubeID;
	cubesOccupied[id] = cubeIndex != 0 && cubeIndex != 255;
	cubeVerticesCnt[id] = tex1Dfetch(vertsCountTex, cubeIndex);
}

__global__ void compactCubes(int *cubesCompact, int *cubesOccupied, int *cubesOccupiedScan, GridCube *cubes, int cCount)
{
	int id = threadIdx.x + blockIdx.x * THREADS_CNT + blockIdx.y * 65535 * THREADS_CNT;
	if (id < cCount && cubesOccupied[id]) {
		cubesCompact[cubesOccupiedScan[id]] = id;
	}
}


__device__ void vertexInterp2(float threshold, GridVertex v0, GridVertex v1, glm::vec3 &p, glm::vec3 &n)
{
	float t = (threshold - v0.value) / (v1.value - v0.value);
	p = lerp(v0.pos, v1.pos, t);
	n = lerp(v0.normal, v1.normal, t);
	n = glm::normalize(n);
}

__global__ void generateTriangles(Vertex *triangleVertices, int *cubesCompact, GridCube *cubes, int *cubeVerticesScan, GridVertex *vertices, int activeCubes, int maxVerts, float threshold)
{
	int id = min(threadIdx.x + blockIdx.x * THREADS_CNT + blockIdx.y * 65535 * THREADS_CNT, activeCubes - 1);
	id = cubesCompact[id];
	glm::vec3 vertlist[12];
	glm::vec3 normlist[12];

	GridCube currentCube = cubes[id];

	GridVertex curVertices[8] = {
		vertices[currentCube.vID[0]],
		vertices[currentCube.vID[1]],	
		vertices[currentCube.vID[2]],	
		vertices[currentCube.vID[3]],	
		vertices[currentCube.vID[4]],	
		vertices[currentCube.vID[5]],	
		vertices[currentCube.vID[6]],	
		vertices[currentCube.vID[7]]	
	};

	vertexInterp2(threshold, curVertices[0], curVertices[1], vertlist[0], normlist[0]);
	vertexInterp2(threshold, curVertices[1], curVertices[2], vertlist[1], normlist[1]);
	vertexInterp2(threshold, curVertices[2], curVertices[3], vertlist[2], normlist[2]);
	vertexInterp2(threshold, curVertices[3], curVertices[0], vertlist[3], normlist[3]);

	vertexInterp2(threshold, curVertices[4], curVertices[5], vertlist[4], normlist[4]);
	vertexInterp2(threshold, curVertices[5], curVertices[6], vertlist[5], normlist[5]);
	vertexInterp2(threshold, curVertices[6], curVertices[7], vertlist[6], normlist[6]);
	vertexInterp2(threshold, curVertices[7], curVertices[4], vertlist[7], normlist[7]);

	vertexInterp2(threshold, curVertices[0], curVertices[4], vertlist[8], normlist[8]);
	vertexInterp2(threshold, curVertices[1], curVertices[5], vertlist[9], normlist[9]);
	vertexInterp2(threshold, curVertices[2], curVertices[6], vertlist[10], normlist[10]);
	vertexInterp2(threshold, curVertices[3], curVertices[7], vertlist[11], normlist[11]);


	int numVerts = tex1Dfetch(vertsCountTex, currentCube.cubeID);
	for (int i = 0; i < numVerts; i++)
	{
		GLuint edge = tex1Dfetch(triTex, currentCube.cubeID * 16 + i);
		int index = cubeVerticesScan[id] + i;

		if (index < maxVerts)
		{
			for (int j = 0; j < 3; j++) {
				triangleVertices[index].position[j] = vertlist[edge][j];
				triangleVertices[index].normal[j] = normlist[edge][j];
			}
			triangleVertices[index].texcoord[0] = 0;
			triangleVertices[index].texcoord[1] = 0;
		}
	}
}

//_______________________________LAUNCHERS___________________________________________________________________________________________________

int Grid::cudaCalcGrid(Particle * particles_dev, int pCount) {	
	resetValuesKernel<<<getBlocks(numVertices), getThreads(numVertices)>>>(vertices_dev, numVertices);
	checkCudaErrorsWithLine("failed reseting vertices values");

	int cnt = NEIGHBOURS_3D * partition3D->maxItemsPerPartition;
	//cnt = ((cnt - 1) / THREADS_CNT + 1) * THREADS_CNT;
	calcGridKernel<<<getBlocks(cnt * pCount), getThreads(cnt * pCount)>>>(particles_dev, vertices_dev, pCount, numVertices, cnt, 
		partition3D->partitions_dev, partition3D->partitionIdx_dev, partition3D->maxItemsPerPartition, partition3D->r,
		partition3D->countx, partition3D->ttlCount);
	checkCudaErrorsWithLine("failed calculating grid");
	return 0;
}

int activeCubes, totalVertices;
int Grid::cudaAnalyzeCubes(float threshold) {
	classifyVertices<<<getBlocks(numVertices), getThreads(numVertices)>>>(vertices_dev, verticesOccupied_dev, numVertices, threshold);
	int activeVertices = ThrustExScanWrapper(verticesOccupiedScan_dev, verticesOccupied_dev, numVertices);
	compactVertices<<<getBlocks(numVertices), getThreads(numVertices)>>>(verticesCompact_dev, verticesOccupied_dev, verticesOccupiedScan_dev, numVertices);
	checkCudaErrorsWithLine("failed classify vertices");

	resetCubes<<<getBlocks(numCubes), getThreads(numCubes)>>>(cubes_dev, numCubes);
	calcCubeIndices<<<getBlocks(activeVertices * 8), getThreads(activeVertices * 8)>>>(vertices_dev, verticesCompact_dev, cubes_dev, activeVertices);
	classifyCubes<<<getBlocks(numCubes), getThreads(numCubes)>>>(cubes_dev, numCubes, cubesOccupied_dev, cubeVerticesCnt_dev);
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

	generateTriangles<<<getBlocks(activeCubes), getThreads(activeCubes)>>>(triangleVertices_dev, cubesCompact_dev, cubes_dev, cubeVerticesScan_dev, vertices_dev, activeCubes, maxVerts, threshold);
	checkCudaErrorsWithLine("generate triangles failed");

	cudaGraphicsUnmapResources(1, &cuda_vbo_resource, 0);
	cudaDeviceSynchronize();
	checkCudaErrorsWithLine("failed unsetting vbo");
}

//_______________________________INIT&CLEANUP_________________________________________________________________________________

//_______________________________INIT&RESTORE KERNELS________________________________________________________________________________________________
__global__ void initVerticesKernel(GridVertex * vertices, int vCount, float delta, int count) {
	int fullID = min(blockIdx.y * 65535 * THREADS_CNT + blockIdx.x * THREADS_CNT + threadIdx.x, vCount - 1);
	int x, y, z;
	int tmpID = fullID;
	z = tmpID % (count + 1);
	tmpID /= count + 1;
	y = tmpID % (count + 1);
	tmpID /= count + 1;
	x = tmpID % (count + 1);
	vertices[fullID].pos = glm::vec3(x * delta, y * delta, z * delta);
#pragma unroll
	for (int i = 0; i < 8; i++) {
		vertices[fullID].cubeID[i] = -1;
	}
}

__global__ void initCubesKernel(GridVertex * vertices, GridCube * cubes, int cCount, int count) {
	int fullID = min(blockIdx.y * 65535 * THREADS_CNT + blockIdx.x * THREADS_CNT + threadIdx.x, cCount - 1);
	int x, y, z;
	int tmpID = fullID;
	z = tmpID % count;
	tmpID /= count;
	y = tmpID % count;
	tmpID /= count;
	x = tmpID % count;

	cubes[fullID].vID[0] = (x*(count+1)     +y)*(count+1)   +z;
	cubes[fullID].vID[1] = (x*(count+1)     +y)*(count+1)   +z+1;
	cubes[fullID].vID[2] = (x*(count+1)    +(y+1))*(count+1)+z+1;
	cubes[fullID].vID[3] = (x*(count+1)    +(y+1))*(count+1)+z;
	cubes[fullID].vID[4] = ((x+1)*(count+1) +y)*(count+1)   +z;
	cubes[fullID].vID[5] = ((x+1)*(count+1) +y)*(count+1)   +z+1;
	cubes[fullID].vID[6] = ((x+1)*(count+1)+(y+1))*(count+1)+z+1;
	cubes[fullID].vID[7] = ((x+1)*(count+1)+(y+1))*(count+1)+z;
#pragma unroll
	for (int i = 0; i < 8; i++) {
		vertices[cubes[fullID].vID[i]].cubeID[i] = fullID;
	}
}

//after vertices are sorted, cube-vertices connections are corupted. So they have to be by changing vID
__global__ void restoreCVconnectionsKernel(GridVertex * vertices, int vCount, GridCube * cubes) {
	int fullID = min(blockIdx.y * 65535 * THREADS_CNT + blockIdx.x * THREADS_CNT + threadIdx.x, vCount - 1);
#pragma unroll
	for (int i = 0; i < 8; i++) {
		if (vertices[fullID].cubeID[i] >= 0) {
			cubes[vertices[fullID].cubeID[i]].vID[i] = fullID;
		}
	}
}


//_______________________________INIT&CLEANUP FUNCTIONS______________________________________________________________
void Grid::cudaRestoreCVConnections() {
	restoreCVconnectionsKernel<<<getBlocks(numVertices), getThreads(numVertices)>>>(vertices_dev, numVertices, cubes_dev);
	checkCudaErrorsWithLine("failed restoring connections");
}

void Grid::initGrid() {
	initVerticesKernel<<<getBlocks(numVertices), getThreads(numVertices)>>>(vertices_dev, numVertices, box.size.x / count, count);
	checkCudaErrorsWithLine("failed init vertices");

	initCubesKernel<<<getBlocks(numCubes), getThreads(numCubes)>>>(vertices_dev, cubes_dev, numCubes, count);
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

	checkCudaErrors(cudaMalloc((void**)&cubesOccupied_dev, numCubes * sizeof(int)));
	checkCudaErrors(cudaMalloc((void**)&cubesOccupiedScan_dev, numCubes * sizeof(int)));
	checkCudaErrors(cudaMalloc((void**)&cubesCompact_dev, numCubes * sizeof(int)));
	checkCudaErrors(cudaMalloc((void**)&cubeVerticesCnt_dev, numCubes * sizeof(int)));
	checkCudaErrors(cudaMalloc((void**)&cubeVerticesScan_dev, numCubes * sizeof(int)));

	checkCudaErrorsWithLine("failed mallocing cubes");
}

void Grid::deleteCudaMemory() {
	cudaFree(vertices_dev);
	cudaFree(cubes_dev);

	cudaFree(verticesOccupied_dev);
	cudaFree(verticesOccupiedScan_dev);
	cudaFree(verticesCompact_dev);

	cudaFree(cubesOccupied_dev);
	cudaFree(cubesOccupiedScan_dev);
	cudaFree(cubesCompact_dev);
	cudaFree(cubeVerticesCnt_dev);
	cudaFree(cubeVerticesScan_dev);

	cudaFree(edgeTable_dev);
	cudaFree(triTable_dev);
	cudaFree(vertsCountTable_dev);
}


void Grid::allocateTextures()
{
	checkCudaErrors(cudaMalloc((void **) &edgeTable_dev, 256*sizeof(GLuint)));
	checkCudaErrors(cudaMemcpy((void *)edgeTable_dev, (void *)edgesTable, 256*sizeof(int), cudaMemcpyHostToDevice));
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindUnsigned);
	checkCudaErrors(cudaBindTexture(0, edgeTex, edgeTable_dev, channelDesc));

	checkCudaErrors(cudaMalloc((void **) &triTable_dev, 256*16*sizeof(GLuint)));
	checkCudaErrors(cudaMemcpy((void *)triTable_dev, (void *)trianglesTable, 256*16*sizeof(int), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaBindTexture(0, triTex, triTable_dev, channelDesc));

	checkCudaErrors(cudaMalloc((void **) &vertsCountTable_dev, 256*sizeof(GLuint)));
	checkCudaErrors(cudaMemcpy((void *)vertsCountTable_dev, (void *)vertexCountTable, 256*sizeof(int), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaBindTexture(0, vertsCountTex, vertsCountTable_dev, channelDesc));
}