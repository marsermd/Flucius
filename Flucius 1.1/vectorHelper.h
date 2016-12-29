#ifndef VECTOR_HELPER_H
#define VECTOR_HELPER_H
#include <thrust/device_vector.h>

void PushBack(thrust::host_vector<int>& vector, int i);
void Clear(thrust::host_vector<int>& vector);
void Clear(thrust::device_vector<int>& vector);
void Copy(thrust::host_vector<int>& vector, thrust::device_vector<int>& vector_dev);

#endif