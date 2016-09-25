#include "vectorHelper.h"

void PushBack(thrust::host_vector<int>& vector, int i) {
	vector.push_back(i);
}

void Clear(thrust::host_vector<int>& vector) {
	vector.clear();
}

void Clear(thrust::device_vector<int>& vector) {
	vector.clear();
}

void Copy(thrust::host_vector<int>& vector, thrust::device_vector<int>& vector_dev) {
	thrust::copy(vector.begin(), vector.end(), vector_dev.begin());
}