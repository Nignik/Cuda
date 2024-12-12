#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>

#define CUDA_CHECK(err) \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl; \
        exit(-1); \
    }

__global__ void matmul_elem(int n, float* a, float* b, float* c)
{
	int column = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;

	if (row < n && column < n)
	{
		float dot_prod = 0.f;
		for (int i = 0; i < n; i++)
		{
			dot_prod += a[row * n + i] * b[i * n + column];
		}
		c[row * n + column] = dot_prod;
	}
}

template<int N>
__device__ int linearIndex(const int coords[N], const int dims[N]) {
	int idx = 0;
	if (N == 1) {
		idx = coords[0];
	}
	else if (N == 2) {
		idx = coords[0] * dims[1] + coords[1];
	}
	else {
		idx = coords[0] * dims[1] * dims[2] + coords[1] * dims[2] + coords[2];
	}

	return idx;
}

__device__ int computeThreadId()
{
	int threadId = threadIdx.x +
		threadIdx.y * blockDim.x +
		threadIdx.z * blockDim.x * blockDim.y +
		blockIdx.x * blockDim.x * blockDim.y * blockDim.z +
		blockIdx.y * gridDim.x * blockDim.x * blockDim.y * blockDim.z +
		blockIdx.z * gridDim.x * gridDim.y * blockDim.x * blockDim.y * blockDim.z;
	return threadId;
}

__device__ int computeGridSize()
{
	int gridSize = gridDim.x * gridDim.y * gridDim.z * blockDim.x * blockDim.y * blockDim.z;
	return gridSize;
}

__global__ void sum333(int n, float* a, float* b, float* c, float* d) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int z = blockIdx.z * blockDim.z + threadIdx.z;

	if (x < n && y < n && z < n) {
		int coords[3] = {x, y, z};
		int dims[3] = {n, n, n};
		int idx = linearIndex<3>(coords, dims);
		d[idx] = a[idx] + b[idx] + c[idx];
	}
}

__global__ void sum321(int n, float* a, float* b, float* c, float* d) {
	int totalElements = n * n * n;
	int threadId = computeThreadId();
	int gridSize = computeGridSize();

	for (int idx = threadId; idx < totalElements; idx += gridSize)
	{
		d[idx] = a[idx] + b[idx % (n*n)] + c[idx % n];
	}
}

int main() {
	constexpr int N = 3;
	float a_h[N][N][N], b_h[N][N], c_h[N], d_h[N][N][N];
	float *a_d, *b_d, *c_d, *d_d;

	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			for (int k = 0; k < N; k++) {
				a_h[i][j][k] = float(i*N*N + j*N + k);
				b_h[i][j] = float(i*N + j);
				c_h[i] = float(i);
			}
		}
	}

	CUDA_CHECK(cudaMalloc(&a_d, N*N*N*sizeof(float)));
	CUDA_CHECK(cudaMalloc(&b_d, N*N*sizeof(float)));
	CUDA_CHECK(cudaMalloc(&c_d, N*sizeof(float)));
	CUDA_CHECK(cudaMalloc(&d_d, N*N*N*sizeof(float)));

	CUDA_CHECK(cudaMemcpy(a_d, a_h, N*N*N*sizeof(float), cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(b_d, b_h, N*N*sizeof(float), cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(c_d, c_h, N*sizeof(float), cudaMemcpyHostToDevice));

	dim3 blockSize(8, 8, 8);
	dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (N + blockSize.y - 1) / blockSize.y, (N + blockSize.z - 1) / blockSize.z);
	sum321<<<gridSize, blockSize>>>(N, a_d, b_d, c_d, d_d);
	CUDA_CHECK(cudaGetLastError());
	CUDA_CHECK(cudaDeviceSynchronize());

	CUDA_CHECK(cudaMemcpy(d_h, d_d, N*N*N*sizeof(float), cudaMemcpyDeviceToHost));

	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			for (int k =  0; k < N; k++) {
				std::cout << int(d_h[i][j][k]) << ' ';
			}
			std::cout << std::endl;
		}
		std::cout << std::endl;
	}

	cudaFree(a_d);
	cudaFree(b_d);
	cudaFree(c_d);
	cudaFree(d_d);
}