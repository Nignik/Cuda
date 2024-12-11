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

int main() {
	float *a_h, *b_h, *c_h;
	float *a_d, *b_d, *c_d;
	int N = 3;

	a_h = new float[N * N];
	b_h = new float[N * N];
	c_h = new float[N * N];

	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			a_h[i * N + j] = float(i * N + j);
			b_h[i * N + j] = float(i * N + j);
		}
	}

	CUDA_CHECK(cudaMalloc((void**) &a_d, N*N*sizeof(float)));
	CUDA_CHECK(cudaMalloc((void**) &b_d, N*N*sizeof(float)));
	CUDA_CHECK(cudaMalloc((void**) &c_d, N*N*sizeof(float)));

	CUDA_CHECK(cudaMemcpy(a_d, a_h, N*N*sizeof(float), cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(b_d, b_h, N*N*sizeof(float), cudaMemcpyHostToDevice));

	dim3 blockSize(16, 16);
	dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (N + blockSize.y - 1) / blockSize.y);
	matmul_elem<<<gridSize, blockSize>>>(N, a_d, b_d, c_d);
	CUDA_CHECK(cudaGetLastError());
	CUDA_CHECK(cudaDeviceSynchronize());

	CUDA_CHECK(cudaMemcpy(c_h, c_d, N*N*sizeof(float), cudaMemcpyDeviceToHost));

	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			std::cout << int(c_h[i * N + j]) << ' ';
		}
		std::cout << std::endl;
	}

	cudaFree(a_d);
	cudaFree(b_d);
	cudaFree(c_d);

	delete[] a_h;
	delete[] b_h;
	delete[] c_h;
}