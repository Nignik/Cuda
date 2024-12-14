#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>

#define CUDA_CHECK(err) \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl; \
        exit(-1); \
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

template<int N>
__device__ void dot(float* A, float* B, float* res) {
  int totalElements = N;
  int threadId = computeThreadId();
  int gridSize = computeGridSize();

	float local_sum = 0.f;
  for (int idx = threadId; idx < totalElements; idx += gridSize) {
    local_sum += A[idx] * B[idx];
  }

  atomicAdd(res, local_sum);
}

template<int N>
__global__ void test_dot(float* A, float* B, float* c) {
  dot<N>(A, B, c);
}

template<int N, int M, int P>
__device__ void mul(float* A, float* B, float* C) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < N && col < P) {
    float dot_prod = 0.0f;

    for (int i = 0; i < M; i++) {
      dot_prod += A[row * M + i] * B[i * P + col];
    }

    C[row * P + col] = dot_prod;
  }
}

template<int N, int M, int P>
__global__ void test_mul(float* A, float* B, float* C) {
  mul<N, M, P>(A, B, C);
}

template<int N, int M>
void print_matrix(float A[N][M]) {
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < M; j++) {
      std::cout << A[i][j] << ' ';
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
}

int main() {
	constexpr int N = 3;
  constexpr int M = 4;
  constexpr int P = 3;
	float h_A[N][M], h_B[M][P], h_C[N][P];
	float *d_A, *d_B, *d_C;

	for (int i = 0; i < N; i++) {
    for (int j = 0; j < M; j++) {
      h_A[i][j] = float(i * M + j);
    }
  }

  for (int i = 0; i < M; i++) {
    for (int j = 0; j < P; j++) {
      h_B[i][j] = float(i * P + j);
    }
  }


	CUDA_CHECK(cudaMalloc(&d_A, N*M*sizeof(float)));
	CUDA_CHECK(cudaMalloc(&d_B, M*P*sizeof(float)));
	CUDA_CHECK(cudaMalloc(&d_C, N*P*sizeof(float)));

	CUDA_CHECK(cudaMemcpy(d_A, h_A, N*M*sizeof(float), cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(d_B, h_B, M*P*sizeof(float), cudaMemcpyHostToDevice));

	dim3 blockSize(16, 16);
	dim3 gridSize((P + blockSize.x - 1) / blockSize.x, (N + blockSize.y - 1) / blockSize.y);
	test_mul<N, M, P><<<gridSize, blockSize>>>(d_A, d_B, d_C);
	CUDA_CHECK(cudaGetLastError());
	CUDA_CHECK(cudaDeviceSynchronize());

	CUDA_CHECK(cudaMemcpy(&h_C, d_C, N*P*sizeof(float), cudaMemcpyDeviceToHost));

  print_matrix<N, M>(h_A);
  print_matrix<M, P>(h_B);
  print_matrix<N, P>(h_C);

	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
}