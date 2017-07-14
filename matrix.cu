/**
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 */

#include <common_functions.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <host_defines.h>
#include <stdio.h>
#include <sys/time.h>
#include <iostream>
#include <ostream>

//Param
#define PRINT false

//Running params
#define CPU_MULT true
#define CUDA_MULT true
#define CUDA_MULT_TRANSPOSE true
#define CUDA_MULT_SHARED true
#define CUDA_MULT_SHARED_TRANSPOSE true

//Matrix params
#define MATRIX_SET_1 false
#define MATRIX_SIZE 1000
#define MATRIX_TYPE float
const int TILE_WIDTH = 16;

//Macros
#define ELEM(row,column,rowSize) (column+row*rowSize)

/**
 * This macro checks return value of the CUDA runtime call and exits
 * the application if the call failed.
 *
 * See cuda.h for error code descriptions.
 */
#define CHECK_CUDA_RESULT(N) {											\
	CUresult result = N;												\
	if (result != 0) {													\
		printf("CUDA call on line %d returned error %d\n", __LINE__,	\
			result);													\
		exit(1);														\
	}                                                                   \
}

using namespace std;

struct timeval tv1, tv2;
double diff;

__host__ void printMatrix(MATRIX_TYPE* matrix, int rowCount, int columnCount) {
	for (int row = 0; row < rowCount; ++row) {
		for (int column = 0; column < columnCount; ++column) {
			cout << matrix[ELEM(row, column, rowCount)] << " ";
		}
		cout << endl;
	}
}

__host__ MATRIX_TYPE * createMatrixTransposta(int rowCount, int columnCount,
		int initValue, bool colValue) {
	MATRIX_TYPE * matrix = (MATRIX_TYPE*) malloc(
			sizeof(MATRIX_TYPE) * rowCount * columnCount);

	for (int row = 0; row < rowCount; ++row) {
		for (int column = 0; column < columnCount; ++column) {
			matrix[ELEM(row, column, rowCount)] =
					colValue ? column + 1 : initValue;
		}
	}

	return matrix;
}

__host__ MATRIX_TYPE * createMatrix(int rowCount, int columnCount,
		int initValue, bool rowValue) {
	MATRIX_TYPE * matrix = (MATRIX_TYPE*) malloc(
			sizeof(MATRIX_TYPE) * rowCount * columnCount);

	for (int row = 0; row < rowCount; ++row) {
		for (int column = 0; column < columnCount; ++column) {
			matrix[ELEM(row, column, rowCount)] =
					rowValue ? row + 1 : initValue;
		}
	}

	return matrix;
}

__host__ MATRIX_TYPE * createMatrix(int rowCount, int columnCount) {
	return createMatrix(rowCount, columnCount, 1, true);
}

__host__ MATRIX_TYPE * createMatrix(int rowCount, int columnCount,
		int initValue) {
	return createMatrix(rowCount, columnCount, initValue, false);
}

__global__ void matrixCompare(bool* result, MATRIX_TYPE* matrixA,
MATRIX_TYPE * matrixB, int matrixSize) {

	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;

	if (col < matrixSize && row < matrixSize) {
		result[ELEM(row, col, matrixSize)] =
				(matrixA[ELEM(row, col, matrixSize)]
						== matrixA[ELEM(row, col, matrixSize)]);
	}

}

__host__ void multMatrix(MATRIX_TYPE * matrixA, MATRIX_TYPE * matrixB,
MATRIX_TYPE * matrixC, int matrixSize) {

	for (int row = 0; row < matrixSize; ++row) {
		for (int column = 0; column < matrixSize; ++column) {
			MATRIX_TYPE sum = 0;

			for (int k = 0; k < matrixSize; ++k) {
				sum += matrixA[ELEM(row, k, matrixSize)]
						* matrixB[ELEM(k, column, matrixSize)];
			}

			matrixC[ELEM(row, column, matrixSize)] = sum;
		}
	}
}

__global__ void multiMatrixCUDA(MATRIX_TYPE * matrixA, MATRIX_TYPE * matrixB,
MATRIX_TYPE * matrixC, int matrixSize) {

	unsigned int column = (blockDim.x * blockIdx.x) + threadIdx.x;
	unsigned int row = (blockDim.y * blockIdx.y) + threadIdx.y;

	if (column < matrixSize && row < matrixSize) {
		MATRIX_TYPE sum = 0;

		for (int k = 0; k < matrixSize; ++k) {
			sum += matrixA[ELEM(row, k, matrixSize)]
					* matrixB[ELEM(k, column, matrixSize)];
		}

		matrixC[ELEM(row, column, matrixSize)] = sum;
	}
}

__global__ void matrixTranspose(MATRIX_TYPE * matrix_in,
MATRIX_TYPE * matrix_out, int matrixSize) {

	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int column = blockIdx.x * blockDim.x + threadIdx.x;

	if (column < matrixSize && row < matrixSize) {
		matrix_out[ELEM(row, column, matrixSize)] = matrix_in[ELEM(column, row,
				matrixSize)];
	}
}

__global__ void multiMatrixCUDABTranspose(MATRIX_TYPE * matrixA,
MATRIX_TYPE * matrixBTransposed,
MATRIX_TYPE * matrixC, int matrixSize) {

	int column = (blockDim.x * blockIdx.x) + threadIdx.x;
	int row = (blockDim.y * blockIdx.y) + threadIdx.y;

	if (column < matrixSize && row < matrixSize) {

		MATRIX_TYPE sum = 0;

		for (int k = 0; k < matrixSize; ++k) {
			sum += matrixA[ELEM(row, k, matrixSize)]
					* matrixBTransposed[ELEM(column, k, matrixSize)];
		}

		matrixC[ELEM(row, column, matrixSize)] = sum;
	}
}

__global__ void matMultTileCuda(const float *A, const float *B, float *C, int N){
	__shared__ float a_tile[TILE_WIDTH][TILE_WIDTH], b_tile[TILE_WIDTH][TILE_WIDTH];

	int qtd_tiles = N/TILE_WIDTH + (N%TILE_WIDTH==0?0:1);

    int i = blockDim.y * blockIdx.y + threadIdx.y;
    int j = blockDim.x * blockIdx.x + threadIdx.x;

	int offset;

	float sum = 0.0;




		for (int tile_ind = 0; tile_ind < qtd_tiles; ++tile_ind) {
			offset = tile_ind*TILE_WIDTH;
			if(i<N && offset+threadIdx.x< N){
				a_tile[threadIdx.y][threadIdx.x] = A[ELEM(i, offset+threadIdx.x, N)];
			} else{
				a_tile[threadIdx.y][threadIdx.x] = 0.0;
			}

			if(threadIdx.y+offset<N && j< N){
				b_tile[threadIdx.y][threadIdx.x] = B[ELEM(threadIdx.y+offset, j, N)];
			} else{
				b_tile[threadIdx.y][threadIdx.x] = 0.0;
			}

			__syncthreads();

			for (int k = 0; k < TILE_WIDTH; ++k) {
				sum += a_tile[threadIdx.y][k]*b_tile[k][threadIdx.x];
			}

			__syncthreads();
		}

		if(i<N && j<N) C[ELEM(i,j,N)] = sum;


}

__global__ void multiMatrixCUDAShared(MATRIX_TYPE* matrixA,
MATRIX_TYPE* matrixB, MATRIX_TYPE* matrixC, int matrixSize) {

	__shared__ int matrixSharedA[TILE_WIDTH * TILE_WIDTH];
	__shared__ int matrixSharedB[TILE_WIDTH * TILE_WIDTH];

	//Row and column of element to calculate
	int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
	int col = blockIdx.x * TILE_WIDTH + threadIdx.x;

	MATRIX_TYPE sum = 0;
	int tileCount = ((matrixSize - 1) / TILE_WIDTH) + 1;

	//Iterate tiles to compute the sum
	for (int tileIndex = 0; tileIndex < tileCount; ++tileIndex) {
		int colA = tileIndex * TILE_WIDTH + threadIdx.x;
		//Collaborative loading of A and B tiles into shared memory
		if (row < matrixSize && colA < matrixSize)
			matrixSharedA[ELEM(threadIdx.y, threadIdx.x, TILE_WIDTH)] =
					matrixA[row * matrixSize + colA];
		else
			matrixSharedA[ELEM(threadIdx.y, threadIdx.x, TILE_WIDTH)] = 0;

		int rowB = tileIndex * TILE_WIDTH + threadIdx.y;
		if (col < matrixSize && rowB < matrixSize)
			matrixSharedB[ELEM(threadIdx.y, threadIdx.x, TILE_WIDTH)] =
					matrixB[(rowB) * matrixSize + col];
		else
			matrixSharedB[ELEM(threadIdx.y, threadIdx.x, TILE_WIDTH)] = 0;

		__syncthreads();

		for (int k = 0; k < TILE_WIDTH; ++k)
			sum += matrixSharedA[ELEM(threadIdx.y, k, TILE_WIDTH)]
					* matrixSharedB[ELEM(k, threadIdx.x, TILE_WIDTH)];

		__syncthreads();

	}
	if (row < matrixSize && col < matrixSize) {
		matrixC[ELEM(row, col, matrixSize)] = sum;
	}
}

__global__ void multiMatrixCUDASharedTransposed(MATRIX_TYPE* matrixA,
MATRIX_TYPE* matrixBTransposed, MATRIX_TYPE* matrixC, int matrixSize) {

	__shared__ int matrixSharedA[TILE_WIDTH * TILE_WIDTH];
	__shared__ int matrixSharedB[TILE_WIDTH * TILE_WIDTH];

	//Row and column of element to calculate
	int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
	int col = blockIdx.x * TILE_WIDTH + threadIdx.x;

	MATRIX_TYPE sum = 0;
	int tileCount = ((matrixSize - 1) / TILE_WIDTH) + 1;

	//Iterate tiles to compute the sum
	for (int tileIndex = 0; tileIndex < tileCount; ++tileIndex) {
		int colA = tileIndex * TILE_WIDTH + threadIdx.x;
		//Collaborative loading of A and B tiles into shared memory
		if (row < matrixSize && colA < matrixSize)
			matrixSharedA[ELEM(threadIdx.y, threadIdx.x, TILE_WIDTH)] =
					matrixA[row * matrixSize + colA];
		else
			matrixSharedA[ELEM(threadIdx.y, threadIdx.x, TILE_WIDTH)] = 0;

		int rowB = tileIndex * TILE_WIDTH + threadIdx.y;
		if (col < matrixSize && rowB < matrixSize)
			matrixSharedB[ELEM(threadIdx.y, threadIdx.x, TILE_WIDTH)] =
					matrixBTransposed[(col) * matrixSize + rowB];
		else
			matrixSharedB[ELEM(threadIdx.y, threadIdx.x, TILE_WIDTH)] = 0;

		__syncthreads();

		for (int k = 0; k < TILE_WIDTH; ++k)
			sum += matrixSharedA[ELEM(threadIdx.y, k, TILE_WIDTH)]
					* matrixSharedB[ELEM(k, threadIdx.x, TILE_WIDTH)];

		__syncthreads();

	}
	if (row < matrixSize && col < matrixSize) {
		matrixC[ELEM(row, col, matrixSize)] = sum;
	}
}

int getSPcores(cudaDeviceProp devProp) {
	int cores = 0;
	int mp = devProp.multiProcessorCount;
	switch (devProp.major) {
	case 2: // Fermi
		if (devProp.minor == 1)
			cores = mp * 48;
		else
			cores = mp * 32;
		break;
	case 3: // Kepler
		cores = mp * 192;
		break;
	case 5: // Maxwell
		cores = mp * 128;
		break;
	case 6: // Pascal
		if (devProp.minor == 1)
			cores = mp * 128;
		else if (devProp.minor == 0)
			cores = mp * 64;
		else
			printf("Unknown device type\n");
		break;
	default:
		printf("Unknown device type\n");
		break;
	}
	return cores;
}

int getBlockSize() {
	int nDevices;
	cudaGetDeviceCount(&nDevices);
	for (int i = 0; i < nDevices; i++) {
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, i);
		return prop.major < 2 ? 16 : 32;
	}
	return 16;
}

void getDeviceInfo() {
	int nDevices;
	cudaGetDeviceCount(&nDevices);
	for (int i = 0; i < nDevices; i++) {
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, i);
		printf("Device Number: %d\n", i);
		printf("  Device name: %s\n", prop.name);
		printf("  Memory Clock Rate (KHz): %d\n", prop.memoryClockRate);
		printf("  Memory Bus Width (bits): %d\n", prop.memoryBusWidth);
		printf("  Peak Memory Bandwidth (GB/s): %f\n\n",
				2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);
		cout << "ECC enabled: " << prop.ECCEnabled << endl;
		cout << "Warp size: " << prop.warpSize << endl;
		cout << "Shared Memory Per Block: " << prop.sharedMemPerBlock << endl;
		cout << "Shared Memory Per Multiprocessor: "
				<< prop.sharedMemPerMultiprocessor << endl;
		cout << "Global Memory : " << prop.totalGlobalMem << endl;
		cout << "Concurrent Kernels : " << prop.concurrentKernels << endl;
		cout << "Integrated : " << prop.integrated << endl;
		cout << "Multiprocessor Count : " << prop.multiProcessorCount << endl;
		cout << "Cuda cores: " << getSPcores(prop) << endl;
		cout << "Concurrent Managed Access: " << prop.concurrentManagedAccess
				<< endl;
		cout << "Max grid Size: " << prop.maxGridSize << endl;
		cout << "Max thread Dim: " << prop.maxThreadsDim << endl;
		cout << "Max thread per block: " << prop.maxThreadsPerBlock << endl;
		cout << "Max thread per multiprocessor: "
				<< prop.maxThreadsPerMultiProcessor << endl;
		cout << "Active warps: "
				<< prop.maxThreadsPerMultiProcessor / prop.warpSize << endl;
	}
}

double diffTime(timeval& tv1, timeval& tv2) {
	return (double) ((tv2.tv_usec - tv1.tv_usec) / 1000
			+ (double) (tv2.tv_sec - tv1.tv_sec) * 1000);
}

void printMatrix(int matrixSize, MATRIX_TYPE* h_matrixC) {
	gettimeofday(&tv1, NULL);
	printMatrix(h_matrixC, matrixSize, matrixSize);
	gettimeofday(&tv2, NULL);
	diff = diffTime(tv1, tv2);

	cout << "Print time:" << diff << endl;
}

int main(int argc, char **argv) {

	//getDeviceInfo();

	cudaSetDevice(0);

	int matrixSize = MATRIX_SIZE;
	cudaError_t err;

	MATRIX_TYPE * h_matrixC_CPU;

#if MATRIX_SET_1 == true

	MATRIX_TYPE * h_matrixA = createMatrix(matrixSize, matrixSize, 1);
	MATRIX_TYPE * h_matrixB = createMatrix(matrixSize, matrixSize, 1);
	MATRIX_TYPE * h_matrixC = createMatrix(matrixSize, matrixSize, 0);

#else
	MATRIX_TYPE * h_matrixA = createMatrix(matrixSize, matrixSize);
	MATRIX_TYPE * h_matrixB = createMatrixTransposta(matrixSize, matrixSize, 1,
			true);
	MATRIX_TYPE * h_matrixC = createMatrix(matrixSize, matrixSize, 0);

#endif

	//Set size of blocks and threads

	//int blockSize = getBlockSize();
	int blockSize = TILE_WIDTH;

	dim3 thread(blockSize, blockSize, 1);
	int gridSizeX = matrixSize / thread.x;
	gridSizeX += (matrixSize % thread.x) == 0 ? 0 : 1;

	int gridSizeY = matrixSize / thread.y;
	gridSizeY += (matrixSize % thread.y) == 0 ? 0 : 1;

	dim3 grid(gridSizeX, gridSizeY, 1);

	cout << "Grid: " << grid.x << " - " << grid.y << " Thread: " << thread.x
			<< " - " << thread.y << endl;

//Alocação das matrizes no dispositivo
	MATRIX_TYPE * d_matrixA, *d_matrixB, *d_matrixC, *d_matrixBTransposed,
			*d_matrixC_CPU;

	cudaMalloc((void **) &d_matrixA,
			sizeof(MATRIX_TYPE) * matrixSize * matrixSize);
	cudaMalloc((void **) &d_matrixB,
			sizeof(MATRIX_TYPE) * matrixSize * matrixSize);
	cudaMalloc((void **) &d_matrixC,
			sizeof(MATRIX_TYPE) * matrixSize * matrixSize);

//Copia dos dados para o dispositivo
	cudaMemcpy(d_matrixA, h_matrixA,
			sizeof(MATRIX_TYPE) * matrixSize * matrixSize,
			cudaMemcpyHostToDevice);
	cudaMemcpy(d_matrixB, h_matrixB,
			sizeof(MATRIX_TYPE) * matrixSize * matrixSize,
			cudaMemcpyHostToDevice);
	cudaMemcpy(d_matrixC, h_matrixC,
			sizeof(MATRIX_TYPE) * matrixSize * matrixSize,
			cudaMemcpyHostToDevice);

#if PRINT == true
	cout << "A: " << endl;
	printMatrix(h_matrixA, matrixSize, matrixSize);

	cout << "B: " << endl;
	printMatrix(h_matrixB, matrixSize, matrixSize);
#endif

#if CPU_MULT == true
//CPU MULT
	h_matrixC_CPU = createMatrix(matrixSize, matrixSize, 0);

	gettimeofday(&tv1, NULL);
	multMatrix(h_matrixA, h_matrixB, h_matrixC_CPU, matrixSize);
	gettimeofday(&tv2, NULL);
	diff = diffTime(tv1, tv2);

	cout << "CPU Mult time: " << diff << endl;

	cudaMalloc((void **) &d_matrixC_CPU,
			sizeof(MATRIX_TYPE) * matrixSize * matrixSize);

	cudaMemcpy(d_matrixC_CPU, h_matrixC_CPU,
			sizeof(MATRIX_TYPE) * matrixSize * matrixSize,
			cudaMemcpyHostToDevice);

#if PRINT == true
	printMatrix(matrixSize, h_matrixC_CPU);
#endif
#endif

#if CUDA_MULT == true
//CUDA MULT

	gettimeofday(&tv1, NULL);
	multiMatrixCUDA<<<grid, thread>>>(d_matrixA, d_matrixB, d_matrixC,
			matrixSize);
	cudaDeviceSynchronize();
	gettimeofday(&tv2, NULL);
	diff = diffTime(tv1, tv2);

	cout << "CUDA Mult time: " << diff << endl;

	cudaMemcpy(h_matrixC, d_matrixC,
			sizeof(MATRIX_TYPE) * matrixSize * matrixSize,
			cudaMemcpyDeviceToHost);

#if PRINT == true
	printMatrix(h_matrixC, matrixSize, matrixSize);
#endif
#endif

#if CUDA_MULT_TRANSPOSE == true

//CUDA MULT B Transpose

	cudaMalloc((void **) &d_matrixBTransposed,
			sizeof(MATRIX_TYPE) * matrixSize * matrixSize);

	matrixTranspose<<<grid, thread>>>(d_matrixB, d_matrixBTransposed,matrixSize);

	cudaDeviceSynchronize();

	gettimeofday(&tv1, NULL);
	multiMatrixCUDABTranspose<<<grid, thread>>>(d_matrixA, d_matrixBTransposed,
			d_matrixC, matrixSize);

	cudaDeviceSynchronize();

	gettimeofday(&tv2, NULL);
	diff = diffTime(tv1, tv2);

	cout << "CUDA Mult B Transpose time: " << diff << endl;

	err = cudaGetLastError();
	if (err != cudaSuccess)
		printf("Error: %s\n", cudaGetErrorString(err));

	cudaMemcpy(h_matrixC, d_matrixC,
			sizeof(MATRIX_TYPE) * matrixSize * matrixSize,
			cudaMemcpyDeviceToHost);

#if PRINT == true
	printMatrix(h_matrixC, matrixSize, matrixSize);
#endif
#endif

#if CUDA_MULT_SHARED
//CUDA MULT Shared

	gettimeofday(&tv1, NULL);

	multiMatrixCUDAShared<<<grid, thread>>>(d_matrixA, d_matrixB, d_matrixC,matrixSize);

	cudaDeviceSynchronize();

	gettimeofday(&tv2, NULL);
	diff = diffTime(tv1, tv2);
	cout << "CUDA Mult Shared time: " << diff << endl;

	err = cudaGetLastError();
	if (err != cudaSuccess)
		printf("Error: %s\n", cudaGetErrorString(err));

	cudaMemcpy(h_matrixC, d_matrixC,
			sizeof(MATRIX_TYPE) * matrixSize * matrixSize,
			cudaMemcpyDeviceToHost);

#if PRINT == true
	printMatrix(matrixSize, h_matrixC);
#endif
#endif

#if CUDA_MULT_SHARED_TRANSPOSE
//CUDA MULT Shared

	cudaMalloc((void **) &d_matrixBTransposed,
			sizeof(MATRIX_TYPE) * matrixSize * matrixSize);

	matrixTranspose<<<grid, thread>>>(d_matrixB, d_matrixBTransposed,matrixSize);

	cudaDeviceSynchronize();

	gettimeofday(&tv1, NULL);

	multiMatrixCUDASharedTransposed<<<grid, thread>>>(d_matrixA, d_matrixBTransposed, d_matrixC,
			matrixSize);

	cudaDeviceSynchronize();

	gettimeofday(&tv2, NULL);
	diff = diffTime(tv1, tv2);
	cout << "CUDA Mult Shared transposed time: " << diff << endl;

	err = cudaGetLastError();
	if (err != cudaSuccess)
		printf("Error: %s\n", cudaGetErrorString(err));

	cudaMemcpy(h_matrixC, d_matrixC,
			sizeof(MATRIX_TYPE) * matrixSize * matrixSize,
			cudaMemcpyDeviceToHost);

	if (h_matrixC_CPU != NULL) {
		cout << "Comparing result:" << endl;

		bool * h_result = (bool *) malloc(
				sizeof(bool) * matrixSize * matrixSize);
		bool * d_result;
		cudaMalloc((void **) &d_result, sizeof(bool) * matrixSize * matrixSize);

		matrixCompare<<<grid, thread>>>(d_result,d_matrixC,d_matrixC_CPU, matrixSize);

		cudaDeviceSynchronize();

		cudaMemcpy(h_result, d_result, sizeof(bool) * matrixSize * matrixSize,
				cudaMemcpyDeviceToHost);

		for (int row = 0; row < matrixSize; ++row) {
			for (int column = 0; column < matrixSize; ++column) {
				if (!h_result[ELEM(row, column, matrixSize)]) {
					cout << "Error on position: " << row << "," << column
							<< endl;
				}
			}
		}

	}

#if PRINT == true
	printMatrix(matrixSize, h_matrixC);

	cudaMemcpy(h_matrixB, d_matrixBTransposed,
			sizeof(MATRIX_TYPE) * matrixSize * matrixSize,
			cudaMemcpyDeviceToHost);

	cout << "Matrix B transposed:" << endl;
	printMatrix(matrixSize, h_matrixB);

#endif
#endif

	free(h_matrixA);
	free(h_matrixB);
	free(h_matrixC);

	cudaFree(d_matrixA);
	cudaFree(d_matrixB);
	cudaFree(d_matrixC);
	cudaFree(d_matrixBTransposed);
	cudaFree(d_matrixC_CPU);

	return 0;
}

