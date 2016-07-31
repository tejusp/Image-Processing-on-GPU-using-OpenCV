#include <cuda_runtime_api.h>
#include <device_functions.hpp>
#include <device_launch_parameters.h>
#include <driver_types.h>
#include <host_defines.h>
#include <opencv2/core/core.hpp>
#include <opencv2/core/cuda_devptrs.hpp>
#include <opencv2/core/gpumat.hpp>
#include <opencv2/gpu/device/common.hpp>
#include <stdio.h>
#include <vector_types.h>
#include <ostream>
#include <string>
#include <iostream>
#include <typeinfo>
using namespace std;
using std::cout;

__global__ void funcKernel(const float* srcptr, float* dstptr, size_t srcstep,
		const size_t dststep, int cols, int rows, int counter) {

	int rowInd = blockIdx.y * blockDim.y + threadIdx.y;
	int colInd = blockIdx.x * blockDim.x + threadIdx.x;
	if (rowInd >= rows || colInd >= cols)
		return;
	const float* rowsrcptr = (const float *) (((char *) srcptr)
			+ rowInd * srcstep);
	float* rowdstPtr = (float *) (((char *) dstptr) + rowInd * dststep);
	float val = rowsrcptr[colInd];

	if ((rowInd > 2 && rowInd < (rows - 2))
			&& (colInd > 2 && colInd < (cols - 2))) {
		if (val == 255) {
			const float* rowsrcptrNxt = (const float *) (((char *) srcptr)
					+ (rowInd + 1) * srcstep);
			const float* rowsrcptrPrev = (const float *) (((char *) srcptr)
					+ (rowInd - 1) * srcstep);
			if (rowsrcptrPrev[colInd - 1] == 0 || rowsrcptrPrev[colInd] == 0
					|| rowsrcptrPrev[colInd + 1] == 0
					|| rowsrcptr[colInd - 1] == 0 || rowsrcptr[colInd - 1] == 0
					|| rowsrcptrNxt[colInd - 1] == 0
					|| rowsrcptrNxt[colInd] == 0
					|| rowsrcptrNxt[colInd + 1] == 0) {

				printf(
						"\nat row=%d col=%d ;val=%f, rowsrcptr[colInd-1]=%f, rowsrcptr[colInd+1]=%f,rowsrcptrNxt =%f, counter=%d",
						rowInd, colInd, val, rowsrcptr[colInd - 1],
						rowsrcptr[colInd + 1], rowsrcptrNxt[colInd],
						counter * rowInd);

			}

		}
	}
	if ((int) val % 90 == 0)
		rowdstPtr[colInd] = -1;
	else {
		float acos_val = acos(val);
		rowdstPtr[colInd] = acos_val;
	}
}

int divUp(int a, int b) {
	return (a + b - 1) / b;
}

//extern "C"
//{
void func(const float* srcptr, float* dstptr, size_t srcstep,
		const size_t dststep, int cols, int rows) {
	dim3 blDim(32, 8);
	dim3 grDim(divUp(cols, blDim.x), divUp(rows, blDim.y));
	int counter = 0;
	std::cout << "calling kernel from func\n";
	funcKernel<<<grDim, blDim>>>(srcptr, dstptr, srcstep, dststep, cols, rows,
			counter);
	std::cout << "done with kernel call\n";
	cudaDeviceSynchronize();
}
//}
