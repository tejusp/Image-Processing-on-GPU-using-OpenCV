#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <driver_types.h>
#include <host_defines.h>
#include <opencv2/core/core.hpp>
#include <opencv2/core/gpumat.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/operations.hpp>
#include <opencv2/core/types_c.h>
#include <opencv2/gpu/gpu.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc/types_c.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector_types.h>
#include <fstream>
#include <iosfwd>
#include <ostream>
#include <sstream>
#include <string>
#include <vector>
#include <iostream>

#include "csvparser.h"

using namespace std;

using namespace cv;
using namespace cv::gpu;

struct Arguments {
	string project;
	string input;
	string output;
	int padding;
	int frames;
	string extension;
	int width;
	int height;
	int area_min;
	int area_max;
	int search_win_size;
	int blur_radius;
	int threshold_win_size;
	float threshold_ratio;
	string log;
	bool verbose;

	Arguments() :
			input("data/"), output("output.txt"), padding(7), frames(1000), extension(
					".jpg"), width(640), height(480), area_min(200), area_max(
					400), search_win_size(100), blur_radius(3), threshold_win_size(
					25), threshold_ratio(0.9), log("wormSeg.log"), verbose(true) {
	}
} cla;

int findCentroidFrom1Image(cv::Mat, int*, int*, int*);

template<typename T> string NumberToString(T pNumber) {
	ostringstream oOStrStream;
	oOStrStream << pNumber;

	return oOStrStream.str();
}

string intToFileName(string fileNameFormat, int fileNumber) {
	string temp = NumberToString(fileNumber);

	return fileNameFormat.replace(fileNameFormat.size() - temp.size(),
			temp.size(), temp);
}
void func(const float*, float*, size_t, const size_t, int, int, int&, int&);
int centroidRow = 0;
int centroidCol = 0;

void callKernel(const cv::gpu::GpuMat &src, cv::gpu::GpuMat &dst, int *pX,
		int *pY) {
	float* p = (float*) src.data;
	float* p2 = (float*) dst.data;
	func(p, p2, src.step, dst.step, src.cols, src.rows, centroidRow,
			centroidCol);
//	*pX = centroidRow;
//	*pY = centroidCol;
//	cout << "print row=" << *pX << endl;
//	cout << "print col=" << *pY << endl;
}

int cudaFindCentroid(cv::Mat src, int *pX, int *pY, int *pArea) {
	//GPU Mat... Copy from CPU memory to GPU memory...
	cv::gpu::GpuMat gpu_src(src);

	cv::gpu::GpuMat matAfterBlur;
	//Filters on GPU...
	cv::gpu::blur(gpu_src, matAfterBlur,
			Size(cla.blur_radius, cla.blur_radius));
	cv::gpu::GpuMat matAfterThreshold;
	//Convert into Binary image on GPU...
	cv::gpu::threshold(matAfterBlur, matAfterThreshold,
			int(cla.threshold_ratio * 255), 255, THRESH_BINARY_INV);

	cv::gpu::GpuMat floatMatForKernel;
	matAfterThreshold.convertTo(floatMatForKernel, CV_32FC1);
//	cout << "i'm here" << endl;

	callKernel(floatMatForKernel, gpu_src, pX, pY);
//	//Copy from GPU memory to CPU memory...
//	if (*pX) {
////		*pX = bRect.x + (bRect.width / 2);
////		*pY = bRect.y + (bRect.height / 2);
//		*pArea = 10;
//	} else {
//		*pX = -1;
//		*pY = -1;
//		*pArea = -1;
//	}
//
//	return 0;
}

int wormSegmenter() {

	fstream outputFile;

	outputFile.open(cla.output.c_str(), ios::out);

	int x = -1, y = -1, area = -1;
	int adjustX = 0, adjustY = 0;

	for (int fileNumber = 0; fileNumber < cla.frames; fileNumber++) {
		string fileName = cla.input + intToFileName("0000000", fileNumber)
				+ cla.extension;
		cv::Mat src = cv::imread(fileName, CV_LOAD_IMAGE_GRAYSCALE);

		if (!src.data) {
//			cout << endl << "Exited." << endl;
			exit(1);
		}

		if ((x == -1) && (y == -1)) {
			findCentroidFrom1Image(src, &x, &y, &area);
			src = cv::imread(fileName, CV_LOAD_IMAGE_GRAYSCALE);
			adjustX = x - (cla.search_win_size / 2);
			adjustY = y - (cla.search_win_size / 2);
		} else {
			src = src(
					cv::Rect(x - (cla.search_win_size / 2),
							y - (cla.search_win_size / 2), cla.search_win_size,
							cla.search_win_size));
			cudaFindCentroid(src, &x, &y, &area);
			if ((x > 0) && (y > 0)) {

				//std::cout << "writing file=" << fileNumber << "x=" << x << "y="					<< y << endl;

//				x += adjustX;
//				y += adjustY;
//				adjustX = x - (cla.search_win_size / 2);
//				adjustY = y - (cla.search_win_size / 2);
				x = 153;
				y = 251;
			}
		}
//		cout << "writing file=" << fileNumber << "x=" << x << "y=" << y << endl;
		outputFile << fileNumber << ", " << x << ", " << y << ", " << area
				<< endl;
	}

	outputFile.close();

	return 0;
}

int findCentroidFrom1Image(cv::Mat src, int *pX, int *pY, int *pArea) {
	// Smoothing the image.
	blur(src, src, Size(cla.blur_radius, cla.blur_radius)); //Blur radius 3 in original java worm segmenter.

	// Convert the image into binary image.
	threshold(src, src, int(cla.threshold_ratio * 255), 255, THRESH_BINARY_INV);

	// Vector for storing contour
	vector<vector<Point> > contours;

	vector<Vec4i> hierarchy;

	// Find contours in the image.
	findContours(src, contours, hierarchy, CV_RETR_CCOMP,
			CV_CHAIN_APPROX_SIMPLE);

	if (contours.size() > 0) {
		int largest_contour_index = 0;
		int largest_area = 0;

		// Iterate through each contour.
		for (int i = 0; i < contours.size(); i++) {
			//  Find the area of contour
			double a = contourArea(contours[i], false);

			if (a > largest_area) {
				largest_area = a;

				// Store the index of largest contour
				largest_contour_index = i;
			}
		}

		Rect bRect = boundingRect(contours[largest_contour_index]);

		*pX = bRect.x + (bRect.width / 2);
		*pY = bRect.y + (bRect.height / 2);
		*pArea = largest_area;
	} else {
		*pX = -1;
		*pY = -1;
		*pArea = -1;
	}

	return 0;
}

int main(int argc, char **argv) {
	int i = 0;
	//                                   file, delimiter, first_line_is_header?
	CsvParser *csvparser = CsvParser_new("example_file.csv", ",", 0);
	CsvRow *row;
	std::vector<std::vector<float> > vec;
	while ((row = CsvParser_getRow(csvparser))) {
		std::vector<float> eachLine;
		//printf("==NEW LINE==\n");
		const char **rowFields = CsvParser_getFields(row);
		for (i = 0; i < CsvParser_getNumFields(row); i++) {
			eachLine.push_back(atof(rowFields[i]));
			//printf("FIELD: %f\n", eachLine[i]);
		}
		vec.push_back(eachLine);
//		printf("\n");
		CsvParser_destroy_row(row);
	}
//	printf("test=%f\n", vec[0][1]);
//	printf("test=%f\n", vec[1][0]);
//	printf("test=%f\n", vec[2][0]);
//	printf("test=%f\n", vec[3][0]);
	CsvParser_destroy(csvparser);

	int a = wormSegmenter();
	return 0;
}

//#define arraySIZE 240
__device__ int edgesValues[100][100];
//__device__ int edgesValues[480][640];
__shared__ int counter;
__constant__ const int maxContourPoints = 300;

__global__ void funcKernel(const float* srcptr, float* dstptr, size_t srcstep,
		const size_t dststep, int cols, int rows, int* inputArray_d,
		int* outputArray_d) {

	int rowInd = blockIdx.y * blockDim.y + threadIdx.y;
	int colInd = blockIdx.x * blockDim.x + threadIdx.x;

	if (rowInd >= rows || colInd >= cols)
		return;
	const float* rowsrcptr = (const float *) (((char *) srcptr)
			+ rowInd * srcstep);
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

				edgesValues[rowInd][colInd] = 1;
			} else {
				edgesValues[rowInd][colInd] = 0;
			}

		}
	}

}

__global__ void funcKernel2(const float* srcptr, float* dstptr, size_t srcstep,
		const size_t dststep, int cols, int rows, int* inputArray_d,
		int* outputArray_d, int *a, int *b, int *c) {

	int rowInd = blockIdx.y * blockDim.y + threadIdx.y;
	int colInd = blockIdx.x * blockDim.x + threadIdx.x;
	if (rowInd >= rows || colInd >= cols)
		return;

	counter = 0;
	int maxRow = 0;
	int minRow = rows;
	int minCol = cols;
	int maxCol = 0;
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			if (edgesValues[i][j] == 1 && counter < maxContourPoints) {
				if (i < minRow) {
					minRow = i;
				}
				if (i > maxRow) {
					maxRow = i;
				}
				if (j < minCol) {
					minCol = j;
				}
				if (j > maxCol) {
					maxCol = j;
				}
				counter++;

			}
		}
	}

	int centroidRow = (minRow + maxRow) / 2;
	int centroidCol = (minCol + maxCol) / 2;
//	printf("%d,%d", centroidRow, centroidCol);
	*a = centroidRow;
	*b = centroidCol;
	*c = *a + *b;

}

int divUp(int a, unsigned int b) {
	return (a + b - 1) / b;
}

void func(const float* srcptr, float* dstptr, size_t srcstep,
		const size_t dststep, int cols, int rows, int& centroidRow,
		int& centroidCol) {
	dim3 blDim(32, 8);
	dim3 grDim(divUp(cols, blDim.x), divUp(rows, blDim.y));

	int inputArray_h[rows * cols];
	int outputArray_h[rows * cols];
	int* cRowNumber;
	int* cColNumber;

	for (int j = 0; j < rows * cols; j++) {
		inputArray_h[j] = 0;
	}

	int int_BYTES = sizeof(int);
	//allocate GPU memory

	cudaMalloc((void**) &cRowNumber, int_BYTES);
	cudaMalloc((void**) &cColNumber, int_BYTES);

	cudaMemcpy(cRowNumber, inputArray_h, int_BYTES, cudaMemcpyHostToDevice);

	funcKernel<<<grDim, blDim>>>(srcptr, dstptr, srcstep, dststep, cols, rows,
			cRowNumber, cColNumber);
//	cudaDeviceSynchronize();

	int *dev_a, *dev_b, *dev_c; // device copies of a, b, c
	int size = sizeof(int); // we need space for an integer
	// allocate device copies of a, b, c
	cudaMalloc((void**) &dev_a, size);
	cudaMalloc((void**) &dev_b, size);
	cudaMalloc((void**) &dev_c, size);
	// copy inputs to device

	funcKernel2<<<1, 1>>>(srcptr, dstptr, srcstep, dststep, cols, rows,
			cRowNumber, cColNumber, dev_a, dev_b, dev_c);

}
